from __future__ import annotations
import ctypes, os, mmap, tempfile, pathlib, array, threading, contextlib, sys, subprocess, struct, re
assert sys.platform != 'win32'
from tinygrad.device import BufferSpec, Compiled, Allocator, Compiler, Program, TinyELF, CompileError
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.uop.ops import Ops, UOp
from tinygrad.helpers import getenv, round_up, mv_address, to_mv, cpu_objdump, system, DEBUG, suppress_finalizing, Target, unwrap
from tinygrad.renderer.cstyle import ClangRenderer, wmma_args
from tinygrad.codegen.opt import tc
from tinygrad.runtime.autogen import libc, qcom_dsp
if getenv("IOCTL"): import extra.dsp.run # noqa: F401 # pylint: disable=unused-import

from tinygrad.uop.ops import PatternMatcher, UPat

# NOTE: this just increases readability of the generated code
dsp_string = PatternMatcher([
  (UPat(Ops.CONST, (dtypes.int8, dtypes.uint8), name="x"), lambda ctx,x: str(x.val)),
])

class DSPRenderer(ClangRenderer):
  has_threads = False
  buffer_suffix = " restrict __attribute__((align_value(128)))"
  kernel_typedef = "__attribute__((noinline)) void"
  string_rewrite = dsp_string+ClangRenderer.string_rewrite
  type_map = { **ClangRenderer.type_map, dtypes.uint64: "unsigned long long", dtypes.int64: "long long" }
  code_for_op = {k:v for k,v in ClangRenderer.code_for_op.items() if k != Ops.SQRT}

  def __init__(self, target:Target): self.target, self.compiler, self.tensor_cores = target, DSPCompiler(), tc.hexagon_v65

  # V6_vrmpyub/V6_vrmpybusv (HVX): D(int32x32) = C(int32x32) + dot4(A(u8x4 broadcast scalar), B(u8x128, 32
  # groups of 4)) in one instruction -- no warp/lane cooperation needed (tensor_cores' threads=1), unlike
  # every other backend's WMMA. The accumulator-add form always needs a real HVX_Vector C; a plain (not
  # "_acc") vrmpy variant only exists for the from-zero case, so we always pass C through.
  #
  # KNOWN PERFORMANCE ISSUE: correct but currently slower than plain scalar code on real hardware,
  # because devectorizer2's do_stack_wmma (codegen/__init__.py) unconditionally decomposes every WMMA's
  # accumulator into per-element scalar loads/stores before rendering -- the right behavior for every
  # other backend (each GPU thread only ever holds a few scalar elements of a warp-distributed
  # fragment), but wrong here: Hexagon's "32 elements" is one HVX vector register that a single thread
  # (threads=1, no warp) processes atomically, and it should stay vector-resident across the reduction
  # loop instead of being rebuilt from 32 scalar reads on every accumulate call. A real fix needs the
  # generic devectorizer (or the accumulator's axis-ordering/layout in postrange.py's _apply_tc_opt) to
  # recognize single-thread/vector-native tensor cores as a distinct case -- out of scope here.
  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None):
    prefix = list(prefix or [])
    for name, _, dtype_in, dtype_out, _, _, upcast_sizes in wmma_args(uops):
      dstr_a, dstr_b, dstr_c = (self._render_dtype(dt, sz, AddrSpace.REG) for dt, sz in
                                 zip([dtype_in, dtype_in, dtype_out], upcast_sizes))
      builtin = "__builtin_HEXAGON_V6_vrmpyub_acc_128B" if dtype_in == dtypes.uint8 else "__builtin_HEXAGON_V6_vrmpybusv_acc_128B"
      prefix.append(f"""static inline {dstr_c} __{name}({dstr_a} a, {dstr_b} b, {dstr_c} c) {{
  unsigned int a_scalar; __builtin_memcpy(&a_scalar, &a, 4);
  return {builtin}(c, b, a_scalar);
}}""")
    return super().render_kernel(function_name, kernel, bufs, uops, prefix)

  def _render_defines(self, uops) -> list[str]:
    return ['''/* DSP boilerplate */ struct dcvs_v2_req { int type; int _pad; _Bool dcvs_enable; char dcvs_option; _Bool set_latency; int latency;
      _Bool set_dcvs_params; short _pad2; char target_corner; char min_corner; char max_corner; int _pad3[3];};''','int HAP_power_set(void*, void*);',
      'typedef union { struct { void *pv; unsigned int len; } buf; struct { int fd; unsigned int offset; } dma; } remote_arg;',
      'void* HAP_mmap(void *addr, int len, int prot, int flags, int fd, long offset);', 'int HAP_munmap(void *addr, int len);',
      'unsigned long long HAP_perf_get_time_us(void);'] + super()._render_defines(uops)

  def _render_entry(self, function_name:str, bufs:list[tuple[str,tuple[UOp,bool]]]) -> str:
    msrc = ['int entry(unsigned long long handle, unsigned int sc, remote_arg* pra) {',
            'struct dcvs_v2_req req = {.type=7, .dcvs_enable=0, .set_latency=1, .latency=100, .set_dcvs_params=1, .target_corner = 6 /* TURBO */};',
            'HAP_power_set((void*)handle, (void*)&req);']
    msrc += ['if ((sc>>24) != 2) return 0;']
    msrc += [f'{self._render_dtype(b[1][0].dtype) if b[1][0].addrspace == AddrSpace.ALU else "int"} sz_or_val_{i} = '
             f'*({self._render_dtype(b[1][0].dtype) if b[1][0].addrspace == AddrSpace.ALU else "int"}*)((char*)pra[0].buf.pv+{i*8});'
             for i,b in enumerate(bufs)]
    msrc += [f'int off{i} = ((int*)pra[1].buf.pv)[{i}];' for i,b in enumerate(bufs) if b[1][0].addrspace == AddrSpace.GLOBAL]
    msrc += [f'void *buf_{i} = HAP_mmap(0,sz_or_val_{i},3,0,pra[{i+3}].dma.fd,0)+off{i};'
             for i,b in enumerate(bufs) if b[1][0].addrspace == AddrSpace.GLOBAL]
    msrc += ["unsigned long long start = HAP_perf_get_time_us();"]
    fbufs = [(f'buf_{i}' if b[1][0].addrspace == AddrSpace.GLOBAL else f'sz_or_val_{i}') for i,b in enumerate(bufs)]
    msrc += [f"{function_name}({', '.join(fbufs)});"]
    msrc += ["*(unsigned long long *)(pra[2].buf.pv) = HAP_perf_get_time_us() - start;"]
    msrc += [f'HAP_munmap(buf_{i}, sz_or_val_{i});' for i,b in enumerate(bufs) if b[1][0].addrspace == AddrSpace.GLOBAL]
    msrc += ["return 0; }"]
    return '\n'.join(msrc)

  def supported_dtypes(self): return {d for d in super().supported_dtypes() if d not in dtypes.fp8s+(dtypes.bfloat16,)}

def rpc_sc(method=0, ins=0, outs=0, fds=0): return (method << 24) | (ins << 16) | (outs << 8) | fds
def rpc_prep_args(ins=None, outs=None, in_fds=None):
  ins, outs, in_fds = ins or list(), outs or list(), in_fds or list()

  pra = (qcom_dsp.union_remote_arg * (len(ins) + len(outs) + len(in_fds)))()
  fds = (ctypes.c_int32 * (len(ins) + len(outs) + len(in_fds)))(*([-1] * (len(ins) + len(outs))), *in_fds)
  attrs = (ctypes.c_uint32 * (len(ins) + len(outs) + len(in_fds)))(*([0] * (len(ins) + len(outs))), *([1] * (len(in_fds))))

  for i, mv in enumerate(ins + outs): pra[i].buf.pv, pra[i].buf.len = ctypes.c_void_p(mv_address(mv) if mv.nbytes > 0 else 0), mv.nbytes
  return pra, fds, attrs, (ins, outs)

class DSPProgram(Program['DSPDevice']):
  def __init__(self, dev:DSPDevice, obj:TinyELF): self.dev, self.lib, self.signature = dev, obj.lib, obj.signature

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
    if len(bufs) >= 16: raise RuntimeError(f"Too many buffers to execute: {len(bufs)}")

    pra, fds, attrs, _ = rpc_prep_args(ins=[var_vals_mv:=memoryview(bytearray((len(bufs)+len(vals))*8)), off_mv:=memoryview(bytearray(len(bufs)*4))],
                                       outs=[timer:=memoryview(bytearray(8)).cast('Q')], in_fds=[b.share_info.fd for b in bufs])
    for i,b in enumerate(bufs): struct.pack_into('i', var_vals_mv, i*8, b.size)
    for i,(v,(_,_,dt,_)) in enumerate(zip(vals, self.signature[len(bufs):]), start=len(bufs)): struct.pack_into(unwrap(dt.fmt), var_vals_mv, i*8, v)
    off_mv.cast('I')[:] = array.array('I', tuple(b.offset for b in bufs))
    self.dev.exec_lib(self.lib, rpc_sc(method=2, ins=2, outs=1, fds=len(bufs)), pra, fds, attrs)
    return timer[0] / 1e6

class DSPBuffer:
  def __init__(self, va_addr:int, size:int, share_info, offset:int=0):
    self.va_addr, self.size, self.share_info, self.offset = va_addr, size, share_info, offset

class DSPAllocator(Allocator['DSPDevice']):
  def _alloc(self, size:int, options:BufferSpec):
    if getenv("MOCKDSP") or getenv("HEXSIM"): fd, share_info, flags = -1, None, mmap.MAP_SHARED|mmap.MAP_ANONYMOUS
    else:
      b = qcom_dsp.ION_IOC_ALLOC(self.dev.ion_fd, len=size, align=0x200, heap_id_mask=1<<qcom_dsp.ION_SYSTEM_HEAP_ID, flags=qcom_dsp.ION_FLAG_CACHED)
      fd, flags = (share_info:=qcom_dsp.ION_IOC_SHARE(self.dev.ion_fd, handle=b.handle)).fd, mmap.MAP_SHARED
    return DSPBuffer(libc.mmap(0, size, mmap.PROT_READ|mmap.PROT_WRITE, flags, fd, 0), size, share_info, offset=0)

  @suppress_finalizing
  def _free(self, opaque:DSPBuffer, options:BufferSpec):
    libc.munmap(opaque.va_addr, opaque.size)
    if opaque.share_info is not None:
      os.close(opaque.share_info.fd)
      qcom_dsp.ION_IOC_FREE(self.dev.ion_fd, handle=opaque.share_info.handle)

  def _as_buffer(self, src:DSPBuffer) -> memoryview: return to_mv(src.va_addr, src.size)
  def _copyin(self, dest:DSPBuffer, src:memoryview): ctypes.memmove(dest.va_addr, mv_address(src), src.nbytes)
  def _copyout(self, dest:memoryview, src:DSPBuffer): ctypes.memmove(mv_address(dest), src.va_addr, dest.nbytes)
  def _offset(self, buf, size:int, offset:int): return DSPBuffer(buf.va_addr+offset, size, buf.share_info, buf.offset+offset)

class DSPCompiler(Compiler):
  def __init__(self, mock:bool=False):
    self.mock, compiler_args = mock, "--target=hexagon -mcpu=hexagonv65 -fuse-ld=lld -nostdlib -mhvx=v65 -mhvx-length=128b"
    if mock: self.args = f"-static {compiler_args}"
    else:
      # Generate link script to pass into clang. Aligning all used sections to 4k fixes invoke problem.
      sections = ['text', 'rela.plt', 'rela.dyn', 'plt', 'data', 'bss', 'hash', 'dynamic',
                  'got', 'got.plt', 'dynsym', 'dynstr', 'symtab', 'shstrtab', 'strtab']
      sections_link = '\n'.join([f'.{n} : ALIGN(4096) {{ *(.{n}) }}' for n in sections])
      with tempfile.NamedTemporaryFile(delete=False) as self.link_ld:
        self.link_ld.write(f"SECTIONS {{ . = 0x0; {sections_link}\n /DISCARD/ : {{ *(.note .note.* .gnu.hash .comment) }} }}".encode())
        self.link_ld.flush()

      self.args = f"-shared {compiler_args} -T{self.link_ld.name}"

    super().__init__(None if mock else "compile_dsp")

  def __del__(self):
    if not self.mock: os.unlink(self.link_ld.name)

  def compile(self, src:str) -> bytes:
    # TODO: remove file write. sadly clang doesn't like the use of /dev/stdout here
    with tempfile.NamedTemporaryFile(delete=True) as f:
      system(f"{getenv('CC','clang')} {self.args} -O2 -Wall -Werror -fno-stack-protector -x c -fPIC " +
             f"-ffreestanding -nostdlib - -o {f.name}", input=src.encode())
      return pathlib.Path(f.name).read_bytes()

  def disassemble(self, lib:bytes): return cpu_objdump(lib, "llvm-objdump")


class DSPDevice(Compiled):
  def __init__(self, device:str=""):
    if getenv("HEXSIM"): super().__init__(device, DSPAllocator(self), [HexagonSimRenderer], HexagonSimProgram)
    elif getenv("MOCKDSP"): super().__init__(device, DSPAllocator(self), [MockDSPRenderer], MockDSPProgram)
    else:
      self.ion_fd = os.open('/dev/ion', os.O_RDONLY)
      super().__init__(device, DSPAllocator(self), [DSPRenderer], DSPProgram)
      fastrpc_shell = memoryview(bytearray(pathlib.Path('/dsp/cdsp/fastrpc_shell_3').read_bytes()))
      self.shell_buf = self.allocator.alloc(round_up(fastrpc_shell.nbytes, 0x1000), BufferSpec(nolru=True))
      ctypes.memmove(self.shell_buf.va_addr, mv_address(fastrpc_shell), fastrpc_shell.nbytes)

      self.init_dsp()
      RPCListener(self).start()

  def open_lib(self, lib):
    self.binded_lib, self.binded_lib_off = lib, 0
    fp = "file:///tinylib?entry&_modver=1.0&_dom=cdsp\0"
    pra, _, _, _ = rpc_prep_args(ins=[memoryview(array.array('I', [len(fp), 0xff])), memoryview(bytearray(fp.encode()))],
                                 outs=[o1:=memoryview(bytearray(0x8)), o2:=memoryview(bytearray(0xff))])
    qcom_dsp.FASTRPC_IOCTL_INVOKE(self.rpc_fd, handle=0, sc=rpc_sc(method=0, ins=2, outs=2), pra=pra)
    if o1.cast('i')[1] < 0: raise RuntimeError(f"Cannot open lib: {o2.tobytes().decode()}")
    return o1.cast('I')[0]

  def close_lib(self, handle):
    pra, _, _, _ = rpc_prep_args(ins=[memoryview(array.array('I', [handle, 0xff]))], outs=[memoryview(bytearray(0x8)), memoryview(bytearray(0xff))])
    qcom_dsp.FASTRPC_IOCTL_INVOKE(self.rpc_fd, handle=0, sc=rpc_sc(method=1, ins=1, outs=2), pra=pra)

  def exec_lib(self, lib, sc, args, fds, attrs):
    def _exec_lib():
      handle = self.open_lib(lib)
      qcom_dsp.FASTRPC_IOCTL_INVOKE_ATTRS(self.rpc_fd, fds=fds, attrs=attrs, inv=qcom_dsp.struct_fastrpc_ioctl_invoke(handle=handle, sc=sc, pra=args))
      self.close_lib(handle)
    try: _exec_lib()
    except (OSError, PermissionError):
      # DSP might ask for a connection reset or just fail with operation not permitted, try to reset connection.
      self.init_dsp()
      try: _exec_lib()
      except (OSError, PermissionError) as e: raise RuntimeError(e)

  def init_dsp(self):
    if hasattr(self, 'rpc_fd'):
      with contextlib.suppress(OSError):
        qcom_dsp.FASTRPC_IOCTL_INVOKE(self.rpc_fd, handle=4, sc=rpc_sc(method=2, ins=0, outs=0)) # pylint: disable=access-member-before-definition
      os.close(self.rpc_fd) # pylint: disable=access-member-before-definition

    self.rpc_fd: int = os.open('/dev/adsprpc-smd', os.O_RDONLY | os.O_NONBLOCK)
    qcom_dsp.FASTRPC_IOCTL_GETINFO(self.rpc_fd, 3)
    qcom_dsp.FASTRPC_IOCTL_CONTROL(self.rpc_fd, req=0x3)
    qcom_dsp.FASTRPC_IOCTL_INIT(self.rpc_fd, flags=0x1, file=self.shell_buf.va_addr, filelen=self.shell_buf.size, filefd=self.shell_buf.share_info.fd)
    qcom_dsp.FASTRPC_IOCTL_INVOKE(self.rpc_fd, handle=3, sc=rpc_sc(method=3, ins=0, outs=0))

class RPCListener(threading.Thread):
  def __init__(self, device:DSPDevice):
    super().__init__()
    self.device, self.daemon = device, True

  def run(self):
    # Setup initial request arguments.
    context, status, TINYFD = 0, 0xffffffff, 0xffff
    req_args, _, _, _ = rpc_prep_args(ins=[msg_send:=memoryview(bytearray(0x10)).cast('I'), out_buf:=memoryview(bytearray(0x10000)).cast('I')],
                                      outs=[msg_recv:=memoryview(bytearray(0x10)).cast('I'), in_buf:=memoryview(bytearray(0x10000)).cast('I')])
    req_args[1].buf.len = 0

    while True:
      # Update message request and send it.
      msg_send[:] = array.array('I', [context, status, req_args[1].buf.len, in_buf.nbytes])

      try: qcom_dsp.FASTRPC_IOCTL_INVOKE(self.device.rpc_fd, handle=0x3, sc=0x04020200, pra=req_args)
      except OSError: continue # retry

      context, inbufs, outbufs = msg_recv[0], ((sc:=msg_recv[2]) >> 16) & 0xff, (msg_recv[2] >> 8) & 0xff

      in_ptr, out_ptr, objs = mv_address(in_buf), mv_address(out_buf), []
      for i in range(inbufs + outbufs):
        obj_ptr = round_up(in_ptr + 4, 8) if i < inbufs else round_up(out_ptr + 4, 8)
        objs.append(to_mv(obj_ptr, obj_size:=to_mv(in_ptr, 4).cast('I')[0]))
        if i < inbufs: in_ptr = obj_ptr + obj_size
        else:
          to_mv(out_ptr, 4).cast('I')[0] = obj_size
          out_ptr = obj_ptr + obj_size
          in_ptr += 4

      in_args, out_args = objs[:inbufs], objs[inbufs:]
      req_args[1].buf.len = out_ptr - mv_address(out_buf)

      status = 0 # reset status, will set if error
      if sc == 0x20200: pass # greating
      elif sc == 0x13050100: # open
        try: out_args[0].cast('I')[0] = TINYFD if (name:=in_args[3].tobytes()[:-1].decode()) == "tinylib" else os.open(name, os.O_RDONLY)
        except OSError: status = 1
      elif sc == 0x3010000:
        if (fd:=in_args[0].cast('I')[0]) != TINYFD: os.close(fd)
      elif sc == 0x9010000: # seek
        if (fd:=in_args[0].cast('I')[0]) == TINYFD:
          assert in_args[0].cast('I')[2] == qcom_dsp.APPS_STD_SEEK_SET, "Supported only SEEK_SET"
          res, self.device.binded_lib_off = 0, in_args[0].cast('I')[1]
        else: res = os.lseek(fd, in_args[0].cast('I')[1], in_args[0].cast('I')[2])
        status = 0 if res >= 0 else res
      elif sc == 0x4010200: # read
        if (fd:=in_args[0].cast('I')[0]) == TINYFD:
          buf = self.device.binded_lib[self.device.binded_lib_off:self.device.binded_lib_off+in_args[0].cast('I')[1]]
          self.device.binded_lib_off += len(buf)
        else: buf = os.read(fd, in_args[0].cast('I')[1])
        out_args[1][:len(buf)] = buf
        out_args[0].cast('I')[0:2] = array.array('I', [len(buf), int(len(buf) == 0)])
      elif sc == 0x1f020100: # stat
        stat = os.stat(in_args[1].tobytes()[:-1].decode())
        out_stat = qcom_dsp.struct_apps_std_STAT.from_address(mv_address(out_args[0]))
        for f in out_stat._real_fields_: out_stat.__setattr__(f[0], int(getattr(stat, f"st_{f[0]}", 0)))
      elif sc == 0x2010100: # mmap
        st = qcom_dsp.FASTRPC_IOCTL_MMAP(self.device.rpc_fd, fd=-1, flags=in_args[0].cast('I')[2], vaddrin=0, size=in_args[0].cast('Q')[3])
        out_args[0].cast('Q')[0:2] = array.array('Q', [0, st.vaddrout])
      else: raise RuntimeError(f"Unknown op: {sc=:X}")

# ***** mock DSP *****

mockdsp_boilerplate = '''/* DSP boilerplate */ static long syscall(long r0, long r1, long r2, long r3, long r4, long r5, long r6) {
long retval; __asm__ volatile("r0 = %1; r1 = %2; r2 = %3; r3 = %4; r4 = %5; r5 = %6; r6 = %7; trap0(#1); %0 = r0" : "=r" (retval)
  : "r" (r0), "r" (r1), "r" (r2), "r" (r3), "r" (r4), "r" (r5), "r" (r6) : "r0", "r1", "r2", "r3", "r4", "r5", "r6"); return retval; }
static int read(int fd, void* buf, int len) {{ return syscall(fd, (long)buf, len, 0, 0, 0, 63); }}
static int write(int fd, void* buf, int len) {{ return syscall(fd, (long)buf, len, 0, 0, 0, 64); }}
static int exit(int ret) {{ return syscall(ret, 0, 0, 0, 0, 0, 93); }}
static unsigned int inscount(void) {{ unsigned int ret; __asm__ volatile(".word 0x6a15c000; %0 = R0" : "=r" (ret) : : "r0"); return ret; }}
static void *mmap2(void *addr, unsigned int length, int prot, int flags, int fd, unsigned long offset) {{
return (void*)syscall((long)addr, length, prot, flags, fd, offset, 222); }}'''

class MockDSPRenderer(DSPRenderer):
  def __init__(self, target:Target): self.target, self.compiler, self.tensor_cores = target, DSPCompiler(mock=True), tc.hexagon_v65
  def _render_defines(self, uops) -> list[str]: return ClangRenderer._render_defines(self, uops)
  def _render_entry(self, function_name:str, bufs:list[tuple[str,tuple[UOp,bool]]]) -> str:
    # https://gpages.juszkiewicz.com.pl/syscalls-table/syscalls.html
    # control register 21 is HEX_REG_QEMU_INSN_CNT, 0x6a15c000 loads it
    msrc = [mockdsp_boilerplate, 'void _start(void) {']
    for i,b in enumerate(bufs):
      if b[1][0].addrspace == AddrSpace.GLOBAL:
        sz = b[1][0].max_numel()*b[1][0].dtype.itemsize
        # for loop for big reads
        msrc.append(f"void *buf{i} = mmap2(0, {sz}, 3, 0x21, -1, 0); for(int rd = 0; rd < {sz}; rd += read(0, buf{i}+rd, {sz}-rd));")
      else:
        msrc.append(f"{self._render_dtype(b[1][0].dtype)} val{i}; read(0, &val{i}, {b[1][0].dtype.itemsize});")
    msrc.append("unsigned int st = inscount();")
    params = [(f'(void*)buf{i}' if b[1][0].addrspace == AddrSpace.GLOBAL else f'val{i}') for i,b in enumerate(bufs)]
    msrc.append(f"{function_name}({', '.join(params)});")
    msrc.append("unsigned int et = inscount() - st; write(1, &et, sizeof(et));")
    for i,b in enumerate(bufs):
      if b[1][0].addrspace == AddrSpace.GLOBAL: msrc.append(f"write(1, buf{i}, {b[1][0].max_numel()*b[1][0].dtype.itemsize});")
    msrc.append('exit(0); }')
    return '\n'.join(msrc)

class MockDSPProgram(Program[DSPDevice]):
  def __init__(self, dev:DSPDevice, obj:TinyELF): self.lib, self.signature = obj.lib, obj.signature
  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
    with tempfile.NamedTemporaryFile(suffix=".out") as dsp_lib:
      dsp_lib.write(self.lib)
      dsp_lib.flush()
      os.chmod(dsp_lib.name, 0o0777)
      proc = subprocess.run(["qemu-hexagon-static", *(['-strace'] if DEBUG >= 5 else []), dsp_lib.name],
        input=b''.join([bytes(to_mv(x.va_addr, x.size)) for x in bufs] +
                       [struct.pack(unwrap(dt.fmt), x) for x,(_,_,dt,_) in zip(vals, self.signature[len(bufs):])]),
        stdout=subprocess.PIPE, check=True)
    offset = 4
    for x in bufs:
      to_mv(x.va_addr, x.size)[:] = proc.stdout[offset:offset+x.size]
      offset += x.size
    assert offset == len(proc.stdout)
    return struct.unpack("I", proc.stdout[0:4])[0] / 1e9  # pretend it's 1 Ghz, but this is an inscount, not a time

# ***** hexagon-sim DSP (BEAM-search timing via Qualcomm's own instruction-set simulator) *****
#
# MOCKDSP's qemu-hexagon-static path times candidates via QEMU's inscount() pseudo-register --
# a raw instruction count, not remotely cycle-accurate, and blind to Hexagon-specific pipeline,
# vector-unit, or memory-hierarchy effects (see scripts/android/tinygrad_hexagon_bridge/README.md
# in the onnx-simplifier repo's "Removing TVM as a transport dependency"/qemu-vs-hexagon-sim note
# for the motivating case: real-hardware speed for a fixed kernel *shape* flipped between faster
# and slower than a baseline purely from cache/channel-count effects instruction count can't see).
#
# hexagon-sim is Qualcomm's own instruction-set simulator (ships in the Hexagon SDK's
# HEXAGON_Tools/*/Tools/bin/hexagon-sim) with a PMU-derived total-cycle count ("Pcycles=", printed
# once at process exit). Its default mode is a fast functional-only estimate not meaningfully
# better than instruction counting, but its `--timing` mode (used below) is a real
# pipeline/dual-issue/cache-hierarchy model: confirmed empirically (see the README section this
# lands with) to report an ~28x cycle difference between two kernels with the IDENTICAL
# instruction count, differing only in whether their memory access pattern stays cache-resident
# or not -- exactly the class of effect raw instruction counting is structurally blind to.
#
# Reading a cycle-counter register live from inside a standalone-sim binary doesn't work (the
# PCYCLE control register pair reads back 0 in this mode -- confirmed empirically in this
# project's separate hexagon_sim_harness.py work), so cycles are measured the same way that
# harness does: compile the SAME kernel wrapper twice, once calling the kernel body once and once
# calling it twice (REPEAT=1 vs REPEAT=2), run both under hexagon-sim, and take the *difference*
# in each run's total Pcycles. The simulator is deterministic, so this exactly isolates the cost
# of one kernel invocation and cancels the fixed process-startup/tear-down overhead.
#
# Kernel *inputs* are zero-initialized static buffers, not real data copied from the caller's
# DSPBuffers: cycle count for a fixed-control-flow kernel (no data-dependent branches -- true of
# every kernel this project generates, conv/gemm with static loop bounds) doesn't depend on the
# data values, only on the shapes/loop-bounds already baked into the generated source. This lets
# HexagonSimCompiler.compile() do the (expensive, ~1s) real hexagon-clang + hexagon-sim round trip
# once per distinct kernel source and cache it via the normal Compiler.compile_cached() path,
# instead of re-running the simulator on every __call__.

HEXSIM_ARCH = getenv("HEXSIM_ARCH", "v73")  # matches this project's real device (Snapdragon/Hexagon v73), not DSPCompiler's v65 baseline
HEXSIM_CLOCK_HZ = 1_000_000_000  # placeholder nominal clock (Hexagon v73 cDSP is close to 1 GHz) -- only the *relative* ranking of
                                  # returned times matters for BEAM; this scales Pcycles into a plausible-looking float, nothing more.

class HexagonSimRenderer(DSPRenderer):
  def __init__(self, target:Target): self.target, self.compiler, self.tensor_cores = target, HexagonSimCompiler(), tc.hexagon_v65
  def _render_defines(self, uops) -> list[str]: return ClangRenderer._render_defines(self, uops)
  def _render_entry(self, function_name:str, bufs:list[tuple[str,tuple[UOp,bool]]]) -> str:
    # Plain hosted main() (hexagon-sim's standalone-OS mode has real libc) -- no raw trap0 dance
    # needed, unlike MockDSPRenderer's qemu-bare-metal entry. Buffers are static, zero-filled,
    # 128B-aligned (HVX vector width) arrays sized from the UOp shapes the renderer already knows
    # at render time; see the module docstring above for why real data isn't needed here.
    # `write()`-ing one byte of each output buffer at the end forces the compiler to treat the
    # whole kernel body as having an externally-observable side effect -- without this, -O1 sees
    # no consumer of the static buffers this synthetic main() writes and dead-code-eliminates the
    # entire kernel call (confirmed empirically: timings came back as exactly 0 without it).
    msrc = ['#include <unistd.h>', '#ifndef REPEAT\n#define REPEAT 1\n#endif', 'int main(void) {']
    global_idxs = []
    for i,b in enumerate(bufs):
      if b[1][0].addrspace == AddrSpace.GLOBAL:
        sz = max(b[1][0].max_numel()*b[1][0].dtype.itemsize, 1)
        msrc.append(f"static unsigned char buf{i}[{sz}] __attribute__((aligned(128)));")
        global_idxs.append(i)
      else:
        msrc.append(f"{self._render_dtype(b[1][0].dtype)} val{i} = 0;")
    params = [(f'(void*)buf{i}' if b[1][0].addrspace == AddrSpace.GLOBAL else f'val{i}') for i,b in enumerate(bufs)]
    msrc.append(f"for (int r = 0; r < REPEAT; r++) {{ {function_name}({', '.join(params)}); }}")
    for i in global_idxs: msrc.append(f"write(1, buf{i}, 1);")
    msrc.append('return 0; }')
    return '\n'.join(msrc)

def _hexsim_tools_dir() -> pathlib.Path:
  root = getenv("HEXAGON_TOOLS", "") or getenv("HEXAGON_TOOLCHAIN", "")
  if not root: raise RuntimeError("HEXSIM=1 needs HEXAGON_TOOLS (or HEXAGON_TOOLCHAIN) set to a Hexagon SDK Tools/ dir with hexagon-sim")
  path = pathlib.Path(root)
  if not (path/"bin"/"hexagon-clang").exists() or not (path/"bin"/"hexagon-sim").exists():
    raise RuntimeError(f"{path} does not look like a Hexagon Tools dir (missing bin/hexagon-clang or bin/hexagon-sim)")
  return path

def _hexsim_env(tools:pathlib.Path, workdir:pathlib.Path) -> dict[str,str]:
  # hexagon-sim links libncurses.so.5, which modern distros only ship as .so.6 (same ABI for
  # this use) -- symlink a shim dir onto LD_LIBRARY_PATH, matching hexagon_sim_harness.py's fix.
  env = dict(os.environ)
  sim = tools/"bin"/"hexagon-sim"
  probe = subprocess.run(["ldd", str(sim)], capture_output=True, text=True, check=False)
  if "not found" not in probe.stdout: return env
  shim = workdir/"shim"
  shim.mkdir(exist_ok=True)
  for name in ("ncurses", "tinfo"):
    link = shim/f"lib{name}.so.5"
    if link.exists(): continue
    for lib_dir in ("/lib/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu", "/usr/lib64"):
      source = pathlib.Path(lib_dir)/f"lib{name}.so.6"
      if source.exists():
        with contextlib.suppress(FileExistsError): link.symlink_to(source)  # benign race under parallel BEAM workers
        break
  env["LD_LIBRARY_PATH"] = f"{shim}:{env.get('LD_LIBRARY_PATH', '')}"
  return env

class HexagonSimCompiler(Compiler):
  def __init__(self): super().__init__("compile_hexsim")

  def _compile_one(self, tools:pathlib.Path, src:str, repeat:int) -> bytes:
    with tempfile.TemporaryDirectory() as d:
      workdir = pathlib.Path(d)
      (workdir/"k.c").write_text(src)
      elf = workdir/"k.elf"
      cmd = [str(tools/"bin"/"hexagon-clang"), f"-m{HEXSIM_ARCH}", "-mhvx", "-mhvx-length=128B", "-O1",
             f"-DREPEAT={repeat}", str(workdir/"k.c"), "-o", str(elf), "-lm"]
      result = subprocess.run(cmd, capture_output=True, text=True, check=False)
      if result.returncode: raise CompileError(f"hexagon-clang failed:\n{result.stderr}")
      return elf.read_bytes()

  def compile(self, src:str) -> bytes:
    tools = _hexsim_tools_dir()
    elf1, elf2 = self._compile_one(tools, src, 1), self._compile_one(tools, src, 2)
    return struct.pack("<Q", len(elf1)) + elf1 + elf2

class HexagonSimProgram(Program[DSPDevice]):
  def __init__(self, dev:DSPDevice, obj:TinyELF):
    n = struct.unpack("<Q", obj.lib[:8])[0]
    self.elf1, self.elf2 = obj.lib[8:8+n], obj.lib[8+n:]

  def _run_pcycles(self, tools:pathlib.Path, elf_bytes:bytes) -> int:
    with tempfile.NamedTemporaryFile(suffix=".elf") as f:
      f.write(elf_bytes)
      f.flush()
      os.chmod(f.name, 0o755)
      env = _hexsim_env(tools, pathlib.Path(f.name).parent)
      # --timing enables hexagon-sim's cycle-accurate pipeline/dual-issue/cache-hierarchy model
      # (confirmed to change the reported Pcycles vs. the default fast functional-only mode --
      # see the README section this lands with); the default mode's Pcycles is a coarser,
      # instruction-scheduling-blind estimate much closer to plain instruction counting.
      result = subprocess.run(
        [str(tools/"bin"/"hexagon-sim"), f"-m{HEXSIM_ARCH}", "--timing", "--simulated_returnval", f.name],
        capture_output=True, text=True, env=env, timeout=900, check=False)
      output = result.stdout + result.stderr
      if result.returncode != 0: raise RuntimeError(f"hexagon-sim exit {result.returncode}:\n{output}")
      m = re.search(r"Pcycles=(\d+)", output)
      if m is None: raise RuntimeError(f"hexagon-sim output has no Pcycles= line:\n{output}")
      return int(m.group(1))

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
    tools = _hexsim_tools_dir()
    base, twice = self._run_pcycles(tools, self.elf1), self._run_pcycles(tools, self.elf2)
    return max(twice - base, 0) / HEXSIM_CLOCK_HZ
