import itertools
from tinygrad.codegen.opt import Opt, OptOps, KernelOptError
from tinygrad.helpers import getenv, DEBUG, prod, NOLOCALS, TC_OPT, TC_SELECT, USE_TC, IMAGE
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, resolve, AxisType
from tinygrad.codegen.late.coalesce import image_valid_dims
from tinygrad.codegen.opt.postrange import Scheduler

HVX_UPCAST_CONTIG = getenv("HVX_UPCAST_CONTIG", 1)

def _unit_stride_bufs(k:Scheduler, axis:int) -> int:
  # how many buffers' indices have this axis's range as a bare term (stride 1): upcasting it gives them contiguous vector accesses
  rng = k.rngs[axis]
  return sum(any(c is rng for c in b.src[1].get_idx().split_uop(Ops.ADD)) for b in k.bufs)

def _unit_stride(k:Scheduler, axis:int) -> bool: return _unit_stride_bufs(k, axis) > 0

def hand_coded_optimizations(k:Scheduler) -> Scheduler:
  # first try the tensor cores
  """ Attempts to apply a tensor core optimization to the kernel. If one exists and applies properly, return true, otherwise return false.
  Tensor cores are optimized instructions that matrix multiply-accumulate across a wave of threads: D(M, N) = A(M, K) * B(K, N) + C(M, N).

  Keyword arguments:
  use_tensor_cores -- controls how tensor cores are applied (default 1)
    0: will disable any tensor core matching
    1: enable tensor cores
    2: apply tensor core shape but don't use UOp.WMMA
  extra_opts -- additional Opt's to apply after the tensor core instead of the hand-coded additional Opt's (default None)
  tc_select -- specifies which tensor core(s) to use for optimization (default -1)
    -1: iterates through all available tensor cores in order and uses the first one that matches the requirements (dims and dtypes)
    [0-N]: uses only the n'th tensor core available; useful for search
  tc_opt -- controls which kinds of kernels may be eligible for tensor cores application (default 2 during BEAM, 0 otherwise)
    0: applies to only kernels with a single reduce axis and direct Ops.LOAD into Ops.MUL
    1: allows kernels with multiple reduce axes and also multiplication of Ops.CAST'd buffers
    2: allows kernels with M, N, K axes that are not multiples of the tensor core dimensions by applying padding those axes as needed
  """
  # NOTE: unless TC_OPT is > 0, we only trigger tensor cores if there's only one reduce axis
  if USE_TC > 0 and (len(k.axes_of(AxisType.GROUP_REDUCE, AxisType.REDUCE)) == 1 or (TC_OPT.value >= 1)):
    for axis in range(3):
      tk = k.copy()
      # check TC first and apply hand-coded opts if successful
      try: rngs = tk.apply_opt(Opt(OptOps.TC, axis, (TC_SELECT.value, TC_OPT.value, USE_TC.value)))
      except KernelOptError: continue
      for tc_dim in [1,0]: # attempt to upcast M and N
        if rngs[tc_dim] is None: continue # M=1 TC (a GEMV) has no M range
        # Hexagon's vrmpy TC is one vector instruction per thread: an extra M/N upcast lands *inside* its 32 accumulator
        # lanes (later upcasts are the faster axes of the register accumulator), so every WMMA's C becomes a strided gather.
        # One WMMA per 32-lane accumulator slice keeps it a single HVX register.
        if tk.ren is not None and tk.ren.target.device == "DSP":
          # DSP_TC_MUPCAST pixels per weight load (after the TC's own lanes, so no strided accumulator): each 128-byte weight vector
          # feeds that many vrmpys before it leaves the register file. 0 or 1 = off; 8 measured best on the V69 phone (2: 228 ms, 4: 204, 8: 203, 16: 218 for driving)
          if tc_dim == 0 and getattr(tk, "tensor_core", None) is not None and tk.tensor_core.dims == (32, 1, 4) and \
              (mu:=getenv("DSP_TC_MUPCAST", 8)) > 1 and rngs[0] is not None and rngs[0].src[0].divides(mu) is not None:
            try: tk.apply_opt(Opt(OptOps.UPCAST, tk.rngs.index(rngs[0]), mu))
            except KernelOptError: pass
          continue
        szs = [sz for sz in [5,4,3,2] if rngs[tc_dim].src[0].divides(sz) is not None]
        if szs:
          # set it to the replaced range
          rngs[tc_dim] = tk.apply_opt(Opt(OptOps.UPCAST, tk.rngs.index(rngs[tc_dim]), szs[0]))[0]
      # attempt to local N -- only for backends that support locals (e.g. Hexagon's vrmpy tensor core is a
      # single-instruction, single-thread op with no warp/lane cooperation, so it has no LOCAL axis to use)
      if tk.ren is not None and tk.ren.has_local and (szs := [sz for sz in [4,2] if rngs[0].src[0].divides(sz) is not None]):
        tk.apply_opt(Opt(OptOps.LOCAL, tk.rngs.index(rngs[0]), szs[0]))
      # the DSP's TensorCore is one HVX thread's instruction: split its kernel over the hardware threads like any other
      return apply_threads(tk) if tk.ren is not None and tk.ren.target.device == "DSP" else tk

  # make a copy so it does not mutate the input
  k = k.copy()
  is_dsp = k.ren is not None and k.ren.target.device == "DSP"
  # a renderer can cap upcasts at one vector register: v65 HVX (128 bytes, no float vectors) sets upcast_max_bytes, so the
  # lane cap follows the reduction dtype (32 f32 lanes, 128 byte lanes). qfloat DSPs keep several-vector float accumulators
  max_bytes = getattr(k.ren, "upcast_max_bytes", None)
  dsp_vector_lanes = max_bytes // k.reduceop.dtype.itemsize if is_dsp and max_bytes and k.reduceop is not None else 128
  # ...and there float is scalar, so a float reduction wants register blocking (small upcasts on the axes whose loads are
  # reused, as on a CPU) rather than one contiguous vector-wide axis: it takes the non-DSP upcast rules below
  v65_float = is_dsp and max_bytes is not None and k.reduceop is not None and dtypes.is_float(k.reduceop.dtype)
  dsp_scalar_float = bool(getenv("DSP_SCALAR_BLOCK", 1)) and v65_float
  # blocking only pays with reuse in two directions (a conv: inputs shared across output channels, weights across pixels); a GEMV
  # has one, and blocking its output axis measured 10x slower on the phone than the contiguous vector-style upcast
  if dsp_scalar_float:
    reuse_axes = [a for a in k.upcastable_dims if k.full_shape[a] > 1 and any(k.rngs[a] not in b.src[1].get_idx().backward_slice for b in k.bufs)]
    dsp_scalar_float = len(reuse_axes) >= 2

  # upcast float4 images, this must be early so we don't accidentally add locals before the upcast
  if IMAGE:
    for buf_index,buf in enumerate(k.bufs):
      if image_valid_dims(buf.src[0].dtype, buf.src[0].max_numel(), k.ren.target.arch):
        idx = k.bufs[buf_index].src[1]
        # IMAGE upcasts require one validity shared by all four unit-stride lanes so memory_coalescing can combine them into one vector read.
        unit_stride_axes_mul_4 = [k.rngs.index(c) for c in idx.get_idx().split_uop(Ops.ADD) if
          c.op is Ops.RANGE and (c.vmax+1)%4 == 0 and c not in idx.get_valid().backward_slice]
        if len(unit_stride_axes_mul_4):
          if (axis:=unit_stride_axes_mul_4[0]) in k.upcastable_dims:
            k.apply_opt(Opt(OptOps.UPCAST, axis, 4))
          elif axis in k.unrollable_dims:
            k.apply_opt(Opt(OptOps.UNROLL, k.unrollable_dims.index(axis), 4))

  # should use matvec - TODO: adjust/tune based on the wide vs tall/large vs small mat
  MV_BLOCKSIZE, MV_THREADS_PER_ROW, MV_ROWS_PER_THREAD = getenv("MV_BLOCKSIZE", 4), getenv("MV_THREADS_PER_ROW", 8), getenv("MV_ROWS_PER_THREAD", 4)
  if k.ren.has_local and getenv("MV",1) != 0 and (MV_BLOCKSIZE > 1 or MV_THREADS_PER_ROW > 1 or MV_ROWS_PER_THREAD > 1) and  \
    k.reduceop is not None and k.reduceop.arg[0] is Ops.ADD and len(k.full_shape) >= 2 and k.ren.has_shared and \
    (mulop:=k.reduceop.src[0]).op is Ops.MUL and mulop.src[0].op is Ops.INDEX and mulop.src[1].op is Ops.INDEX:
    idx0, idx1 = mulop.src[0].src[1].get_idx(), mulop.src[1].src[1].get_idx()
    if k.ranges_of(AxisType.REDUCE):
      first_reduce_rng = k.ranges_of(AxisType.REDUCE)[0]
      if any(u is first_reduce_rng for u in idx0.split_uop(Ops.ADD)) and all(r in idx1.ranges for r in idx0.ranges):
        for global_idx in k.axes_of(AxisType.GLOBAL):
          if first_reduce_rng.src[0].divides(MV_THREADS_PER_ROW) is not None and k.full_shape[global_idx]%(MV_BLOCKSIZE*MV_ROWS_PER_THREAD) == 0:
            if DEBUG >= 3:
              print(f"MATVEC: {k.full_shape=} {first_reduce_rng.render()} {MV_BLOCKSIZE=} {MV_THREADS_PER_ROW=} {MV_ROWS_PER_THREAD=}")
            try:
              if MV_THREADS_PER_ROW > 1: k.apply_opt(Opt(OptOps.GROUP, 0, MV_THREADS_PER_ROW))
            except KernelOptError: pass
            if MV_BLOCKSIZE > 1: k.apply_opt(Opt(OptOps.LOCAL, global_idx, MV_BLOCKSIZE))
            if MV_ROWS_PER_THREAD > 1: k.apply_opt(Opt(OptOps.UPCAST, global_idx, MV_ROWS_PER_THREAD))
            return k

  # are we grouping? (requires local shape support)
  if resolve(prod(k.output_shape[i] for i in k.upcastable_dims) <= (240 if NOLOCALS else 2048), False):
    for axis, sz in itertools.product((0, 1, 2), (16,)):
      try:
        k.apply_opt(Opt(OptOps.GROUPTOP, axis, sz))
        break
      except KernelOptError: pass

  # no more opt if we are grouping
  if k.group_for_reduces: return k

  # **** below this line need to be optional and benchmarked ****

  # if there are small dims with lots of valid masks, upcast them (they might be from Tensor.stack)
  to_upcast: list[int] = []
  where_gate_rngs = {r for u in k.ast.backward_slice if u.op is Ops.WHERE for r in u.src[0].ranges}
  # upcast leading axes first (hack-ish for winograd; we actually want to upcast masked axes with low stride first)
  for axis in k.upcastable_dims:
    # for Schedule, we check if the range is used in INDEX gates or WHERE gates
    is_masked = k.rngs[axis] in where_gate_rngs
    max_masked_upcast = min(7 * 7, dsp_vector_lanes // k.upcast_size()) if is_dsp else 7 * 7
    if k.full_shape[axis] <= 7 and is_masked and prod(k.full_shape[j] for j in to_upcast) * k.full_shape[axis] <= max_masked_upcast:
      # upcasting a masked global axis moves that range out of the launch grid into each work-item
      # under IMAGE, skip the upcast unless enough global work-items remain after it to hide memory latency
      if IMAGE and k.axis_types[axis] is AxisType.GLOBAL:
        global_upcast = prod(k.full_shape[i] for i in to_upcast if k.axis_types[i] is AxisType.GLOBAL) * k.full_shape[axis]
        global_items_after = prod(k.full_shape[i] for i in k.axes_of(AxisType.GLOBAL)) // global_upcast
        if resolve(global_items_after < getenv("OCCUPANCY_FLOOR", 4096), False): continue
      if DEBUG >= 4: print(f"upcasting masked axis : {axis}")
      to_upcast.append(axis)
  is_dsp = k.ren is not None and k.ren.target.device == "DSP"
  # on the DSP, the full upcast of small masked axes (up to 7x7 = 49 lanes) fills the upcast budget below (< 32), so a
  # unit-stride axis never gets its 128-lane HVX vector -- a RoiAlign's 7x7 bins x 2x2 samples ended up 784 scalar
  # gathers per channel with the 256-channel loop scalar. Keep the masked upcasts only when no axis can take a
  # contiguous HVX-width upcast (HVX_UPCAST_CONTIG). Only for reductions: for elementwise kernels (a bitonic sort's
  # split/cat stages) the masked upcasts are what keep the cat's selects out of the inner loop, and dropping them doubled
  # a TopK's instructions
  if is_dsp and HVX_UPCAST_CONTIG and k.axes_of(AxisType.REDUCE) and to_upcast and any(
      k.full_shape[a] % w == 0 and a not in to_upcast and _unit_stride(k, a) for a in k.upcastable_dims for w in (128, 64, 32)):
    to_upcast = []
  for axis in to_upcast[::-1]: k.apply_opt(Opt(OptOps.UPCAST, axis, 0))

  # potentially do more upcasts of non reduce axes based on a heuristic
  upcasted_axis: set[int] = set()
  while resolve(prod(k.output_shape[i] for i in k.upcastable_dims) >= 1024) and (k.upcast_size() < 32):
    xb_choices = []
    # consider upcasts up to one HVX vector (the lane count depends on the reduction dtype); real shapes like ...x272
    # aren't multiples of 128 and would otherwise fall back to a 4-wide upcast for byte-sized reductions
    dsp_upcast_sizes = [s for s in [128,64,32,16,8,4] if s * k.upcast_size() <= dsp_vector_lanes]
    vector_dsp = is_dsp and not dsp_scalar_float
    for axis, upcast_amount in itertools.product(k.upcastable_dims,
        (dsp_upcast_sizes if not len(upcasted_axis) else []) if vector_dsp else [3,4]):
      # if we haven't upcasted it, it mods, and buffer has stride 0 on axis while having no stride 0 in the upcasted axis already
      if axis in upcasted_axis or k.full_shape[axis]%upcast_amount != 0: continue
      rng = k.rngs[axis]
      if any(rng not in b.src[1].get_idx().backward_slice and all(r2 in b.src[1].get_idx().backward_slice
          for r2 in k.ranges_of(AxisType.UPCAST, AxisType.UNROLL)) for b in k.bufs):
        num_strides, sum_strides, gathers = 0, 0, 0
        for b in k.bufs:
          idx = b.src[1].get_idx()
          if rng in idx.backward_slice: num_strides += 1
          unit = False
          for c in idx.split_uop(Ops.ADD):
            if c is rng: sum_strides, unit = sum_strides + 1, True
            if c.op is Ops.MUL and c.src[0] is rng and c.src[1].op is Ops.CONST: sum_strides += c.src[1].val
            if c.op is Ops.MUL and c.src[1] is rng and c.src[0].op is Ops.CONST: sum_strides += c.src[0].val
          # a buffer indexed along the axis other than at unit stride: its vector access is a gather (scalar loads on HVX)
          if rng in idx.backward_slice and not unit: gathers += 1
        # on the DSP first avoid gathers (HVX_UPCAST_CONTIG=1), then prefer the widest vector for the same axis (both keys are 0
        # elsewhere, so ordering is unchanged)
        # v65 (upcast_max_bytes set): an upcast that makes a load a gather only costs -- float code there is scalar, and integer
        # vectors need unit stride. Such a kernel (a conv epilogue over 1350 pixels: only the channel axis divides) stays scalar
        # (integer reductions too: they are HVX vector code; only the scalar float register blocking gains from such upcasts)
        if is_dsp and max_bytes is not None and not dsp_scalar_float and gathers and getenv("DSP_V65_NO_GATHER_UPCAST", 1): continue
        xb_choices.append((gathers if vector_dsp and HVX_UPCAST_CONTIG else 0, num_strides, sum_strides, -upcast_amount if vector_dsp else 0,
                           axis, upcast_amount))
    if xb_choices:
      xb_choices = sorted(xb_choices)
      if DEBUG >= 4: print(f"more upcast axis : {xb_choices}")
      k.apply_opt(Opt(OptOps.UPCAST, xb_choices[0][-2], xb_choices[0][-1]))
      upcasted_axis.add(xb_choices[0][-2])
    else: break

  # on the DSP, a reduction nothing broadcasts into (a per-element dot product like q . k over a small head dim) got no upcast
  # above; the unroll below would then take every "nothing upcasted" case and leave it scalar. Upcast the innermost output
  # axis first instead: a vector accumulator per 128 outputs, the reduce stays a loop
  # (not for scalar float on v65: 128 accumulators live in memory, and a GEMV with [out, in] weights reads 128 rows per step)
  if is_dsp and HVX_UPCAST_CONTIG and not (v65_float and getenv("DSP_SCALAR_GEMV", 1)) and not k.axes_of(AxisType.UPCAST) and \
      k.axes_of(AxisType.REDUCE):
    # (on v65: the unit-stride axis, which needn't be the last -- a channel-blocked depthwise conv's is the 32 channels, its last
    # the image width -- and at most one register of lanes). The axis at unit stride in the most buffers first: a W16 depthwise conv
    # writing NCHW has its width at unit stride in the output only, and upcasting it made the input and weight loads 32-lane gathers
    axes = [k.upcastable_dims[-1]] if k.upcastable_dims else []
    if max_bytes is not None:
      axes = sorted([a for a in k.upcastable_dims if _unit_stride(k, a)][::-1], key=lambda a: -_unit_stride_bufs(k, a)) + axes
    for axis, splits in itertools.product(axes, [s for s in [128,64,32] if max_bytes is None or s <= dsp_vector_lanes]):
      if k.full_shape[axis] % splits == 0:
        k.apply_opt(Opt(OptOps.UPCAST, axis, splits))
        break

  # if last reduce dim is small(ish), loop unroll the reduce
  # NOTE: this can fail on multireduce with mismatching dimensions, this is okay
  try:
    # scalar float on v65: accumulators x unrolled taps must stay near the 32-register file, or the loop body is spills
    scalar_cap = getenv("DSP_SCALAR_UNROLL", 16) if dsp_scalar_float else None
    if k.unrollable_dims and (k.upcast_size() <= 4 or not k.axes_of(AxisType.UNROLL)) and (k.upcast_size() < 64) and \
        (scalar_cap is None or k.upcast_size() * k.full_shape[k.unrollable_dims[-1]] <= scalar_cap):
      if (s:=k.full_shape[k.unrollable_dims[-1]]) <= 32:
        k.apply_opt(Opt(OptOps.UNROLL, len(k.unrollable_dims)-1, 0))
        # if it's small, upcast a second reduce dimension too
        if k.unrollable_dims and s <= 3 and k.full_shape[k.unrollable_dims[-1]] <= 3 and \
            (scalar_cap is None or k.upcast_size() * k.full_shape[k.unrollable_dims[-1]] <= scalar_cap):
          k.apply_opt(Opt(OptOps.UNROLL, len(k.unrollable_dims)-1, 0))
      else:
        for splits in [4]:
          if k.full_shape[axis:=k.unrollable_dims[-1]]%splits == 0:
            k.apply_opt(Opt(OptOps.UNROLL, len(k.unrollable_dims)-1, splits))
            break
  except KernelOptError: pass

  # if nothing at all is upcasted and it's easy to, do an upcast (on the DSP, up to one HVX vector)
  for splits in ([s for s in [128,64,32,16,8,4] if s <= dsp_vector_lanes] if is_dsp and not (v65_float and getenv("DSP_SCALAR_GEMV", 1))
                 else [4]):
    if not k.upcasted and k.upcastable_dims and k.full_shape[k.upcastable_dims[-1]] % splits == 0:
      k.apply_opt(Opt(OptOps.UPCAST, k.upcastable_dims[-1], splits))
      break

  # **** local groups ****

  if k.ren.has_local:
    if NOLOCALS:
      k.apply_opt(Opt(OptOps.NOLOCALS))
    else:
      # prioritize making expand axes local
      local_axis_ranking = [(any(k.rngs[axis] not in b.src[1].get_idx().backward_slice for b in k.bufs), axis) \
                              for axis in k.axes_of(AxisType.GLOBAL, AxisType.WEAK) if k.rngs[axis].src[0].op is Ops.CONST]
      to_local: list[tuple[int, int]] = []
      for _, axis in sorted(local_axis_ranking, key=lambda x: (-x[0], -x[1])):
        local_size = prod(sz for _, sz in to_local)
        local_sz: int|None = next((x for x in ([32] * (axis == 0) + [16,8,4,3,2]) if k.full_shape[axis] % x == 0 and local_size * x <= 128), None)
        if local_sz is not None: to_local.append((axis, local_sz))
      deleted_shape = 0
      for axis, local_sz in sorted(to_local[:3]):
        axis = axis - deleted_shape
        will_delete_shape = local_sz == k.full_shape[axis]
        k.apply_opt(Opt(OptOps.LOCAL, axis, local_sz))
        if will_delete_shape: deleted_shape += 1

  return apply_threads(k)

def apply_threads(k:Scheduler) -> Scheduler:
  if k.ren.has_threads and k.ren.global_max is not None:
    for threads in [32,16,12,8,6,5,4,3,2]:
      # Skip if too many threads. Heuristic: use about 128K ops per thread (a renderer whose per-element cost is far higher, like
      # the DSP's scalar float, sets a lower thread_min_elems)
      if threads > k.ren.global_max[0] or resolve(prod(k.full_shape) // getattr(k.ren, "thread_min_elems", 128 << 10) < threads): continue
      for axis in k.axes_of(AxisType.WEAK):
        if k.full_shape[axis] % threads == 0:
          try: k.apply_opt(Opt(OptOps.THREAD, axis, threads))
          except KernelOptError: pass
          break
      if k.applied_opts and k.applied_opts[-1].op is OptOps.THREAD: break
  return k
