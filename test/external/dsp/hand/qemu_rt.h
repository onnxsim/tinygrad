/* Freestanding qemu-hexagon (linux-user) runtime for the hand-kernel oracle drivers: raw trap0 syscalls, file
 * load/store, QEMU's executed-instruction counter, and the libc bits the kernels call. Same style as the
 * onnx-simplifier bridge's *_qemu.c drivers, factored out so each oracle driver is only its argv plumbing.
 *
 * A driver defines `int oracle_main(int argc, char** argv)` and includes this file last-but-one (before its own
 * code, after the kernel header). argv: input files, then output files, then integers -- the driver decides. The
 * process exit code is oracle_main's return value. Instruction counts go to stdout as "insns <label> <n>" lines,
 * which harness.run_hand parses. */
#ifndef QEMU_RT_H
#define QEMU_RT_H
#include <stddef.h>
#include <stdint.h>

#ifndef QEMU_RT_NO_LIBC
void* memcpy(void* d, const void* s, size_t n) { char* a = d; const char* b = s; while (n--) *a++ = *b++; return d; }
void* memset(void* d, int c, size_t n) { char* a = d; while (n--) *a++ = (char)c; return d; }
void* memmove(void* d, const void* s, size_t n) {
  char* a = d; const char* b = s;
  if (a < b) while (n--) *a++ = *b++; else { a += n; b += n; while (n--) *--a = *--b; }
  return d; }
int memcmp(const void* x, const void* y, size_t n) {
  const unsigned char *a = x, *b = y;
  for (; n; n--, a++, b++) if (*a != *b) return *a - *b;
  return 0; }
/* Hexagon has no divide instruction; the DSP builds get these from libgcc */
unsigned __hexagon_udivsi3(unsigned a, unsigned b) {
  unsigned q = 0, r = 0;
  for (int i = 31; i >= 0; i--) { r = (r << 1) | ((a >> i) & 1); if (r >= b) { r -= b; q |= 1u << i; } }
  return q; }
unsigned __hexagon_umodsi3(unsigned a, unsigned b) { return a - __hexagon_udivsi3(a, b) * b; }
int __hexagon_divsi3(int a, int b) {
  unsigned ua = a < 0 ? -(unsigned)a : (unsigned)a, ub = b < 0 ? -(unsigned)b : (unsigned)b, q = __hexagon_udivsi3(ua, ub);
  return (a < 0) != (b < 0) ? -(int)q : (int)q; }
int __hexagon_modsi3(int a, int b) { return a - __hexagon_divsi3(a, b) * b; }
#endif

static long qrt_sys6(long a, long b, long c, long d, long e, long f, long n) {
  long r;
  __asm__ volatile("r0=%1;r1=%2;r2=%3;r3=%4;r4=%5;r5=%6;r6=%7;trap0(#1);%0=r0"
                   : "=r"(r) : "r"(a), "r"(b), "r"(c), "r"(d), "r"(e), "r"(f), "r"(n)
                   : "r0", "r1", "r2", "r3", "r4", "r5", "r6", "memory");
  return r; }
/* control register 21 = HEX_REG_QEMU_INSN_CNT */
static inline unsigned qrt_inscount(void) { unsigned r; __asm__ volatile(".word 0x6a15c000; %0 = R0" : "=r"(r) : : "r0"); return r; }
static void qrt_exit(int code) { qrt_sys6(code, 0, 0, 0, 0, 0, 93); for (;;) {} }
static void qrt_puts(const char* s) { int n = 0; while (s[n]) n++; qrt_sys6(1, (long)s, n, 0, 0, 0, 64); }
static void qrt_putu(unsigned long v) { char b[24]; int i = 23; b[i] = 0; do { b[--i] = '0' + v % 10; v /= 10; } while (v); qrt_puts(b + i); }
static void qrt_insns(const char* label, unsigned n) { qrt_puts("insns "); qrt_puts(label); qrt_puts(" "); qrt_putu(n); qrt_puts("\n"); }
/* zero-filled, 128-byte (HVX) aligned: anonymous mmap is page aligned */
static void* qrt_map(long bytes) { return (void*)qrt_sys6(0, (bytes + 4095) & ~4095L, 3, 0x22, -1, 0, 222); }
static void* qrt_load(const char* path, long bytes) {
  int fd = qrt_sys6(-100, (long)path, 0, 0, 0, 0, 56);
  if (fd < 0) { qrt_puts("open failed: "); qrt_puts(path); qrt_puts("\n"); qrt_exit(2); }
  char* b = qrt_map(bytes);
  for (long o = 0; o < bytes;) { long n = qrt_sys6(fd, (long)(b + o), bytes - o, 0, 0, 0, 63); if (n <= 0) break; o += n; }
  qrt_sys6(fd, 0, 0, 0, 0, 0, 57);
  return b; }
static void qrt_store(const char* path, const void* p, long bytes) {
  int fd = qrt_sys6(-100, (long)path, 0x241 /* O_WRONLY|O_CREAT|O_TRUNC */, 0644, 0, 0, 56);
  if (fd < 0) { qrt_puts("create failed: "); qrt_puts(path); qrt_puts("\n"); qrt_exit(2); }
  for (long o = 0; o < bytes;) { long n = qrt_sys6(fd, (long)((const char*)p + o), bytes - o, 0, 0, 0, 64); if (n <= 0) break; o += n; }
  qrt_sys6(fd, 0, 0, 0, 0, 0, 57); }
static int qrt_atoi(const char* s) {
  int v = 0, neg = *s == '-';
  if (neg) s++;
  while (*s >= '0' && *s <= '9') v = v * 10 + (*s++ - '0');
  return neg ? -v : v; }

int oracle_main(int argc, char** argv);
void __attribute__((noreturn, used)) qrt_main_(long* sp) { qrt_exit(oracle_main((int)sp[0], (char**)(sp + 1))); for (;;) {} }
__asm__(".global _start\n_start:\n r0 = r29\n jump qrt_main_\n");
#endif
