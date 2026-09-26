#!/bin/bash
# Build the golden-tile capture FastRPC skel (mcc_golden_rpc.so) + client for the MCC HVX steps, with the
# Hexagon SDK's qaic/headers and a Hexagon toolchain that knows -mhmx (the SDK's own 19.0.04, or the
# login-free Hexagon_open_access 19.0.02). This is mcc_hmx's build.sh without the weights, the block or
# the HMX context: mcc_golden_impl.c only calls mb_hvx.h's mbv_* steps.
#
#   HEXAGON_SDK_ROOT=<SDK 6.x> HEXAGON_TOOLCHAIN=<SDK>/tools/HEXAGON_Tools/19.0.04/Tools ./build.sh
set -euo pipefail
: "${HEXAGON_SDK_ROOT:?}" "${HEXAGON_TOOLCHAIN:?}"
NDK_CLANG="${NDK_CLANG:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang}"
HEX_ARCH="${HEX_ARCH:-v69}"
SRC="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-$SRC/build}"
mkdir -p "$OUT" && cd "$OUT"
cp "$SRC"/mcc_golden_rpc.idl "$SRC"/mcc_golden_impl.c "$SRC"/mcc_golden_client.c "$SRC"/mcc_block.h "$SRC"/mb_hvx.h .
"$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic" -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" mcc_golden_rpc.idl
INC=(-I . -I "$SRC" -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef")
QURT_INC=(-I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/qurt" -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/posix")
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" "${INC[@]}" -o skel.o mcc_golden_rpc_skel.c
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" -mhvx="$HEX_ARCH" -mhvx-length=128b -mhmx -Wall \
  "${INC[@]}" "${QURT_INC[@]}" -o impl.o mcc_golden_impl.c
LIBPATH="$HEXAGON_TOOLCHAIN/target/hexagon/lib/$HEX_ARCH/G0"
"$HEXAGON_TOOLCHAIN/bin/hexagon-link" -Bdynamic -shared -export-dynamic -o mcc_golden_rpc.so skel.o impl.o "$LIBPATH/pic/libgcc.so"
"$NDK_CLANG" -O2 "${INC[@]}" -o mcc_golden_client mcc_golden_client.c -lm mcc_golden_rpc_stub.c \
  -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc
echo "built $OUT/mcc_golden_rpc.so $OUT/mcc_golden_client"
