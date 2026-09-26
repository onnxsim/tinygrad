#!/bin/bash
# Push + run the golden-tile capture skel on the phone, into this directory's goldens/ case:
#   ./run.sh capture          # push the client + skel, run the case, pull gold_*.bin back
#   ./run.sh health           # the known-good msda health check, after every phone run
# Run only under the host phone lock: PHONE_LOCK_OWNER=<branch> ~/.cache/android-phone/phone-run ./run.sh ...
set -euo pipefail
S="${ANDROID_SERIAL:-239dbd8f}"
SRC="$(cd "$(dirname "$0")" && pwd)"
B="${OUT:-$SRC/build}"
CASE="${CASE:-$SRC/goldens/mcc_r4}"
D="${D:-/data/local/tmp/codex-dsp-mcc-golden}"
MSDA_BUILD="${MSDA_BUILD:-$HOME/.cache/msda_generic_build}"
MSDA_CASE="${MSDA_CASE:-$HOME/.cache/msda_generic_cases/bevformer_tsa}"
A=(adb -s "$S")
RT="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['rt'])" "$CASE/case.json")"
NT="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['nt'])" "$CASE/case.json")"
case "$1" in
  capture)
    "${A[@]}" shell "mkdir -p $D/case"
    "${A[@]}" push "$B/mcc_golden_client" "$B/mcc_golden_rpc.so" $D/ >/dev/null
    for f in x.bin s.bin q.bin k.bin g.bin ln.bin; do "${A[@]}" push "$CASE/$f" "$D/case/" >/dev/null; done
    "${A[@]}" shell "chmod 755 $D/mcc_golden_client"
    "${A[@]}" shell "cd $D && LD_LIBRARY_PATH=/vendor/lib64 ADSP_LIBRARY_PATH=$D timeout ${RUN_TIMEOUT:-120} ./mcc_golden_client \
      'file:///mcc_golden_rpc.so?mcc_golden_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp' case $RT $NT ${ITERS:-3} ${STEPS:-7}; echo exit=\$?"
    "${A[@]}" pull "$D/case/gold_h.bin" "$D/case/gold_spv.bin" "$D/case/gold_pv.bin" "$D/case/gold_gelu.bin" \
      "$D/case/gold_times.txt" "$CASE/" >/dev/null
    sha256sum "$CASE"/gold_*.bin ;;
  health)
    sleep 2
    "${A[@]}" shell "cd $D && LD_LIBRARY_PATH=/vendor/lib64 ADSP_LIBRARY_PATH=$D timeout 60 ./msda_client \
      'file:///msda_rpc.so?msda_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp' 3 0 4 case 2>&1 | tail -1; echo exit=\$?" ;;
  clean) "${A[@]}" shell "rm -rf $D" ;;
esac
