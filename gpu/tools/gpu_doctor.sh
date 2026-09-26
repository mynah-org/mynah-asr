#!/bin/bash
# gpu/tools/gpu_doctor.sh — is this GPU what it claims to be, and is it ours?
#
#   gpu/tools/gpu_doctor.sh [--peak-bw-gbs 300 --peak-tflops 30.3]   (L4 datasheet)
#
# Static facts from nvidia-smi (P-state, clocks, application clocks, power
# limit vs default, compute mode, MIG, other processes), then the measurement
# binary with a 20 s burn while nvidia-smi samples clocks, power, P-state and
# the throttle reasons every second. Run it on an idle box, before any
# measurement, and keep the output with the run's evidence.
set -u
cd "$(dirname "$0")/../.." || exit 2
BIN=build/gpu/gpu_doctor
NVCC=${NVCC:-nvcc}
mkdir -p build/gpu
if [ ! -x "$BIN" ] || [ gpu/tools/gpu_doctor.cu -nt "$BIN" ]; then
    $NVCC -O3 -arch=native -o "$BIN" gpu/tools/gpu_doctor.cu || exit 2
fi
echo "== nvidia-smi static"
nvidia-smi --query-gpu=name,pci.bus_id,driver_version,pstate,compute_mode,mig.mode.current,persistence_mode,power.limit,power.default_limit,power.max_limit,enforced.power.limit,clocks.max.sm,clocks.max.mem,clocks.applications.graphics,clocks.applications.memory,memory.total,memory.used --format=csv
echo "== processes on the GPU (another tenant would show here or as used memory)"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
echo "== clock/power/throttle samples during the burn (1 s)"
nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_throttle_reasons.active,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_slowdown,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.applications_clocks_setting --format=csv -l 1 > build/gpu/doctor_samples.csv 2>&1 &
SMI=$!
"$BIN" --burn-seconds 20 "$@"
rc=$?
kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
cat build/gpu/doctor_samples.csv
exit $rc
