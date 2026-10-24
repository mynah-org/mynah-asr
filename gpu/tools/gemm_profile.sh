#!/bin/bash
# gpu/tools/gemm_profile.sh — the own-tc GEMM measurement package (real shapes).
#
#   [MODELS="nemotron eou"] gpu/tools/gemm_profile.sh bench [extra gemm_shapes_bench args, e.g. --dec-iters 9]
#   gpu/tools/gemm_profile.sh ncu   [--model nemotron] [--lanes 128] [--shapes "ff_w1 ff_w2 qkv pw1"]
#   gpu/tools/gemm_profile.sh nsys  <model_dir> [--lanes 128] [--steps 12]
#   gpu/tools/gemm_profile.sh all   <model_dir>
#
# bench  tests/gemm_shapes_bench at C = 32 and 128: own-tc vs cuBLAS bf16 vs the
#        roofline on every (M, N, K) a pass issues, weights cold; table + CSV.
# ncu    Nsight Compute on gemm_tc_kernel and gemm_reduce_kernel for the
#        dominant shapes: speed of light, tensor-pipe use, warp stall reasons,
#        DRAM vs L2 bytes. One shape per run, 3 timed reps, one profiled
#        launch of each kernel after warm-up.
# nsys   Nsight Systems over one `test_cuda_stream --bench C` run of the bf16
#        own-tc engine: each kernel's share of the pass (cuda_gpu_kern_sum).
#
# Every step runs under `timeout` (TB seconds, default 900): a hung profiler
# never holds the box. Output: build/gpu/gemm-prof/<UTC stamp>/.
#
# Env: NVCC, NCU (ncu), NSYS (nsys), TB (timeout s), PEAK (assumed bf16 dense
# TFLOP/s for the roofline, default 62 — GB10 is NOT verified, see the bench
# header; the cuBLAS column is the measured ceiling), CUDA_ARCH (Makefile).
set -u
cd "$(dirname "$0")/../.." || exit 2
TB=${TB:-900}
PEAK=${PEAK:-62}
NCU=${NCU:-ncu}
NSYS=${NSYS:-nsys}
OUT=build/gpu/gemm-prof/$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
BIN=tests/gemm_shapes_bench

build_bench() {
    if [ ! -x "$BIN" ]; then
        timeout "$TB" make -C gpu gemm-shapes ${CUDA_ARCH:+CUDA_ARCH="$CUDA_ARCH"} > "$OUT/build.log" 2>&1 || { echo "build failed: $OUT/build.log"; exit 2; }
    fi
}

cmd=${1:-}; shift || true
case "$cmd" in
bench)
    build_bench
    for model in ${MODELS:-nemotron eou}; do
        timeout "$TB" "$BIN" --model "$model" --lanes 32,128 --peak-tflops "$PEAK" "$@" | tee "$OUT/bench-$model.txt"
        timeout "$TB" "$BIN" --model "$model" --lanes 32,128 --peak-tflops "$PEAK" --csv "$@" | grep '^CSV' > "$OUT/bench-$model.csv"
    done
    ;;
ncu)
    build_bench
    model=nemotron lanes=128 shapes="ff_w1 ff_w2 qkv pw1"
    while [ "$#" -gt 0 ]; do
        case "$1" in --model) model=$2; shift 2;; --lanes) lanes=$2; shift 2;; --shapes) shapes=$2; shift 2;; *) echo "unknown $1"; exit 2;; esac
    done
    # metrics: keep only the ones this ncu knows for this device (names move
    # between architectures; an unknown metric aborts the whole run)
    want="sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active
sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active
sm__throughput.avg.pct_of_peak_sustained_elapsed
dram__throughput.avg.pct_of_peak_sustained_elapsed
dram__bytes_read.sum
dram__bytes_write.sum
lts__t_bytes.sum
lts__t_sector_hit_rate.pct
l1tex__data_pipe_lsu_wavefronts_mem_shared.sum
sm__warps_active.avg.pct_of_peak_sustained_active
smsp__average_warp_latency_issue_stalled_barrier.ratio
smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio
smsp__average_warp_latency_issue_stalled_mio_throttle.ratio
smsp__average_warp_latency_issue_stalled_short_scoreboard.ratio
smsp__average_warp_latency_issue_stalled_math_pipe_throttle.ratio
gpu__time_duration.sum"
    timeout 120 "$NCU" --query-metrics > "$OUT/ncu-metrics-available.txt" 2>&1
    have=""
    for mt in $want; do
        base=${mt%%.*}
        grep -q "^$base\b" "$OUT/ncu-metrics-available.txt" && have="$have,$mt"
    done
    have=${have#,}
    echo "ncu metrics used: $have" | tee "$OUT/ncu-metrics-used.txt"
    for s in $shapes; do
        # warm-up (2 reps) is skipped by --launch-skip; then one launch of each kernel
        for k in gemm_tc_kernel gemm_reduce_kernel; do
            timeout "$TB" "$NCU" --target-processes all -k "regex:$k" --launch-skip 2 --launch-count 1 \
                --section SpeedOfLight --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis \
                --section WarpStateStats --section Occupancy --section LaunchStats --section SchedulerStats \
                ${have:+--metrics "$have"} --csv --page details \
                "$BIN" --model "$model" --lanes "$lanes" --only "$s" --reps 3 --no-cublas \
                > "$OUT/ncu-$model-c$lanes-$s-$k.csv" 2> "$OUT/ncu-$model-c$lanes-$s-$k.err"
            echo "ncu $s $k -> $OUT/ncu-$model-c$lanes-$s-$k.csv (rc $?)"
        done
        # the ceiling, same shape: cuBLAS's kernel(s) (name varies by version)
        timeout "$TB" "$NCU" --target-processes all -k "regex:^(?!gemm_tc_kernel|gemm_reduce_kernel|fill_kernel|copy_kernel|k_f32_to_bf16|reduce_model).*" \
            --launch-skip 2 --launch-count 1 --section SpeedOfLight ${have:+--metrics "$have"} --csv --page details \
            "$BIN" --model "$model" --lanes "$lanes" --only "$s" --reps 3 \
            > "$OUT/ncu-$model-c$lanes-$s-cublas.csv" 2> "$OUT/ncu-$model-c$lanes-$s-cublas.err"
    done
    ;;
nsys)
    model_dir=${1:?usage: nsys <model_dir> [--lanes 128] [--steps 12]}; shift
    lanes=128 steps=12
    while [ "$#" -gt 0 ]; do
        case "$1" in --lanes) lanes=$2; shift 2;; --steps) steps=$2; shift 2;; *) echo "unknown $1"; exit 2;; esac
    done
    [ -x tests/test_cuda_stream ] || timeout "$TB" make -C gpu test-stream ${CUDA_ARCH:+CUDA_ARCH="$CUDA_ARCH"} > "$OUT/build-stream.log" 2>&1 || { echo "build failed"; exit 2; }
    timeout "$TB" "$NSYS" profile -t cuda,nvtx --cuda-graph-trace=node --force-overwrite true -o "$OUT/pass-c$lanes" \
        tests/test_cuda_stream "$model_dir" --precision bf16 --gemm own-tc --bench "$lanes" --steps "$steps" \
        | tee "$OUT/pass-c$lanes.bench.txt"
    timeout "$TB" "$NSYS" stats --report cuda_gpu_kern_sum --format csv --output "$OUT/pass-c$lanes" "$OUT/pass-c$lanes.nsys-rep" > /dev/null
    timeout "$TB" "$NSYS" stats --report cuda_gpu_kern_sum "$OUT/pass-c$lanes.nsys-rep" | head -40
    ;;
all)
    model_dir=${1:?usage: all <model_dir>}
    "$0" bench
    "$0" ncu --lanes 128
    "$0" ncu --lanes 32
    "$0" nsys "$model_dir" --lanes 128
    "$0" nsys "$model_dir" --lanes 32
    ;;
*)
    sed -n '2,24p' "$0"; exit 2;;
esac
echo "evidence: $OUT"
