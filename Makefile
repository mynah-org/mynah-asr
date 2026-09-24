# Mynah — build.
#
# BLAS = who computes the f32 GEMMs.  `make BLAS=none|openblas|accelerate`.
#
#   none         src/sgemm.c, ours.  No cblas symbol anywhere in the binary.
#                THE LINUX DEFAULT since 2026-09-19.
#   openblas     -lopenblas.  Kept building forever: it is the comparison arm,
#                and the arm a box may still choose if its own numbers say so.
#   accelerate   macOS only, and the macOS default: a development convenience
#                behind the same seam, never a production claim.
#
# The choice is stamped into the binary and reported by `--dispatch-map` and
# the [EFFECTIVE-CONFIG] line, so a run says which provider it actually linked
# instead of leaving it to be inferred from this file (ENGINEERING.md §5).
CC      ?= cc
CFLAGS  ?= -std=c11 -O3 -march=native -ffast-math -Wall -Wextra -iquote src -D_DEFAULT_SOURCE
LDFLAGS ?=

CFLAGS += -fPIC

UNAME_S := $(shell uname -s)
# The f32 GEMM provider.
#
# `own` (BLAS=none) is the Linux default.  Two reasons, one of ownership and
# one measured:
#
#   OWNERSHIP.  OpenBLAS brings its own thread pool with its own policies:
#   inside a prefork worker pinned to T cpus our pool builds T threads and
#   OpenBLAS builds T more, so the worker runs 2T threads on T cpus and the two
#   pools take turns owning the cores.  Measured on the Axion: one worker pinned
#   to 8 cpus held 63 threads with OpenBLAS linked and 32 without.  On top of
#   that comes an env var that has to be ABSENT for a profile to be valid, and a
#   team size that ignores our own pool.  Each is a trap to recheck on every
#   host, forever.
#
#   THE KERNELS.  Until 2026-09-19 `own` was ~8.7 GF/s on the DOT family, one
#   FMA chain per output element and therefore FMA-LATENCY bound at a tenth of
#   the core.  It is now register-tiled: 75-80 GF/s on the same M1 core, and at
#   the shapes a streaming step actually issues (4-16 stacked rows) it BEATS
#   Accelerate's AMX, which only pays from m >= 16.  See
#   tests/bench_gemm_shapes, which A/Bs the two arms shape by shape in one
#   process, and .work/threadpool-and-lane.md.
#
# What is still OWED is the same table on Linux against OpenBLAS (S1-6a,
# .work/box-day-plan.md step 4): that measurement is now two commands, not a
# box-day.  If OpenBLAS wins there by enough to cost capacity, this line goes
# back to `openblas` and the note says why -- a default is a decision, and this
# one is reversible by a number (ENGINEERING.md §12).
ifeq ($(UNAME_S),Darwin)
  BLAS ?= accelerate
else
  BLAS ?= none
endif

# Accelerate the FRAMEWORK is linked on every macOS build: Metal needs it and
# so does the vForce exp in mynah_asr_silu.  That is a different question from
# "which library computes sgemm", and conflating the two is how BLAS=none would
# silently change an activation's arithmetic and make the A/B meaningless.
ifeq ($(UNAME_S),Darwin)
  LDFLAGS += -framework Accelerate -framework Metal -framework MetalPerformanceShaders -framework Foundation
  PLATFORM_CFLAGS := -DMYNAH_ASR_ACCELERATE -DACCELERATE_NEW_LAPACK
  METAL_CFLAGS := -DMYNAH_ASR_METAL
  OBJ_EXTRA := build/src/metal_mps.o
else
  LDFLAGS += -lm -lpthread
  PLATFORM_CFLAGS :=
  METAL_CFLAGS :=
endif
BLAS_CFLAGS :=

# Does this build need a cblas header and library at all?  Only `openblas` and
# `accelerate` do; `none` must build on a box that has never seen OpenBLAS, so
# the $(error) below is inside the openblas branch and nowhere else.
ifeq ($(BLAS),accelerate)
  ifneq ($(UNAME_S),Darwin)
    $(error BLAS=accelerate is macOS only. Use BLAS=none (the default here) or BLAS=openblas)
  endif
  BLAS_CFLAGS += -DMYNAH_ASR_BLAS_ACCELERATE
else ifeq ($(BLAS),openblas)
  BLAS_CFLAGS += -DMYNAH_ASR_BLAS_OPENBLAS
  LDFLAGS += -lopenblas
  # fail early with a clear hint instead of "cblas.h: No such file or directory"
  ifeq ($(filter clean,$(MAKECMDGOALS)),)
    ifeq ($(shell printf '\043include <cblas.h>\n' | $(CC) -E -xc - >/dev/null 2>&1 && echo ok),)
      $(error OpenBLAS headers not found. Install them first: `sudo apt install libopenblas-dev` (Debian/Ubuntu) or `sudo dnf install openblas-devel` (Fedora))
    endif
  endif
else ifeq ($(BLAS),none)
  # nothing to add: src/sgemm.c is always compiled and is the provider here
else
  $(error unknown BLAS=$(BLAS). Use none, openblas or accelerate)
endif

CFLAGS += $(PLATFORM_CFLAGS) $(METAL_CFLAGS) $(BLAS_CFLAGS)

# SIMD baseline for the diagnostic builds (debug / ubsan / asan).
#
# THIS IS NOT A TUNING FLAG, it is a coverage one.  Those three targets replace
# CFLAGS wholesale, and until 2026-09-19 they dropped -march with it.  qmat did
# not care -- its kernels carry __attribute__((target(...))) and are chosen by a
# runtime probe, so they are compiled into every build regardless.  src/sgemm.c
# is gated on __ARM_NEON / __AVX2__ instead, so with no -march it silently
# compiled SCALAR: on x86 the sanitizer job was checking a code path production
# never executes, and once BLAS=none became the Linux default it was doing every
# f32 GEMM there at about 2.5 GF/s instead of OpenBLAS's hundreds, instrumented,
# at -O1.  The ASan job went from 6m47s to over its 30-minute timeout, three
# times, and the cause was invisible because "slower under a sanitizer" is what
# one expects.
#
# `--dispatch-map` now reports a scalar sgemm on a vector host as a DOWNGRADE
# rather than as a resolved value, which is how this is caught next time.
SAN_MARCH ?= -march=native

# hook for the recursive variant builds (cuda): these add to the flags the
# Makefile computed instead of overriding CFLAGS (which would lose the quoting of
# MYNAH_ASR_BUILD)
CFLAGS  += $(EXTRA_CFLAGS)
LDFLAGS += $(EXTRA_LDFLAGS)

# ingot: the GGUF/safetensors reader, vendored as a subtree and built by its own
# Makefile so this one never learns how it is compiled.
INGOT_DIR := third_party/ingot
INGOT_LIB := $(INGOT_DIR)/libingot.a
CFLAGS  += -I$(INGOT_DIR)/include
LDFLAGS += $(INGOT_LIB)

SRC := $(wildcard src/*.c) vendor/cJSON.c
OBJ := $(SRC:%.c=build/%.o) $(OBJ_EXTRA)
HDR := $(wildcard src/*.h)

# version injected from git (informational string in `mynah-asr --version`)
MYNAH_ASR_BUILD := $(shell git describe --always --dirty 2>/dev/null || echo dev)
CFLAGS += -DMYNAH_ASR_BUILD='"$(MYNAH_ASR_BUILD)"'

MODEL_DIR ?= models/nemotron-3.5-asr-streaming-0.6b
PARAKEET_DIR ?= models/parakeet-tdt-0.6b-v3
PARAKEET110_DIR ?= models/parakeet-tdt_ctc-110m
VAD_DIR ?= models/silero-vad
VAD_WAV ?= samples/long/en_long.wav

all: mynah-asr mynah-asr-server

mynah-asr: $(OBJ) build/cli/main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

mynah-asr-server: $(OBJ) build/server/main.o build/server/http_util.o build/server/prefork.o \
                  build/server/stream_out.o build/server/slot.o build/server/sched.o \
                  build/server/metrics.o build/server/obs.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -lpthread

# objects in build/ (never next to the sources: the variant builds — ubsan, cuda
# — no longer pollute the normal one)
build/%.o: %.c $(HDR)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

build/src/metal_mps.o: src/metal_mps.m $(HDR)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -fobjc-arc -c $< -o $@

TESTS := tests/test_qmat tests/test_sgemm tests/test_threads tests/test_flags tests/test_align tests/test_vadseg tests/test_tokenize tests/test_stream_out tests/test_features tests/test_subsampling tests/test_encoder tests/test_streaming tests/test_batch tests/test_stream_batch tests/test_kv_layout

$(INGOT_LIB):
	$(MAKE) -C $(INGOT_DIR) lib

$(OBJ): | $(INGOT_LIB)

tests/%: build/tests/%.o build/tests/npy.o build/tests/testcfg.o $(OBJ) $(INGOT_LIB)
	$(CC) $(CFLAGS) -o $@ $(filter %.o,$^) $(LDFLAGS)

# The output writer is server-side and knows nothing about the model: its test
# links that one object and pthreads, nothing else. Explicit rule, so it does
# not drag in libmynah_asr through the pattern rule above.
tests/test_stream_out: build/tests/test_stream_out.o build/server/stream_out.o
	$(CC) $(CFLAGS) -o $@ $^ -lpthread

# Allocation counter for the S1-3 gate: a shared library inserted into the CLI's
# process (DYLD_INSERT_LIBRARIES / LD_PRELOAD). Nothing in src/ links it.
ifeq ($(UNAME_S),Darwin)
  MALLOC_COUNT_LIB := tests/libmalloc_count.dylib
else
  MALLOC_COUNT_LIB := tests/libmalloc_count.so
endif
$(MALLOC_COUNT_LIB): tests/malloc_count.c
	$(CC) -std=c11 -O2 -Wall -Wextra -fPIC -shared -o $@ $< $(if $(filter Darwin,$(UNAME_S)),,-ldl)

# C vs oracle parity (Nemotron streaming + Parakeet TDT offline).
# Skipped (exit 77) when the model or the golden dumps are missing. Regenerate
# with: make golden-dump
PARITY_BOTH := tests/test_features tests/test_subsampling tests/test_encoder tests/test_batch
# tests driven by a shell script (their own arguments): built here, run below
SCRIPTED_TESTS := tests/test_vad
test: $(TESTS) $(SCRIPTED_TESTS) mynah-asr mynah-asr-server examples/minimal
	@for t in $(TESTS); do \
	  if [ $$t = tests/test_qmat ] || [ $$t = tests/test_sgemm ] || [ $$t = tests/test_threads ] || [ $$t = tests/test_flags ] || [ $$t = tests/test_align ] || [ $$t = tests/test_vadseg ] || [ $$t = tests/test_tokenize ] || [ $$t = tests/test_stream_out ]; then $$t; rc=$$?; \
	  elif [ $$t = tests/test_stream_batch ] || [ $$t = tests/test_kv_layout ]; then $$t $(MODEL_DIR); rc=$$?; \
	  else $$t $(MODEL_DIR) tests/audio/test_it.wav tests/golden/test_it; rc=$$?; fi; \
	  if [ $$rc -eq 77 ]; then echo "SKIP $$t: model or golden dumps missing (make golden-dump)"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi; \
	done
	@for spec in "$(PARAKEET_DIR) tests/audio/test_it.wav tests/golden/parakeet_it" \
	             "$(PARAKEET110_DIR) tests/audio/test_en.wav tests/golden/parakeet110_en"; do \
	  for t in $(PARITY_BOTH); do \
	    $$t $$spec; rc=$$?; \
	    if [ $$rc -eq 77 ]; then echo "SKIP $$t ($$spec): model or golden dumps missing"; \
	    elif [ $$rc -ne 0 ]; then exit $$rc; fi; \
	  done; \
	done
	@for m in $(MODEL_DIR) $(PARAKEET_DIR) $(PARAKEET110_DIR) \
	          models/parakeet-rnnt-0.6b models/parakeet-ctc-0.6b \
	          models/parakeet-rnnt-1.1b models/parakeet-ctc-1.1b \
	          models/canary-180m-flash models/canary-1b-flash models/canary-1b-v2; do \
	  sh tests/test_e2e.sh $$m; rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP e2e $$m: not downloaded (HF-native: scripts/download_model.sh + convert_nemo.py; .nemo: curl from the HF repo + convert_nemo.py — see docs/models.md)"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi; \
	done
	@sh tests/test_gguf.sh $(PARAKEET110_DIR); rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP gguf parity: converted 110m or uv missing"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi
	@sh tests/test_kquant.sh; rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP kquant parity: uv or the gguf package missing"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi
	@sh tests/test_vad.sh $(VAD_DIR); rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP vad parity: $(VAD_DIR)/silero_vad.onnx or uv missing (see tests/test_vad.sh)"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi
	@sh tests/test_server_stream.sh $(MODEL_DIR); rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP server-stream: model, binaries or python3 missing"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi
	@sh tests/test_server_protocol.sh $(MODEL_DIR); rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP server-protocol: model, binaries or python3 missing"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi
	@sh tests/test_server_metrics.sh $(CONC_MODEL_DIR); rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP server-metrics: model, binaries, curl or python3 missing"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi
	@sh tests/test_server_models.sh $(MODEL_DIR) $(MULTI_MODEL_DIR); rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP server-models: needs BOTH $(MODEL_DIR) and $(MULTI_MODEL_DIR)"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi
	@$(MAKE) --no-print-directory test-stream-allocs
	@o=`python3 tools/bench/streaming_metrics.py --self-test` || { echo "$$o"; exit 1; }; echo "$$o" | tail -1
	@$(MAKE) --no-print-directory test-stream-batch-allocs

# S1-4: the BATCHED step allocates nothing per step either. Same counter, but
# sampled in-process (the batched API has no CLI entry to difference two runs of).
# Under a sanitizer the counter cannot be preloaded (ASan must be the first
# library in the process), so the gate is skipped there rather than failing
# before the test can say 77.
ifneq (,$(findstring sanitize,$(CFLAGS)))
test-stream-batch-allocs: tests/test_stream_batch
	@echo "SKIP stream-batch-allocs: sanitized build, the malloc counter cannot be preloaded"
test-stream-allocs: mynah-asr
	@echo "SKIP stream-allocs: sanitized build, the malloc counter cannot be preloaded"
else
test-stream-batch-allocs: tests/test_stream_batch $(MALLOC_COUNT_LIB)
	@if [ "$(UNAME_S)" = "Darwin" ]; then \
	  DYLD_INSERT_LIBRARIES=$(MALLOC_COUNT_LIB) DYLD_FORCE_FLAT_NAMESPACE=1 \
	  MALLOC_COUNT_OUT=/dev/null tests/test_stream_batch $(MODEL_DIR) --allocs; rc=$$?; \
	else \
	  LD_PRELOAD=$(MALLOC_COUNT_LIB) MALLOC_COUNT_OUT=/dev/null \
	  tests/test_stream_batch $(MODEL_DIR) --allocs; rc=$$?; \
	fi; \
	if [ $$rc -eq 77 ]; then echo "SKIP stream-batch-allocs: model missing or interposition unavailable"; \
	elif [ $$rc -ne 0 ]; then exit $$rc; fi
# R-8: the diagnostic injection must not leak into anything a user can see
# (model-gated; 77 without a converted streaming pack).
test-rnnt-inject: mynah-asr
	@MODEL_DIR=$(MODEL_DIR) sh tests/test_rnnt_inject.sh; rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP rnnt-inject: no converted streaming pack"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi

# S1-3: zero allocations per streaming chunk after warm-up (model-gated).
test-stream-allocs: mynah-asr $(MALLOC_COUNT_LIB)
	@sh tests/test_stream_allocs.sh $(MODEL_DIR) tests/audio/test_it.wav; rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP stream-allocs: model missing or interposition unavailable"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi

endif

golden-dump:
	cd tools && uv run python -m oracle.transcribe ../$(MODEL_DIR) ../tests/audio/test_it.wav \
	  --lang it-IT --dump-dir ../tests/golden/test_it
	@if [ -f $(PARAKEET_DIR)/mynah.json ]; then \
	  cd tools && uv run python -m oracle.transcribe ../$(PARAKEET_DIR) \
	    ../tests/audio/test_it.wav --dump-dir ../tests/golden/parakeet_it; fi
	@if [ -f $(PARAKEET110_DIR)/mynah.json ]; then \
	  cd tools && uv run python -m oracle.transcribe ../$(PARAKEET110_DIR) \
	    ../tests/audio/test_en.wav --dump-dir ../tests/golden/parakeet110_en; fi

# static library (without the CLI)
lib: libmynah_asr.a
libmynah_asr.a: $(OBJ)
	ar rcs $@ $^

# shared library (for the bindings: Python ctypes, Node, ...)
ifeq ($(UNAME_S),Darwin)
  SOEXT := .dylib
else
  SOEXT := .so
endif
shared: libmynah_asr$(SOEXT)
libmynah_asr$(SOEXT): $(OBJ)
	$(CC) $(CFLAGS) -shared -o $@ $^ $(LDFLAGS)

# API example (built by `make test`: a guard on the public surface)
example: examples/minimal
examples/minimal: examples/minimal.c libmynah_asr.a
	$(CC) $(CFLAGS) -o $@ examples/minimal.c libmynah_asr.a $(LDFLAGS)

# CUDA (Linux, needs nvcc): the large GEMMs on GPU through cuBLAS. Validated on
# an A100 (2026-07-20): output identical to CPU on every model, RTF in
# docs/benchmarks.md.
NVCC ?= nvcc
cuda:
	$(MAKE) clean && $(MAKE) EXTRA_CFLAGS="-DMYNAH_ASR_CUDA" \
	  OBJ_EXTRA="build/src/cuda_gemm.o" \
	  EXTRA_LDFLAGS="-lcublas -lcudart -L/usr/local/cuda/lib64"

build/src/cuda_gemm.o: src/cuda_gemm.cu
	@mkdir -p $(@D)
	$(NVCC) -O3 -DMYNAH_ASR_CUDA -c $< -o $@

# alternative builds.
# Memory/UB policy on macOS: `make leaks` (native, fast) + `make ubsan` (low
# overhead). ASan is VERY SLOW on a Mac and tends to hang with the large model:
# Linux CI only.
debug:
	$(MAKE) clean && $(MAKE) CFLAGS="-std=c11 -O0 -g $(SAN_MARCH) -Wall -Wextra -iquote src -I$(INGOT_DIR)/include -D_DEFAULT_SOURCE $(PLATFORM_CFLAGS) $(BLAS_CFLAGS)"
# NOTE: clean at the end too — the sanitized objects (without -DMYNAH_ASR_METAL
# and referencing the ubsan runtime) must NOT be left behind to pollute the
# normal build
ubsan:
	$(MAKE) clean && $(MAKE) CFLAGS="-std=c11 -O2 -g $(SAN_MARCH) -fsanitize=undefined \
	  -fno-omit-frame-pointer -Wall -Wextra -iquote src -I$(INGOT_DIR)/include -D_DEFAULT_SOURCE $(PLATFORM_CFLAGS) $(BLAS_CFLAGS)" \
	  LDFLAGS="$(LDFLAGS) -fsanitize=undefined" all test && $(MAKE) clean
# -O2, not -O1, and it is not a cosmetic difference.  Since BLAS=none became the
# Linux default this job sanitizes OUR f32 GEMM instead of calling into an
# uninstrumented OpenBLAS, so the sanitizer now pays for all of the arithmetic.
# Measured on the M1, our DOT family through tests/bench_gemm_shapes with ASan
# on: 21.9 GF/s at -O1 against 60.4 at -O2, with the uninstrumented production
# build at 172.  -fno-omit-frame-pointer keeps the stack traces readable, which
# is the only thing -O1 was buying, and the ubsan target above has always been
# -O2 -- this one was the outlier.
asan:
	$(MAKE) clean && $(MAKE) CFLAGS="-std=c11 -O2 -g $(SAN_MARCH) -fsanitize=address,undefined \
	  -fno-omit-frame-pointer -Wall -Wextra -iquote src -I$(INGOT_DIR)/include -D_DEFAULT_SOURCE $(PLATFORM_CFLAGS) $(BLAS_CFLAGS)" \
	  LDFLAGS="$(LDFLAGS) -fsanitize=address,undefined" all test && $(MAKE) clean

# reproducible bench: warm RTF + peak RAM for every model present
bench: mynah-asr
	@sh tests/bench.sh

# Batch throughput (parallel requests simulated through mynah_asr_transcribe_batch):
# how many times realtime the backend sustains as the batch grows. Meant for GPUs
# (make cuda; --backend cuda) but it runs on cpu/metal too.
#   tests/bench_throughput models/<m> tests/audio/long_60s.wav --backend cuda --max-batch 64
# The f32 provider A/B, shape by shape, in one process: `own` against whatever
# this build linked. With a profile it weights each shape by the call count the
# model really issued:
#   MYNAH_ASR_GEMM_PROFILE=1 ./mynah-asr transcribe -m <model> a.wav 2> shapes.txt
#   tests/bench_gemm_shapes shapes.txt
# Model-free without one: `tests/bench_gemm_shapes --demo`.
# Three tools, three verbs, and they are separate on purpose:
#   box-doctor   DESCRIBES the machine and refuses when it is not fit to measure
#   box-advise   PREDICTS capacity and RECOMMENDS a configuration (calibrates first)
#   box_qualify  MEASURES, and only a SOAK promotes
# The advisor fits THIS machine's step table rather than carrying a constant in from
# another box, which is the whole reason to trust its arithmetic.
box-doctor:
	sh tools/bench/box_doctor.sh

box-advise: tests/test_stream_batch mynah-asr
	@test -n "$(MODEL_DIR)" || { echo "usage: make box-advise MODEL_DIR=models/<pack>"; exit 2; }
	python3 tools/bench/box_advisor.py -m $(MODEL_DIR) $(ADVISOR_ARGS)

bench-gemm: tests/bench_gemm_shapes
	@echo "usage: tests/bench_gemm_shapes <profile>|--demo [--batches N] [--target-ms N]"
	@echo "       MYNAH_ASR_GEMM_PROFILE=1 ./mynah-asr transcribe -m <model> a.wav 2> shapes.txt"

bench-throughput: tests/bench_throughput
	@echo "usage: tests/bench_throughput <model_dir> <wav> [--backend cuda] [--max-batch N] [--runs R]"

# S4-2 streaming load against a RUNNING server (see docs/serving.md for the order of work).
# WAVE screens and may disqualify; only a SOAK with a drift gate promotes.
# Override: make bench-stream-soak STREAM_N=8 STREAM_PORT=8090 STREAM_DURATION=900
STREAM_CLIPS ?= samples/*/fleurs_*.wav tests/audio/test_*.wav
STREAM_PORT ?= 8090
STREAM_N ?= 4
STREAM_DURATION ?= 600
bench-stream-wave:
	@python3 tools/bench/stream_load.py --mode wave --streams $(STREAM_N) --repeat 2 --port $(STREAM_PORT) --clips $(STREAM_CLIPS) --json wave-$(STREAM_N).json
bench-stream-soak:
	@python3 tools/bench/stream_load.py --mode soak --streams $(STREAM_N) --duration $(STREAM_DURATION) --warmup 30 --window 60 --bank short,medium,long --seed 42 --port $(STREAM_PORT) --clips $(STREAM_CLIPS) --json soak-$(STREAM_N).json

# The three tiers: three questions, three budgets, one command each. See the
# header of tools/bench/tier.sh for why they exist and what each refuses.
#   tier0 ~90 s   does this build serve correctly at all (run it after a change)
#   tier1 ~8 min  where is the knee on THIS host (bisection, one warm server)
#   tier2 ~30 min certify one point: two soaks that must agree, with references
# PARAKEET_DIR gives tier 0 its light-model REST probe.
TIER_C ?= 0
tier0: mynah-asr mynah-asr-server
	@sh tools/bench/tier.sh 0 -m $(MODEL_DIR) $(if $(wildcard $(PARAKEET110_DIR)),--parakeet $(PARAKEET110_DIR),)
tier1: mynah-asr mynah-asr-server
	@sh tools/bench/tier.sh 1 -m $(MODEL_DIR)
tier2: mynah-asr mynah-asr-server
	@if [ "$(TIER_C)" = "0" ]; then \
	  echo "tier2 needs the concurrency tier1 screened: make tier2 TIER_C=32"; exit 2; fi
	@sh tools/bench/tier.sh 2 -m $(MODEL_DIR) -C $(TIER_C)

# The quality baseline a later run is compared against, and the regression gate.
# QUALITY_BASELINE defaults to the file cer_offline writes for this model+quant.
QUALITY_BASELINE ?= configs/quality/$(notdir $(MODEL_DIR))-$(QUANT)-offline.json
QUANT ?= int8
cer-baseline: mynah-asr
	@mkdir -p configs/quality
	@python3 tools/eval/cer_offline.py --model $(MODEL_DIR) --quant $(QUANT) \
	  --manifest samples/manifest.json --json $(QUALITY_BASELINE)
cer-check: mynah-asr
	@python3 tools/eval/cer_offline.py --model $(MODEL_DIR) --quant $(QUANT) \
	  --manifest samples/manifest.json --baseline $(QUALITY_BASELINE)

# End-to-end server test (REST + concurrency + WebSocket)
test-server: mynah-asr-server
	@sh tests/test_server.sh $(MODEL_DIR); rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP test-server: model missing"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi
	@sh tests/test_serve_repro.sh $(MODEL_DIR); rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP serve-repro: model missing"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi
	@sh tests/test_server_concurrency.sh $(MODEL_DIR); rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP server-concurrency: model missing"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi

# Concurrent WebSocket streaming: identity against the CLI under 4 real-time
# streams, slow-reader isolation, the same identity under --prefork. Needs a
# streaming model (Nemotron), so it is gated like test-server.
test-server-stream: mynah-asr-server mynah-asr
	@sh tests/test_server_stream.sh $(MODEL_DIR); rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP server-stream: model, binaries or python3 missing"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi

# WebSocket protocol v2 and the per-worker admission ladder (S2-3 / S2-5):
# three utterances on one socket byte-identical to the CLI, unknown control and
# unserved language survived, the pre-upgrade 400/503, idle, pings, SIGTERM.
# Needs a streaming model, so it is gated like test-server-stream.
test-server-protocol: mynah-asr-server mynah-asr
	@sh tests/test_server_protocol.sh $(MODEL_DIR); rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP server-protocol: model, binaries or python3 missing"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi

# Model-agnostic server check (concurrency + adaptive-BLAS accounting): unlike
# test-server it asserts nothing about the transcript, so it runs with ANY
# converted model. CI uses it with the 110m (CONC_MODEL_DIR=...), which is how
# the server finally gets exercised there at all.
CONC_MODEL_DIR ?= $(PARAKEET110_DIR)
# S3-3/S3-4: the banner, /v1/health as facts, /metrics on its own port (the
# token bucket, the double bind, the router's fleet view) and the SIGUSR1 dump.
# Model-agnostic and REST-only, so it runs wherever test-server-concurrency does.
test-server-metrics: mynah-asr-server
	@sh tests/test_server_metrics.sh $(CONC_MODEL_DIR); rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP server-metrics: model, binaries, curl or python3 missing"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi

# S2-6: SEVERAL MODELS IN ONE FLEET. One server, two worker groups (the
# streaming pack and an offline-only one), routing by model: /v1/models from the
# table, a REST transcription to each group byte-identical to THAT model's CLI,
# a WebSocket to the streaming group, 400 model_not_streaming on the offline
# one, 404 model_not_found on a name nobody holds, capacity refused per group,
# survivors=0. Needs BOTH models, so it is gated on both.
MULTI_MODEL_DIR ?= models/parakeet-tdt_ctc-110m-gguf
test-server-models: mynah-asr-server mynah-asr
	@sh tests/test_server_models.sh $(MODEL_DIR) $(MULTI_MODEL_DIR); rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP server-models: needs BOTH $(MODEL_DIR) and $(MULTI_MODEL_DIR)"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi

test-server-concurrency: mynah-asr-server
	@sh tests/test_server_concurrency.sh $(CONC_MODEL_DIR); rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP server-concurrency: model missing"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi

# Silero VAD parity on its own (no ASR model needed: the VAD is 2.3 MB and the
# oracle is onnxruntime). fetch-vad downloads the checkpoint.
test-vad: tests/test_vad
	@sh tests/test_vad.sh $(VAD_DIR); rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP vad parity: run make fetch-vad first"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi

# Speech-span parity against silero's own get_speech_timestamps. Separate target
# because it needs the silero-vad package (hence torch): the hysteresis itself is
# covered model-free by tests/test_vadseg, this pins it to the reference.
test-vad-spans: tests/test_vad
	@VAD_SPANS=1 sh tests/test_vad.sh $(VAD_DIR) $(VAD_WAV); rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP vad spans: run make fetch-vad first"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi

SILERO_URL := https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
fetch-vad:
	@mkdir -p $(VAD_DIR)
	@if [ -f $(VAD_DIR)/silero_vad.onnx ]; then echo "already there: $(VAD_DIR)/silero_vad.onnx"; \
	 else curl -sSLo $(VAD_DIR)/silero_vad.onnx $(SILERO_URL) && \
	      echo "downloaded $(VAD_DIR)/silero_vad.onnx (Silero VAD v5, MIT)"; fi

# Multilingual suite: real audio samples (Tatoeba, CC) for every supported
# language, checking language detection + CER against the reference text.
# First time: make fetch-lang-samples (needs ffmpeg + tools/ uv).
test-nemo-langs: mynah-asr
	@cd tools && uv run python -m eval.test_langs; rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP test-nemo-langs: samples or model missing"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi

fetch-lang-samples:
	cd tools && uv run python fetch_lang_samples.py 3

# English STRESS BANK for the streaming load harness: FLEURS (CC-BY 4.0) clips
# stratified over the short/medium/long classes stream_load.py uses, sized so a
# 10-min SOAK at c=32 never replays a clip. ~1.8 GB of one-time transfer, ~640 MB
# on disk, NOT committed (samples/stress-en is gitignored except manifest+README).
# make fetch-stress-bank STRESS_BANK_ARGS="--dry-run"   # the plan, no audio
STRESS_BANK_ARGS ?=
fetch-stress-bank:
	cd tools && uv run python fetch_stress_bank.py $(STRESS_BANK_ARGS)

# Quality on real audio (the committed samples/ FLEURS): ASR CER + Canary
# translation quality against parallel references, cpu+metal backends.
test-samples: mynah-asr
	cd tools && uv run python -m eval.test_samples; rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP test-samples: samples/ or models missing"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi

# fast leak check on macOS (the native `leaks` tool, no rebuild — ASan is very
# slow on a Mac: use it only in Linux CI. Same pattern as qwen-tts).
leaks: mynah-asr tests/test_streaming tests/test_vad tests/test_align tests/test_stream_out
	leaks --atExit -- tests/test_align 2>&1 | tail -2
	leaks --atExit -- tests/test_stream_out 2>&1 | tail -2
	leaks --atExit -- ./mynah-asr transcribe -m $(MODEL_DIR) -i tests/audio/test_it.wav \
	  --lang it-IT 2>&1 | tail -3
	leaks --atExit -- tests/test_streaming $(MODEL_DIR) tests/audio/test_it.wav \
	  tests/golden/test_it 2>&1 | tail -3
	@WRAP="leaks --atExit --" sh tests/test_vad.sh $(VAD_DIR) 2>&1 | tail -4

# plan + repository integrity (ENGINEERING.md §1, §12): no dangling board links,
# no tracked script depending on an untracked file
check:
	@sh tests/test_check_plan.sh
	@out=$$(python3 tools/bench/streaming_metrics.py --self-test) || { echo "$$out"; exit 1; }; echo "$$out" | tail -1
	@out=$$(sh tests/test_partial_quality.sh) || { echo "$$out"; exit 1; }; echo "$$out" | tail -1
	@out=$$(sh tests/test_v2_verdict.sh) || { echo "$$out"; exit 1; }; echo "$$out" | tail -1
	@out=$$(sh tests/test_v2_promote.sh) || { echo "$$out"; exit 1; }; echo "$$out" | tail -1
	@out=$$(sh tests/test_rnnt_trace_parsers.sh) || { echo "$$out"; exit 1; }; echo "$$out" | tail -1
	@python3 tools/check_plan.py
	@python3 tools/check_repo_integrity.py
	@python3 tools/check_flag_registry.py

clean:
	rm -rf build mynah-asr mynah-asr-server libmynah_asr.a $(TESTS) $(SCRIPTED_TESTS) examples/minimal dist \
	       tests/libmalloc_count.dylib tests/libmalloc_count.so
	@# Without this, libingot.a survives a clean: update the subtree and the
	@# next build silently links the previous library.
	@test -d $(INGOT_DIR) && $(MAKE) -C $(INGOT_DIR) clean || true

# Refresh the vendored ingot subtree from upstream. A plain clone already
# contains ingot (subtree = real files in-tree, nothing to init); this is only
# needed to pick up new upstream commits. Requires a clean working tree.
update-ingot:
	git subtree pull --prefix $(INGOT_DIR) https://github.com/mynah-org/ingot.git main --squash
	@$(MAKE) -C $(INGOT_DIR) clean

# install: CLI + server + static library + header
PREFIX ?= /usr/local
install: mynah-asr mynah-asr-server libmynah_asr.a
	install -d $(DESTDIR)$(PREFIX)/bin $(DESTDIR)$(PREFIX)/lib $(DESTDIR)$(PREFIX)/include
	install -m 755 mynah-asr mynah-asr-server $(DESTDIR)$(PREFIX)/bin/
	install -m 644 libmynah_asr.a $(DESTDIR)$(PREFIX)/lib/
	install -m 644 src/mynah_asr.h $(DESTDIR)$(PREFIX)/include/

# Release tarball: CLI + server + static library + header + licence/readme +
# model download script. Name = git version + os + arch; binaries stripped and
# SHA-256 checksummed. Everything in dist/ (gitignored). Usage: make dist
DIST_OS   := $(shell uname -s | tr '[:upper:]' '[:lower:]')
DIST_ARCH := $(shell uname -m)
DIST_NAME := mynah-asr-$(MYNAH_ASR_BUILD)-$(DIST_OS)-$(DIST_ARCH)
DIST_DIR  := dist/$(DIST_NAME)
dist: mynah-asr mynah-asr-server libmynah_asr.a
	@rm -rf $(DIST_DIR)
	@mkdir -p $(DIST_DIR)/bin $(DIST_DIR)/lib $(DIST_DIR)/include $(DIST_DIR)/scripts
	install -m 755 mynah-asr mynah-asr-server $(DIST_DIR)/bin/
	install -m 644 libmynah_asr.a $(DIST_DIR)/lib/
	install -m 644 src/mynah_asr.h $(DIST_DIR)/include/
	install -m 644 LICENSE README.md $(DIST_DIR)/
	install -m 755 scripts/download_model.sh $(DIST_DIR)/scripts/
	@strip $(DIST_DIR)/bin/mynah-asr $(DIST_DIR)/bin/mynah-asr-server 2>/dev/null || true
	cd dist && tar czf $(DIST_NAME).tar.gz $(DIST_NAME)
	@rm -rf $(DIST_DIR)
	@echo "" && echo "-> dist/$(DIST_NAME).tar.gz"
	@cd dist && shasum -a 256 $(DIST_NAME).tar.gz 2>/dev/null || (cd dist && sha256sum $(DIST_NAME).tar.gz)

.PHONY: all clean check bench-gemm bench-throughput box-doctor box-advise install dist test golden-dump lib shared example debug ubsan asan bench leaks test-vad test-vad-spans fetch-vad test-nemo-langs fetch-lang-samples fetch-stress-bank test-server test-server-stream test-server-protocol test-server-concurrency test-samples test-stream-allocs bench-stream-wave bench-stream-soak cuda update-ingot test-stream-batch-allocs test-server-metrics
