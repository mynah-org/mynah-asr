# Mynah — build. CPU-first: BLAS = Accelerate (macOS) / OpenBLAS (Linux).
CC      ?= cc
CFLAGS  ?= -std=c11 -O3 -march=native -ffast-math -Wall -Wextra -iquote src -D_DEFAULT_SOURCE
LDFLAGS ?=

CFLAGS += -fPIC

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  LDFLAGS += -framework Accelerate -framework Metal -framework MetalPerformanceShaders -framework Foundation
  BLAS_DEF := MYNAH_ASR_BLAS_ACCELERATE
  CFLAGS  += -DMYNAH_ASR_BLAS_ACCELERATE -DACCELERATE_NEW_LAPACK -DMYNAH_ASR_METAL
  OBJ_EXTRA := build/src/metal_mps.o
else
  LDFLAGS += -lopenblas -lm -lpthread
  BLAS_DEF := MYNAH_ASR_BLAS_OPENBLAS
  CFLAGS  += -DMYNAH_ASR_BLAS_OPENBLAS
  # fail early with a clear hint instead of "cblas.h: No such file or directory"
  ifeq ($(filter clean,$(MAKECMDGOALS)),)
    ifeq ($(shell printf '\043include <cblas.h>\n' | $(CC) -E -xc - >/dev/null 2>&1 && echo ok),)
      $(error OpenBLAS headers not found. Install them first: `sudo apt install libopenblas-dev` (Debian/Ubuntu) or `sudo dnf install openblas-devel` (Fedora))
    endif
  endif
endif

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
                  build/server/stream_out.o build/server/slot.o build/server/sched.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -lpthread

# objects in build/ (never next to the sources: the variant builds — ubsan, cuda
# — no longer pollute the normal one)
build/%.o: %.c $(HDR)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

build/src/metal_mps.o: src/metal_mps.m $(HDR)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -fobjc-arc -c $< -o $@

TESTS := tests/test_qmat tests/test_threads tests/test_flags tests/test_align tests/test_vadseg tests/test_tokenize tests/test_stream_out tests/test_features tests/test_subsampling tests/test_encoder tests/test_streaming tests/test_batch tests/test_stream_batch

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
	  if [ $$t = tests/test_qmat ] || [ $$t = tests/test_threads ] || [ $$t = tests/test_flags ] || [ $$t = tests/test_align ] || [ $$t = tests/test_vadseg ] || [ $$t = tests/test_tokenize ] || [ $$t = tests/test_stream_out ]; then $$t; rc=$$?; \
	  elif [ $$t = tests/test_stream_batch ]; then $$t $(MODEL_DIR); rc=$$?; \
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
	$(MAKE) clean && $(MAKE) CFLAGS="-std=c11 -O0 -g -Wall -Wextra -iquote src -I$(INGOT_DIR)/include -D_DEFAULT_SOURCE -D$(BLAS_DEF)"
# NOTE: clean at the end too — the sanitized objects (without -DMYNAH_ASR_METAL
# and referencing the ubsan runtime) must NOT be left behind to pollute the
# normal build
ubsan:
	$(MAKE) clean && $(MAKE) CFLAGS="-std=c11 -O2 -g -fsanitize=undefined \
	  -fno-omit-frame-pointer -Wall -Wextra -iquote src -I$(INGOT_DIR)/include -D_DEFAULT_SOURCE -D$(BLAS_DEF) -DACCELERATE_NEW_LAPACK" \
	  LDFLAGS="$(LDFLAGS) -fsanitize=undefined" all test && $(MAKE) clean
asan:
	$(MAKE) clean && $(MAKE) CFLAGS="-std=c11 -O1 -g -fsanitize=address,undefined \
	  -fno-omit-frame-pointer -Wall -Wextra -iquote src -I$(INGOT_DIR)/include -D_DEFAULT_SOURCE -D$(BLAS_DEF) -DACCELERATE_NEW_LAPACK" \
	  LDFLAGS="$(LDFLAGS) -fsanitize=address,undefined" all test && $(MAKE) clean

# reproducible bench: warm RTF + peak RAM for every model present
bench: mynah-asr
	@sh tests/bench.sh

# Batch throughput (parallel requests simulated through mynah_asr_transcribe_batch):
# how many times realtime the backend sustains as the batch grows. Meant for GPUs
# (make cuda; --backend cuda) but it runs on cpu/metal too.
#   tests/bench_throughput models/<m> tests/audio/long_60s.wav --backend cuda --max-batch 64
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

.PHONY: all clean check install dist test golden-dump lib shared example debug ubsan asan bench leaks test-vad test-vad-spans fetch-vad test-nemo-langs fetch-lang-samples test-server test-server-stream test-server-protocol test-server-concurrency test-samples test-stream-allocs bench-stream-wave bench-stream-soak cuda update-ingot test-stream-batch-allocs
