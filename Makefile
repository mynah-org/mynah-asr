# Mynah — build. CPU-first: BLAS = Accelerate (macOS) / OpenBLAS (Linux).
CC      ?= cc
CFLAGS  ?= -std=c11 -O3 -march=native -ffast-math -Wall -Wextra -iquote src -D_DEFAULT_SOURCE
LDFLAGS ?=

CFLAGS += -fPIC

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  LDFLAGS += -framework Accelerate -framework Metal -framework MetalPerformanceShaders -framework Foundation
  BLAS_DEF := MYNAH_BLAS_ACCELERATE
  CFLAGS  += -DMYNAH_BLAS_ACCELERATE -DACCELERATE_NEW_LAPACK -DMYNAH_METAL
  OBJ_EXTRA := build/src/metal_mps.o
else
  LDFLAGS += -lopenblas -lm -lpthread
  BLAS_DEF := MYNAH_BLAS_OPENBLAS
  CFLAGS  += -DMYNAH_BLAS_OPENBLAS
  # fail early with a clear hint instead of "cblas.h: No such file or directory"
  ifeq ($(filter clean,$(MAKECMDGOALS)),)
    ifeq ($(shell printf '\043include <cblas.h>\n' | $(CC) -E -xc - >/dev/null 2>&1 && echo ok),)
      $(error OpenBLAS headers not found. Install them first: `sudo apt install libopenblas-dev` (Debian/Ubuntu) or `sudo dnf install openblas-devel` (Fedora))
    endif
  endif
endif

# hook per le build varianti ricorsive (cuda): si sommano ai flag calcolati dal
# Makefile invece di sovrascrivere CFLAGS (che perderebbe il quoting di MYNAH_BUILD)
CFLAGS  += $(EXTRA_CFLAGS)
LDFLAGS += $(EXTRA_LDFLAGS)

SRC := $(wildcard src/*.c) vendor/cJSON.c
OBJ := $(SRC:%.c=build/%.o) $(OBJ_EXTRA)
HDR := $(wildcard src/*.h)

# versione iniettata da git (stringa informativa in `mynah --version`)
MYNAH_BUILD := $(shell git describe --always --dirty 2>/dev/null || echo dev)
CFLAGS += -DMYNAH_BUILD='"$(MYNAH_BUILD)"'

MODEL_DIR ?= models/nemotron-3.5-asr-streaming-0.6b
PARAKEET_DIR ?= models/parakeet-tdt-0.6b-v3
PARAKEET110_DIR ?= models/parakeet-tdt_ctc-110m

all: mynah mynah-server

mynah: $(OBJ) build/cli/main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

mynah-server: $(OBJ) build/server/main.o build/server/http_util.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -lpthread

# oggetti in build/ (mai accanto ai sorgenti: le build varianti — ubsan, cuda —
# non inquinano più quella normale)
build/%.o: %.c $(HDR)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

build/src/metal_mps.o: src/metal_mps.m $(HDR)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -fobjc-arc -c $< -o $@

TESTS := tests/test_qmat tests/test_gguf tests/test_features tests/test_subsampling tests/test_encoder tests/test_streaming tests/test_batch

tests/%: build/tests/%.o build/tests/npy.o build/tests/testcfg.o $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Parità C vs oracolo (Nemotron streaming + Parakeet TDT offline).
# Skip (exit 77) se mancano modello o dump golden. Rigenera con: make golden-dump
PARITY_BOTH := tests/test_features tests/test_subsampling tests/test_encoder tests/test_batch
test: $(TESTS) mynah examples/minimal
	@for t in $(TESTS); do \
	  if [ $$t = tests/test_qmat ] || [ $$t = tests/test_gguf ]; then $$t; rc=$$?; \
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

golden-dump:
	cd tools && uv run python -m oracle.transcribe ../$(MODEL_DIR) ../tests/audio/test_it.wav \
	  --lang it-IT --dump-dir ../tests/golden/test_it
	@if [ -f $(PARAKEET_DIR)/mynah.json ]; then \
	  cd tools && uv run python -m oracle.transcribe ../$(PARAKEET_DIR) \
	    ../tests/audio/test_it.wav --dump-dir ../tests/golden/parakeet_it; fi
	@if [ -f $(PARAKEET110_DIR)/mynah.json ]; then \
	  cd tools && uv run python -m oracle.transcribe ../$(PARAKEET110_DIR) \
	    ../tests/audio/test_en.wav --dump-dir ../tests/golden/parakeet110_en; fi

# libreria statica (senza CLI)
lib: libmynah.a
libmynah.a: $(OBJ)
	ar rcs $@ $^

# libreria condivisa (per i bindings: Python ctypes, Node, ...)
ifeq ($(UNAME_S),Darwin)
  SOEXT := .dylib
else
  SOEXT := .so
endif
shared: libmynah$(SOEXT)
libmynah$(SOEXT): $(OBJ)
	$(CC) $(CFLAGS) -shared -o $@ $^ $(LDFLAGS)

# esempio API (compilato in `make test`: guardia sulla superficie pubblica)
example: examples/minimal
examples/minimal: examples/minimal.c libmynah.a
	$(CC) $(CFLAGS) -o $@ examples/minimal.c libmynah.a $(LDFLAGS)

# CUDA (Linux, richiede nvcc): GEMM grandi su GPU via cuBLAS. Validato su
# A100 (2026-07-20): output identico a CPU su tutti i modelli, RTF in docs/benchmarks.md.
NVCC ?= nvcc
cuda:
	$(MAKE) clean && $(MAKE) EXTRA_CFLAGS="-DMYNAH_CUDA" \
	  OBJ_EXTRA="build/src/cuda_gemm.o" \
	  EXTRA_LDFLAGS="-lcublas -lcudart -L/usr/local/cuda/lib64"

build/src/cuda_gemm.o: src/cuda_gemm.cu
	@mkdir -p $(@D)
	$(NVCC) -O3 -DMYNAH_CUDA -c $< -o $@

# build alternative.
# Policy memoria/UB su macOS: `make leaks` (nativo, veloce) + `make ubsan` (overhead
# basso). ASan è LENTISSIMO su Mac e tende a impallarsi col modello grande: solo CI Linux.
debug:
	$(MAKE) clean && $(MAKE) CFLAGS="-std=c11 -O0 -g -Wall -Wextra -iquote src -D_DEFAULT_SOURCE -D$(BLAS_DEF)"
# NOTA: clean anche in coda — gli oggetti sanitizzati (senza -DMYNAH_METAL e con
# riferimenti al runtime ubsan) NON devono restare a inquinare la build normale
ubsan:
	$(MAKE) clean && $(MAKE) CFLAGS="-std=c11 -O2 -g -fsanitize=undefined \
	  -fno-omit-frame-pointer -Wall -Wextra -iquote src -D_DEFAULT_SOURCE -D$(BLAS_DEF) -DACCELERATE_NEW_LAPACK" \
	  LDFLAGS="$(LDFLAGS) -fsanitize=undefined" all test && $(MAKE) clean
asan:
	$(MAKE) clean && $(MAKE) CFLAGS="-std=c11 -O1 -g -fsanitize=address,undefined \
	  -fno-omit-frame-pointer -Wall -Wextra -iquote src -D_DEFAULT_SOURCE -D$(BLAS_DEF) -DACCELERATE_NEW_LAPACK" \
	  LDFLAGS="$(LDFLAGS) -fsanitize=address,undefined" all test && $(MAKE) clean

# bench riproducibile: RTF warm + picco RAM per ogni modello presente
bench: mynah
	@sh tests/bench.sh

# Throughput batch (richieste parallele simulate via mynah_transcribe_batch):
# quante volte il realtime regge il backend al crescere del batch. Pensato per
# GPU (make cuda; --backend cuda) ma gira anche su cpu/metal.
#   tests/bench_throughput models/<m> tests/audio/long_60s.wav --backend cuda --max-batch 64
bench-throughput: tests/bench_throughput
	@echo "usage: tests/bench_throughput <model_dir> <wav> [--backend cuda] [--max-batch N] [--runs R]"

# Test end-to-end del server (REST + concorrenza + WebSocket)
test-server: mynah-server
	@sh tests/test_server.sh $(MODEL_DIR); rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP test-server: model missing"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi
	@sh tests/test_serve_repro.sh $(MODEL_DIR); rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP serve-repro: model missing"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi

# Suite multilingua: sample audio reali (Tatoeba, CC) per ogni lingua supportata,
# verifica language detection + CER vs testo di riferimento.
# Prima volta: make fetch-lang-samples (richiede ffmpeg + tools/ uv).
test-nemo-langs: mynah
	cd tools && uv run python -m eval.test_langs

fetch-lang-samples:
	cd tools && uv run python fetch_lang_samples.py 3

# Qualità su audio reale (samples/ FLEURS committati): CER ASR + qualità
# traduzione Canary vs riferimenti paralleli, backend cpu+metal.
test-samples: mynah
	cd tools && uv run python -m eval.test_samples; rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP test-samples: samples/ or models missing"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi

# leak check veloce su macOS (tool nativo `leaks`, nessuna rebuild — su Mac ASan
# è lentissimo: usarlo solo in CI Linux. Stesso pattern di qwen-tts).
leaks: mynah tests/test_streaming
	leaks --atExit -- ./mynah transcribe -m $(MODEL_DIR) -i tests/audio/test_it.wav \
	  --lang it-IT 2>&1 | tail -3
	leaks --atExit -- tests/test_streaming $(MODEL_DIR) tests/audio/test_it.wav \
	  tests/golden/test_it 2>&1 | tail -3

clean:
	rm -rf build mynah mynah-server libmynah.a $(TESTS) examples/minimal dist

# installazione: CLI + server + libreria statica + header
PREFIX ?= /usr/local
install: mynah mynah-server libmynah.a
	install -d $(DESTDIR)$(PREFIX)/bin $(DESTDIR)$(PREFIX)/lib $(DESTDIR)$(PREFIX)/include
	install -m 755 mynah mynah-server $(DESTDIR)$(PREFIX)/bin/
	install -m 644 libmynah.a $(DESTDIR)$(PREFIX)/lib/
	install -m 644 src/mynah.h $(DESTDIR)$(PREFIX)/include/

# Tarball di release: CLI + server + libreria statica + header + licenza/readme +
# script di download modelli. Nome = versione git + os + arch; strip dei binari e
# checksum SHA-256. Tutto in dist/ (gitignored). Uso: make dist
DIST_OS   := $(shell uname -s | tr '[:upper:]' '[:lower:]')
DIST_ARCH := $(shell uname -m)
DIST_NAME := mynah-$(MYNAH_BUILD)-$(DIST_OS)-$(DIST_ARCH)
DIST_DIR  := dist/$(DIST_NAME)
dist: mynah mynah-server libmynah.a
	@rm -rf $(DIST_DIR)
	@mkdir -p $(DIST_DIR)/bin $(DIST_DIR)/lib $(DIST_DIR)/include $(DIST_DIR)/scripts
	install -m 755 mynah mynah-server $(DIST_DIR)/bin/
	install -m 644 libmynah.a $(DIST_DIR)/lib/
	install -m 644 src/mynah.h $(DIST_DIR)/include/
	install -m 644 LICENSE README.md $(DIST_DIR)/
	install -m 755 scripts/download_model.sh $(DIST_DIR)/scripts/
	@strip $(DIST_DIR)/bin/mynah $(DIST_DIR)/bin/mynah-server 2>/dev/null || true
	cd dist && tar czf $(DIST_NAME).tar.gz $(DIST_NAME)
	@rm -rf $(DIST_DIR)
	@echo "" && echo "-> dist/$(DIST_NAME).tar.gz"
	@cd dist && shasum -a 256 $(DIST_NAME).tar.gz 2>/dev/null || (cd dist && sha256sum $(DIST_NAME).tar.gz)

.PHONY: all clean install dist test golden-dump lib shared example debug ubsan asan bench leaks test-nemo-langs fetch-lang-samples test-server test-samples cuda
