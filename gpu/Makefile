# gpu/Makefile — the CUDA batched streaming server, a SEPARATE tree.
#
#   make -C gpu               mynah-asr-server-cuda (needs nvcc)
#   make -C gpu cpu           the same server with the cpu reference engine
#                             only (no nvcc; a dev box or the CI runner)
#   make -C gpu test-kernels  tests/test_cuda_kernels (GPU) -- compile here,
#                             run on a device
#   make -C gpu test-stream   tests/test_cuda_stream (GPU + pack)
#
# The library (../libmynah_asr.a, `make lib` at the top) and the two reusable
# server modules (../server/http_util.c, ../server/stream_out.c) are used
# read-only and compiled into build/gpu/ with THIS tree's flags. Nothing under
# ../src or ../server is modified by this tree (CLAUDE.md, S14).
#
# CUDA_ARCH: an explicit list for a portable artefact (default covers Ampere,
# Ada and Hopper; add sm_120 with CUDA >= 12.8 for Blackwell). `native` is for
# local bring-up only and never for CI or a release.

TOP      := ..
CC       ?= cc
NVCC     ?= nvcc
CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= sm_80 sm_86 sm_89 sm_90
GENCODE  := $(foreach a,$(CUDA_ARCH),-gencode arch=compute_$(subst sm_,,$(a)),code=$(a))

UNAME_S := $(shell uname -s)
CFLAGS  ?= -std=c11 -O2 -Wall -Wextra -D_DEFAULT_SOURCE
CFLAGS  += -iquote $(TOP)/src -iquote $(TOP)/server -iquote $(TOP)/vendor -iquote . \
           -I$(TOP)/third_party/ingot/include -DMYNAH_ASR_BUILD='"$(shell git -C $(TOP) describe --always --dirty 2>/dev/null || echo dev)"'
NVCCFLAGS ?= -O3 -std=c++14 -Xcompiler -fPIC $(GENCODE) --expt-relaxed-constexpr
NVCCFLAGS += -I$(TOP)/src -I$(TOP)/server -I$(TOP)/vendor -I. -Icuda -I$(TOP)/third_party/ingot/include
LDLIBS   := $(TOP)/libmynah_asr.a $(TOP)/third_party/ingot/libingot.a -lpthread -lm
ifeq ($(UNAME_S),Darwin)
  LDLIBS += -framework Accelerate -framework Metal -framework MetalPerformanceShaders -framework Foundation
endif
CUDA_LIBS := -L$(CUDA_HOME)/lib64 -lcudart -lcublas

B := $(TOP)/build/gpu

C_OBJ    := $(B)/pack.o $(B)/engine_cpu.o $(B)/server_main.o $(B)/ws.o $(B)/http_util.o $(B)/stream_out.o
CU_OBJ   := $(B)/gemm.o $(B)/kernels.o $(B)/engine.o

.PHONY: all cpu clean test-kernels test-stream lib
all: $(TOP)/mynah-asr-server-cuda
cpu: $(TOP)/mynah-asr-server-cuda-cpuonly

lib:
	$(MAKE) -C $(TOP) lib

$(TOP)/libmynah_asr.a:
	$(MAKE) -C $(TOP) lib

$(B)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@
$(B)/server_main.o: server/main.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@
$(B)/ws.o: server/ws.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@
$(B)/http_util.o: $(TOP)/server/http_util.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@
$(B)/stream_out.o: $(TOP)/server/stream_out.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@
$(B)/%.o: cuda/%.cu cuda/kernels.cuh asr_engine.h pack.h
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# the product: both engines linked, cuda the default
$(TOP)/mynah-asr-server-cuda: $(C_OBJ) $(CU_OBJ) $(TOP)/libmynah_asr.a
	$(NVCC) $(GENCODE) -o $@ $(C_OBJ) $(CU_OBJ) $(LDLIBS) $(CUDA_LIBS)

# no nvcc: the server with the cpu reference engine and a cuda engine that
# says "not compiled" -- for protocol tests on a dev box and for CI
$(B)/engine_stub.o: cuda/engine_stub.c asr_engine.h
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@
$(TOP)/mynah-asr-server-cuda-cpuonly: $(C_OBJ) $(B)/engine_stub.o $(TOP)/libmynah_asr.a
	$(CC) $(CFLAGS) -o $@ $(C_OBJ) $(B)/engine_stub.o $(LDLIBS)

# tests
$(B)/test_cuda_kernels.o: $(TOP)/tests/test_cuda_kernels.cu cuda/kernels.cuh
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
test-kernels: $(TOP)/tests/test_cuda_kernels
$(TOP)/tests/test_cuda_kernels: $(B)/test_cuda_kernels.o $(B)/gemm.o $(B)/kernels.o
	$(NVCC) $(GENCODE) -o $@ $^ $(CUDA_LIBS)

$(B)/test_cuda_stream.o: $(TOP)/tests/test_cuda_stream.c asr_engine.h
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@
test-stream: $(TOP)/tests/test_cuda_stream
$(TOP)/tests/test_cuda_stream: $(B)/test_cuda_stream.o $(B)/pack.o $(B)/engine_cpu.o $(CU_OBJ) $(TOP)/libmynah_asr.a
	$(NVCC) $(GENCODE) -o $@ $(B)/test_cuda_stream.o $(B)/pack.o $(B)/engine_cpu.o $(CU_OBJ) $(LDLIBS) $(CUDA_LIBS)

clean:
	rm -rf $(B) $(TOP)/mynah-asr-server-cuda $(TOP)/mynah-asr-server-cuda-cpuonly \
	       $(TOP)/tests/test_cuda_kernels $(TOP)/tests/test_cuda_stream
