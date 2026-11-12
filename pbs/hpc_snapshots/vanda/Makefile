# Makefile for the K-means project (serial / MPI / MPI+CUDA).
#
# Targets:
#   make serial     -> bin/kmeans_serial
#   make mpi        -> bin/kmeans_mpi
#   make cuda       -> bin/kmeans_mpi_cuda   (needs nvcc + MPI)
#   make all        -> everything available
#   make clean
#
# On Atlas/Vanda you will typically `module load` gcc/openmpi/cuda first; the
# wrappers (mpicxx) and nvcc then resolve automatically. Override CUDA_ARCH to
# match the target GPU, e.g. `make cuda CUDA_ARCH=sm_80`.

CXX      ?= g++
MPICXX   ?= mpicxx
NVCC     ?= nvcc

CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -march=native
# CUDA architecture. sm_70=V100, sm_80=A100, sm_86=RTX30xx, sm_89=RTX40xx,
# sm_90=H100. Adjust for Atlas/Vanda's GPUs.
CUDA_ARCH ?= sm_75
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(CUDA_ARCH)

# MPI include/library paths for nvcc. We deliberately use OpenMPI's
# --showme:incdirs / --showme:libdirs (which emit bare paths) rather than
# --showme:compile / --showme:link, because the latter include host-linker
# flags like `-Wl,-rpath,...` that nvcc cannot parse. From bare paths we build
# plain -I / -L / -lmpi flags, which nvcc understands natively and portably
# (works with both the local and Vanda OpenMPI builds).
MPI_INCDIRS := $(shell $(MPICXX) --showme:incdirs 2>/dev/null)
MPI_LIBDIRS := $(shell $(MPICXX) --showme:libdirs 2>/dev/null)
MPI_INC  := $(foreach d,$(MPI_INCDIRS),-I$(d))
MPI_LINK_NVCC := $(foreach d,$(MPI_LIBDIRS),-L$(d)) -lmpi

SRC     := src
SCRIPTS := scripts
BIN     := bin

.PHONY: all serial mpi cuda gen clean dirs
all: serial mpi cuda gen

dirs:
	@mkdir -p $(BIN)

serial: dirs
	$(CXX) $(CXXFLAGS) $(SRC)/kmeans_serial.cpp -o $(BIN)/kmeans_serial

# Fast synthetic-data generator (compiled helper; source lives in scripts/).
gen: dirs
	$(CXX) $(CXXFLAGS) $(SCRIPTS)/gen_blobs.cpp -o $(BIN)/gen_blobs

mpi: dirs
	$(MPICXX) $(CXXFLAGS) $(SRC)/kmeans_mpi.cpp -o $(BIN)/kmeans_mpi

cuda: dirs
	$(NVCC) $(NVCCFLAGS) $(MPI_INC) $(SRC)/kmeans_mpi_cuda.cu \
		$(MPI_LINK_NVCC) -o $(BIN)/kmeans_mpi_cuda

clean:
	rm -rf $(BIN)
