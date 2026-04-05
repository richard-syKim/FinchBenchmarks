CC       = gcc
CXX      = g++
CXXFLAGS += -std=c++17 -O3 -fPIC -fopenmp

# Auto-detect core count for parallel TACO build
ifeq ("$(shell uname)","Darwin")
    export NPROC_VAL := $(shell sysctl -n hw.logicalcpu_max)
else
    export NPROC_VAL := $(shell lscpu -p | egrep -v '^\#' | wc -l)
endif

TACO_DIR      = ../../deps/taco
TACO_CLONE    = $(TACO_DIR)/.git
TACO_LIB      = $(TACO_DIR)/build/lib/libtaco.so
TACO_CXXFLAGS = $(CXXFLAGS) -I$(TACO_DIR)/include -I$(TACO_DIR)/src
TACO_LDLIBS   = -L$(TACO_DIR)/build/lib -ltaco -ldl \
                -Wl,-rpath,$(abspath $(TACO_DIR)/build/lib)

EIGEN_DIR      = ../../deps/eigen
EIGEN_CLONE    = $(EIGEN_DIR)/.git
EIGEN_CXXFLAGS = $(CXXFLAGS) -I$(EIGEN_DIR)


all: libtaco_kernel.so libeigen_kernel.so libmkl_kernel.so


# Clone TACO submodule if missing
$(TACO_CLONE):
	git submodule update --init $(TACO_DIR)

$(TACO_LIB): $(TACO_CLONE)
	cd $(TACO_DIR) && \
	mkdir -p build && \
	cd build && \
	cmake -DPYTHON=false -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DOPENMP=ON .. && \
	make taco -j$(NPROC_VAL)

libtaco_kernel.so: taco_impl.cpp $(TACO_LIB)
	$(CXX) $(TACO_CXXFLAGS) -shared -o $@ $< $(TACO_LDLIBS)


# Clone Eigen submodule if missing
$(EIGEN_CLONE):
	git submodule update --init $(EIGEN_DIR)

# Eigen is header-only
libeigen_kernel.so: eigen_impl.cpp | $(EIGEN_CLONE)
	$(CXX) $(EIGEN_CXXFLAGS) -shared -o $@ $<


MKL_ROOT    ?= /opt/intel/oneapi/mkl/latest
MKL_INCLUDE  = $(MKL_ROOT)/include
MKL_LIB      = $(MKL_ROOT)/lib/intel64

MKL_FLAGS   = -I$(MKL_INCLUDE) \
              -L$(MKL_LIB) \
              -lmkl_rt \
              -Wl,-rpath,$(MKL_LIB)

libmkl_kernel.so: mkl_impl.cpp
	$(CXX) $(CXXFLAGS) -shared $< -o $@ $(TACO_FLAGS) $(MKL_FLAGS)


clean:
	rm -f libtaco_kernel.so libeigen_kernel.so libmkl_kernel.so

clean-all: clean
	rm -rf $(TACO_DIR)/build