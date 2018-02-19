CXX=nvcc
CXXFLAGS=-O3 -g -DNDEBUG -std=c++11 -lineinfo --use_fast_math -gencode=arch=compute_61,code=sm_61 \
								     -gencode=arch=compute_50,code=sm_50 \
								     -gencode=arch=compute_35,code=sm_35 \
								     -gencode=arch=compute_30,code=sm_30
BIN=crs-cuda
OUT_DIR=bin

all:
	$(CXX) $(CXXFLAGS) src/*.cpp src/*.cu -I./include/ -lcuda -lm -o ${OUT_DIR}/${BIN}
