#    Copyright 2014-2015 Dake Feng, Peri LLC, dakefeng@gmail.com
#    This file is part of TomograPeri.
#    TomograPeri is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    TomograPeri is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    You should have received a copy of the GNU Lesser General Public License
#    along with TomograPeri.  If not, see <http://www.gnu.org/licenses/>.

CUDADIR ?= /usr/local/cuda

OUT ?= .

CC = gcc

NVCC = $(CUDADIR)/bin/nvcc

LIBS = -L$(CUDADIR)/lib64 -lm -lcuda -lcudart
INCLUDES = -I$(CUDADIR)/include

CCFLAGS = -fPIC -Wall -fno-strict-aliasing
LNKFLAGS = -ldl -shared

CUOBJS = utils_cuda.o pml_quad_cuda.o ospml_quad_cuda.o pml_hybrid_cuda.o ospml_hybrid_cuda.o pml_weight_cuda.o 
CUSRCS = utils_cuda.cu pml_quad_cuda.cu ospml_quad_cuda.cu pml_hybrid_cuda.cu ospml_hybrid_cuda.cu pml_weight_cuda.cu
CCOBJS     = recon.o 
CCSRCS     = recon.cpp
TESTSRCS   = ../main.cpp ../cnpy/cnpy.cpp
TESTOBJS   = main.o cnpy.o

TARGET   = libtomoperi_gpu.so
TEST    = tomoperi_gpu_test.exe

$(TARGET):	$(CCOBJS) $(CUOBJS)
	mkdir -p $(OUT)
	$(NVCC) -o $(OUT)/$(TARGET) $(CUOBJS) $(CCOBJS) $(LIBS) $(LNKFLAGS) 

$(CCOBJS):	$(CCSRCS)
	$(CC) -c $(CCFLAGS) $(CCSRCS) $(INCLUDES)

$(CUOBJS):	$(CUSRCS)
	$(NVCC) -c -arch=sm_20 $(CUSRCS) --compiler-options '-fPIC'

$(TEST):	$(TARGET) $(TESTSRCS)
	$(CC) -o $(OUT)/$(TEST) $(TESTSRCS)  -ldl -Wl,-rpath '-Wl,$$ORIGIN' -L$(OUT) -ltomoperi_gpu -lstdc++

all:	$(TARGET)

test:	$(TEST)

clean:
	rm -f $(CCOBJS) $(CUOBJS) $(TESTOBJS) $(TARGET) $(TEST)



