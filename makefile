CC_FLAGS=-DMKL_ILP64 -m64
LIBS=-lcublas -lcusolver -lcurand -lmkl_intel_ilp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl
DEBUG_FLAGS=-g -G -O0 -std=c++11
RELEASE_FLAGS=-O3 -std=c++11
FILE_LIST=Source/main.cu Source/Kernels.cu Source/UnitTests.cu Source/RFIM.cu Source/RFIMHelperFunctions.cu Source/UtilityFunctions.cu Source/RFIMMemoryStruct.cu Source/RFIMMemoryStructBatched.cu Source/RFIMMemoryStructComplex.cu Source/RFIMMemoryStructCPU.cpp Source/Benchmark.cu Source/CudaUtilityFunctions.cu Source/FilterbankMultiplexedSignal.cpp

debug:
	nvcc $(CC_FLAGS) -o debug.out $(FILE_LIST) $(LIBS) $(DEBUG_FLAGS)

debugMulti:
	nvcc $(CC_FLAGS) -o debugMultiStream.out $(FILE_LIST) $(LIBS) $(DEBUG_FLAGS)

release:
	nvcc $(CC_FLAGS) -o release.out $(FILE_LIST) $(LIBS) $(RELEASE_FLAGS)


clean:
	rm debug.out release.out
