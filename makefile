
all:
	nvcc -o debug.out Source/main.cu Source/Kernels.cu Source/UnitTests.cu Source/RFIM.cu Source/RFIMHelperFunctions.cu Source/CudaUtilityFunctions.cu Source/UtilityFunctions.cu Source/RFIMMemoryStruct.cu Source/Benchmark.cu -lcurand -lcublas -lcusolver -arch=sm_20 -g -G -O0

optimised:
	nvcc -o release.out Source/main.cu Source/Kernels.cu Source/UnitTests.cu Source/RFIM.cu Source/RFIMHelperFunctions.cu Source/CudaUtilityFunctions.cu Source/UtilityFunctions.cu Source/RFIMMemoryStruct.cu Source/Benchmark.cu -lcurand -lcublas -lcusolver -arch=sm_20 -O3


clean:
	rm debug.out release.out
