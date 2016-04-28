
debug:
	nvcc -o debug.out Source/main.cu Source/Kernels.cu Source/UnitTests.cu Source/RFIM.cu Source/RFIMHelperFunctions.cu Source/UtilityFunctions.cu Source/RFIMMemoryStruct.cu Source/RFIMMemoryStructBatched.cu Source/RFIMMemoryStructComplex.cu Source/RFIMMemoryStructCPU.cpp Source/Benchmark.cu Source/CudaUtilityFunctions.cu -lcurand -lcublas -lcusolver -g -G -O0 -std=c++11

debugMulti:
	nvcc -o debugMultiStream.out Source/main.cu Source/Kernels.cu Source/UnitTests.cu Source/RFIM.cu Source/RFIMHelperFunctions.cu Source/UtilityFunctions.cu Source/RFIMMemoryStruct.cu Source/RFIMMemoryStructBatched.cu Source/RFIMMemoryStructComplex.cu Source/RFIMMemoryStructCPU.cpp Source/Benchmark.cu Source/CudaUtilityFunctions.cu -lcurand -lcublas -lcusolver -g -G -O0 -std=c++11

release:
	nvcc -o release.out Source/main.cu Source/Kernels.cu Source/UnitTests.cu Source/RFIM.cu Source/RFIMHelperFunctions.cu Source/UtilityFunctions.cu Source/RFIMMemoryStruct.cu Source/RFIMMemoryStructBatched.cu Source/RFIMMemoryStructComplex.cu Source/RFIMMemoryStructCPU.cpp Source/Benchmark.cu Source/CudaUtilityFunctions.cu -lcurand -lcublas -lcusolver -O3 -std=c++11


clean:
	rm debug.out release.out
