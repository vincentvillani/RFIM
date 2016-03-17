
all:
	nvcc -o a.out Source/main.cu Source/Kernels.cu Source/UnitTests.cu Source/RFIM.cu Source/RFIMHelperFunctions.cu -lcurand -lcublas -arch=sm_20

clean:
	rm a.out *.o
