
all:
	nvcc -o a.out Source/main.cu Source/Kernels.cu Source/UnitTests.cu -lcurand -arch=sm_20

clean:
	rm a.out *.o
