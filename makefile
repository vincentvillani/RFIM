
all:
	nvcc -o a.out Source/main.cpp -lcurand -arch=sm_20

clean:
	rm a.out *.o
