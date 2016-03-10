/*
 * UnitTests.cu
 *
 *  Created on: 10/03/2016
 *      Author: vincentvillani
 */


#include "../Header/UnitTests.h"

#include "../Header/CudaMacros.h"
#include "../Header/Kernels.h"

#include <assert.h>

//Private dec's
void ParallelMeanUnitTest()
{
	float* d_knownSignal;
	float* d_mean;

	float* h_knownSignal;
	float* h_mean;

	float expectedMean = 0;


	uint64_t n = 8;

	//Allocate memory
	cudaMalloc(&d_knownSignal, sizeof(float) * n);
	cudaMalloc(&d_mean, sizeof(float));
	CudaCheckError();

	h_knownSignal = (float*)calloc(n, sizeof(float));
	h_mean = (float*)calloc(1, sizeof(float));

	//Create a signal
	for(uint32_t i = 0; i < n; ++i)
	{
		h_knownSignal[i] = i;

		expectedMean += i;
	}

	expectedMean /= (n - 1);

	cudaMemcpy(d_knownSignal, h_knownSignal, sizeof(float) * n, cudaMemcpyHostToDevice);
	CudaCheckError();

	//Run the kernel
	parallelMeanUnroll2 <<<2, 2>>> (d_knownSignal, n, d_mean);
	CudaCheckError();

	//copy the result back to the host
	cudaMemcpy(h_mean, d_mean, sizeof(float), cudaMemcpyDeviceToHost);
	CudaCheckError();

	if(*h_mean - expectedMean > 0.000001)
	{
		fprintf(stderr, "ParallelMeanUnitTest() failed. Expected: %f, Actual: %f\n", expectedMean, *h_mean);
		exit(1);
	}


}



void RunAllUnitTests()
{
	ParallelMeanUnitTest();

	printf("All tests passed!\n");
}

