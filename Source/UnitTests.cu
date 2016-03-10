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
void ParallelMeanUnitTest();

void CovarianceMatrixUnitTest();

//-------------------------------------



//outerProductSmartBruteForceLessThreads(float* resultMatrix, float* vec, int vectorLength);


void CovarianceMatrixUnitTest()
{
	float* d_vec;
	float* d_resultMatrix;

	float* h_vec;
	float* h_resultMatrix;

	uint64_t n = 2;
	uint64_t resultMatrixLength = upperTriangularLength(n);

	dim3 block = dim3(2, 2);
	dim3 grid = dim3(1, 1);


	//Allocate memory
	cudaMalloc(&d_vec, sizeof(float) * n);
	cudaMalloc(&d_resultMatrix, sizeof(float) * resultMatrixLength);
	CudaCheckError();

	h_vec = (float*)calloc(n, sizeof(float));
	h_resultMatrix = (float*)calloc(resultMatrixLength, sizeof(float));


	//Generate the vector/signal
	for(uint64_t i = 0; i < n; ++i)
	{
		h_vec[i] = 1 + i;
	}

	cudaMemcpy(d_vec, h_vec, n * sizeof(float), cudaMemcpyHostToDevice);
	CudaCheckError();

	/*
	printf("Launching kernel with parameters\nGrid(%d, %d), Block(%d, %d)\n",
			grid.x, grid.y, block.x, block.y);
	*/

	//Run the kernel
	outerProductSmartBruteForceLessThreads <<<grid, block>>> (d_resultMatrix, d_vec, n);
	CudaCheckError();

	//check the results
	cudaMemcpy(h_resultMatrix, d_resultMatrix, resultMatrixLength * sizeof(float), cudaMemcpyDeviceToHost);
	CudaCheckError();

	/*
	for(uint64_t i = 0; i < resultMatrixLength; ++i)
	{
		printf("%llu: %f\n", i, h_resultMatrix[i]);
	}
	*/

	bool failed = false;

	if(h_resultMatrix[0] - 1.0f > 0.000001)
		failed = true;
	else if(h_resultMatrix[1] - 2.0f > 0.000001)
		failed = true;
	else if(h_resultMatrix[2] - 4.0f > 0.000001)
		failed = true;

	if(failed)
	{
		fprintf(stderr, "CovarianceMatrixUnitTest() failed.\nExpected: %f, %f, %f\nActual: %f, %f, %f\n", 1.0f, 2.0f, 4.0f,
				h_resultMatrix[0], h_resultMatrix[1], h_resultMatrix[2]);
		exit(1);
	}

	cudaFree(d_vec);
	cudaFree(d_resultMatrix);
	CudaCheckError();

	free(h_vec);
	free(h_resultMatrix);
}



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

	//Free all memory
	free(h_knownSignal);
	free(h_mean);

	cudaFree(d_knownSignal);
	cudaFree(d_mean);

}



void RunAllUnitTests()
{
	ParallelMeanUnitTest();
	CovarianceMatrixUnitTest();

	printf("All tests passed!\n");
}

