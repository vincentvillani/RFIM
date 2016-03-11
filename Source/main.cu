
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand.h>

#include <stdint.h>

#include "../Header/Kernels.h"
#include "../Header/UnitTests.h"
#include "../Header/CudaMacros.h"


//TODO: Test the mean kernel on known data
//TODO: Have the signal number be a parameter, store it all in one big array


int main(int argc, char **argv)
{
	//Run all the unit tests
	RunAllUnitTests();

/*
	uint64_t sampleElements = 26; //The number of elements per sample (i.e. b1p1, b1p2, b2p1,b2p2) where b = beam, p = polarisation
	uint64_t sampleNumber = 1024; //Numbers of samples being used to compute the covariance matrix in this iteration

	curandGenerator_t rngGen;






	//Allocate data on the device
	cudaMalloc(&d_signal1, sizeof(float) * n);
	cudaMalloc(&d_signal2, sizeof(float) * n);
	cudaMalloc(&d_tempWorkingSpace, sizeof(float) * n);


	cudaMalloc(&d_covarianceVector, sizeof(float) * 2);
	cudaMalloc(&d_meanVector, sizeof(float) * 2);
	cudaMalloc(&d_covarianceMatrix, sizeof(float) * 2);


	//Create RNG
	//Might have to change this to something else if it isn't good enough
	if( curandCreateGenerator(&rngGen, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS)
	{
		printf("Error at %s:%d\n",__FILE__,__LINE__);
		exit(1);
	}

	//Set the RNG seed
	if((curandSetPseudoRandomGeneratorSeed(rngGen, 1)) != CURAND_STATUS_SUCCESS)
	{
		printf("Error at %s:%d\n",__FILE__,__LINE__);
		exit(1);
	}

	//Generate random numbers on the device
	//Generate random numbers using a normal distribution
	//Normal distribution should emulate white noise hopefully?
	//Generate signal 1
	if(curandGenerateNormal(rngGen, d_signal1, n, 0.0f, 1.0f) != CURAND_STATUS_SUCCESS)
	{
		printf("Error at %s:%d\n",__FILE__,__LINE__);
		exit(1);
	}

	//Generate signal 2
	if(curandGenerateNormal(rngGen, d_signal2, n, 0.0f, 1.0f) != CURAND_STATUS_SUCCESS)
	{
		printf("Error at %s:%d\n",__FILE__,__LINE__);
		exit(1);
	}


	//Calculate the mean of each signal, should be around zero based on the RNG, but we will need it when we do this for real anyway
	dim3 grid(2); //Number of blocks in the grid
	dim3 block(256); //Number of threads per block


	//Copy the signal into the temp working space before we start using it, this algorithm computes the mean in place
	cudaMemcpy(d_tempWorkingSpace, d_signal1, sizeof(float) * n, cudaMemcpyDeviceToDevice);
	parallelMeanUnroll2 <<<grid.x, block.x>>> (d_tempWorkingSpace, n, d_meanVector);
	CudaCheckError();

	cudaMemcpy(d_tempWorkingSpace, d_signal2, sizeof(float) * n, cudaMemcpyDeviceToDevice);
	parallelMeanUnroll2 <<< grid.x, block.x >>> (d_tempWorkingSpace, n, d_meanVector + 1);
	CudaCheckError();

	//Subtract the mean from each of the signals
	subtractMean <<<4, 256>>> (d_signal1, n, *d_meanVector);
	subtractMean <<<4, 256>>> (d_signal2, n, *d_meanVector + 1);
	CudaCheckError();

	//Calculate the covariance matrix
	//outerProductSmartBruteForceLessThreads <<<1, 2>>> (d_covarianceMatrix, d_covarianceVector, 2);
	CudaCheckError();

	//Copy the results back over
	//


	//Destroy the RNG
	if(curandDestroyGenerator(rngGen) != CURAND_STATUS_SUCCESS)
	{
		printf("Error at %s:%d\n",__FILE__,__LINE__);
		exit(1);
	}

	//Free memory
	cudaFree(d_signal1);
	cudaFree(d_signal2);
	cudaFree(d_tempWorkingSpace);

	cudaFree(d_covarianceVector);
	cudaFree(d_meanVector);
	cudaFree(d_covarianceMatrix);

*/
	return 0;
}
