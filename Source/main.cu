
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


//TODO: Test the mean kernel on known data



int main(int argc, char **argv)
{
	//Run all the unit tests
	RunAllUnitTests();



	uint64_t n = 1024;
	curandGenerator_t rngGen;
	float* d_Signal1;
	float* d_Signal2;

	float* d_Signal1Mean;
	float* d_Signal2Mean;



	//Allocate data on the device
	cudaMalloc(&d_Signal1, sizeof(float) * n);
	cudaMalloc(&d_Signal2, sizeof(float) * n);

	cudaMalloc(&d_Signal1Mean, sizeof(float));
	cudaMalloc(&d_Signal2Mean, sizeof(float));

	//Create RN generator
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
	if(curandGenerateNormal(rngGen, d_Signal1, n, 0.0f, 1.0f) != CURAND_STATUS_SUCCESS)
	{
		printf("Error at %s:%d\n",__FILE__,__LINE__);
		exit(1);
	}

	//Generate signal 2
	if(curandGenerateNormal(rngGen, d_Signal2, n, 0.0f, 1.0f) != CURAND_STATUS_SUCCESS)
	{
		printf("Error at %s:%d\n",__FILE__,__LINE__);
		exit(1);
	}


	//Calculate the mean of each signal, should be around zero based on the RNG, but we will need it when we do this for real anyway
	dim3 grid(2); //Number of blocks in the grid
	dim3 block(256); //Number of threads per block

	//parallelMeanUnroll2 <<<grid.x, block.x>>> (d_Signal1, n, d_Signal1Mean);
	//parallelMeanUnroll2 <<< grid.x, block.x >>>(d_Signal2, n, d_Signal2Mean);




	//Destroy the RNG
	if(curandDestroyGenerator(rngGen) != CURAND_STATUS_SUCCESS)
	{
		printf("Error at %s:%d\n",__FILE__,__LINE__);
		exit(1);
	}

	//Free memory
	cudaFree(d_Signal1);
	cudaFree(d_Signal2);
	cudaFree(d_Signal1Mean);
	cudaFree(d_Signal2Mean);


	return 0;
}
