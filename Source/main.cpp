
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

//static const int WORK_SIZE = 256;

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 *
 * See cuda.h for error code descriptions.
 */
#define CHECK_CUDA_RESULT(N) {											\
	CUresult result = N;												\
	if (result != 0) {													\
		printf("CUDA call on line %d returned error %d\n", __LINE__,	\
			result);													\
		exit(1);														\
	} }

int main(int argc, char **argv)
{
	size_t n = 1024;
	curandGenerator_t rngGen;
	float* d_Data;
	float* h_Data;

	//Allocate data on the host
	h_Data = (float*)calloc(n, sizeof(float));

	//Allocate data on the device
	cudaMalloc(&d_Data, sizeof(float) * n);

	//Create RN generator
	//Might have to change this to something else if it isn't good enough
	curandCreateGenerator(&rngGen, CURAND_RNG_PSEUDO_DEFAULT);

	//Set the RNG seed
	curandSetPseudoRandomGeneratorSeed(rngGen, 1);

	//Generate random numbers on the device
	//Generate random numbers using a normal distribution
	//Normal distribution should emulate white noise hopefully?
	curandGenerateNormal(rngGen, d_Data, n, 0.0f, 1.0f);

	//Copy to the host
	cudaMemcpy(h_Data, d_Data, n, cudaMemcpyDeviceToHost);


	//Destroy the RNG
	curandDestroyGenerator(rngGen);

	//Free memory
	cudaFree(d_Data);
	free(h_Data);


	return 0;
}
