
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
	c result = N;												\
	if (result != 0) {													\
		printf("CUDA call on line %d returned error %d\n", __LINE__,	\
			result);													\
		exit(1);														\
	} }


inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }



//#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \ printf("Error at %s:%d\n",__FILE__,__LINE__);\return EXIT_FAILURE;}} while(0)




int main(int argc, char **argv)
{
	size_t n = 1024;
	curandGenerator_t rngGen;
	float* d_Data;
	float* h_Data;

	//Allocate data on the host
	h_Data = (float*)calloc(n, sizeof(float));

	//Allocate data on the device
	gpuErrchk(cudaMalloc(&d_Data, sizeof(float) * n));

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
	if(curandGenerateNormal(rngGen, d_Data, n, 0.0f, 1.0f) != CURAND_STATUS_SUCCESS)
	{
		printf("Error at %s:%d\n",__FILE__,__LINE__);
		exit(1);
	}

	//Copy to the host
	gpuErrchk(cudaMemcpy(h_Data, d_Data, n * sizeof(float), cudaMemcpyDeviceToHost));

	for(int i = 0; i < n; ++i)
	{
		printf("%f\n", h_Data[i]);
	}

	//Destroy the RNG
	if(curandDestroyGenerator(rngGen) != CURAND_STATUS_SUCCESS)
	{
		printf("Error at %s:%d\n",__FILE__,__LINE__);
		exit(1);
	}

	//Free memory
	cudaFree(d_Data);
	free(h_Data);


	return 0;
}
