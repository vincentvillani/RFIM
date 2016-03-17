/*
 * RFIMHelperFunctions.cu
 *
 *  Created on: 17/03/2016
 *      Author: vincentvillani
 */

#include "../Header/RFIMHelperFunctions.h"

#include <stdio.h>

#include <cuda.h>
#include <curand.h>
#include <cublas.h>

float* Device_GenerateWhiteNoiseSignal(uint64_t h_valuesPerSample, uint64_t h_numberOfSamples)
{
	uint64_t totalSignalLength = h_valuesPerSample * h_numberOfSamples;
	uint64_t totalSignalByteSize = totalSignalLength * sizeof(float);

	float* d_signal;

	cudaError_t error;

	//Allocate the memory required to store the signal
	error =  cudaMalloc(&d_signal, totalSignalByteSize);

	//Check that it was allocated successfully
	if(error != cudaSuccess)
	{
		fprintf(stderr, "Device_GenerateWhiteNoiseSignal: Unable to allocate %llu bytes of memory on the device\n", totalSignalByteSize);
		exit(1);
	}

	//Start cuda rand library
	curandGenerator_t rngGen;

	if( curandCreateGenerator(&rngGen, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS)
	{
		fprintf(stderr, "Device_GenerateWhiteNoiseSignal: Unable to start cuRand library\n");
		exit(1);
	}


	//Set the RNG seed
	if((curandSetPseudoRandomGeneratorSeed(rngGen, 1)) != CURAND_STATUS_SUCCESS)
	{
		fprintf(stderr, "Device_GenerateWhiteNoiseSignal: Unable to set the RNG Seed value\n");
		exit(1);
	}


	//Generate the signal!
	//Generate random numbers on the device
	//Generate random numbers using a normal distribution
	//Normal distribution should emulate white noise hopefully?
	//Generate signal
	if(curandGenerateNormal(rngGen, d_signal, totalSignalLength, 0.0f, 1.0f) != CURAND_STATUS_SUCCESS)
	{
		fprintf(stderr, "Device_GenerateWhiteNoiseSignal: Error when generating the signal\n");
		exit(1);
	}



	//Destroy the RNG
	if(curandDestroyGenerator(rngGen) != CURAND_STATUS_SUCCESS)
	{
		fprintf(stderr, "Device_GenerateWhiteNoiseSignal: Error in destroying the RNG generator \n");
		exit(1);
	}


	//Return the generated signal that resides in DEVICE memory
	return d_signal;

}

