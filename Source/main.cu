
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
#include <curand.h>

#include <string>
#include <stdint.h>

#include "../Header/UnitTests.h"
#include "../Header/RFIMHelperFunctions.h"
#include "../Header/UtilityFunctions.h"
#include "../Header/RFIM.h"
#include "../Header/Benchmark.h"


//TODO: Make sure memory that can be used again, is still in a valid state after the first execution

int main(int argc, char **argv)
{
	//Run all the unit tests
	//RunAllUnitTests();


	uint32_t h_valuesPerSample = 26;
	uint32_t h_numberOfSamples = 1024;
	uint32_t h_batchSize = 512;


	RFIMMemoryStruct* RFIMStruct = RFIMMemoryStructCreate(h_valuesPerSample, h_numberOfSamples, 2, h_batchSize);

	/*
	//1. Generate a signal on the device
	//----------------------------------
	//Start cuda rand library
	curandGenerator_t rngGen;

	if( curandCreateGenerator(&rngGen, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS)
	{
		fprintf(stderr, "main: Unable to start cuRand library\n");
		//exit(1);
	}

	//Set the RNG seed
	if((curandSetPseudoRandomGeneratorSeed(rngGen, 1)) != CURAND_STATUS_SUCCESS)
	{
		fprintf(stderr, "main: Unable to set the RNG Seed value\n");
		//exit(1);
	}

	//Allocate memory for the pointers to the signal matrices
	float** d_signalMatrices;
	cudaMalloc(&d_signalMatrices, sizeof(float*) * h_batchSize);
	float* d_whiteNoiseSignalMatrix = Device_GenerateWhiteNoiseSignal(&rngGen, h_valuesPerSample, h_numberOfSamples);

	//Set each pointer to point to the same signal matrix for now
	for(uint32_t i = 0; i < h_batchSize; ++i)
	{
		d_signalMatrices[i] = d_whiteNoiseSignalMatrix;
	}



	//2. Create a RFIM Struct
	//--------------------------
	RFIMMemoryStruct* RFIMStruct = RFIMMemoryStructCreate(h_valuesPerSample, h_numberOfSamples, 2, h_batchSize);

	//Create space to store the filtered signal
	float** d_filteredSignals;
	cudaMalloc(&d_filteredSignals, sizeof(float*) * h_batchSize); //Allocate space for the pointers
	uint32_t filteredSignalByteSize = sizeof(float) * h_valuesPerSample * h_numberOfSamples;

	for(uint32_t i = 0; i < h_batchSize; ++i)
	{
		cudaMalloc(&(d_filteredSignals[i]), filteredSignalByteSize);
	}



	//3. Run RFIM benchmark
	//--------------------------
	//Benchmark(RFIMStruct, d_whiteNoiseSignalMatrix, d_filteredSignal, 200, 1);

	//RFIMRoutine(RFIMStruct, d_whiteNoiseSignalMatrix, d_filteredSignal);




	//4. Free everything
	//--------------------------
	//Free the RFIM Struct
	for(uint32_t i = 0; i < h_batchSize; ++i)
	{
		cudaFree(d_filteredSignals[i]);
	}

	RFIMMemoryStructDestroy(RFIMStruct);
	cudaFree(d_signalMatrices);
	cudaFree(d_whiteNoiseSignalMatrix);
	cudaFree(d_filteredSignals);
	*/


	return 0;
}
