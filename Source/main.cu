
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

//#include "../Header/Kernels.h"
#include "../Header/UnitTests.h"
//#include "../Header/CudaMacros.h"
#include "../Header/RFIMHelperFunctions.h"
//#include "../Header/CudaUtilityFunctions.h"
#include "../Header/UtilityFunctions.h"
#include "../Header/RFIM.h"
#include "../Header/Benchmark.h"


//TODO: Look at ways to reuse allocated memory if possible
//TODO: Make sure memory that can be used again, is still in a valid state after the first execution
//TODO: Move everything over to GEMM because of the solver?

int main(int argc, char **argv)
{
	//Run all the unit tests
	//RunAllUnitTests();


	uint32_t h_valuesPerSample = 26;
	uint32_t h_numberOfSamples = 524288;



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

	float* d_whiteNoiseSignalMatrix = Device_GenerateWhiteNoiseSignal(&rngGen, h_valuesPerSample, h_numberOfSamples);

	//TODO: Debug remove this
	//Utility_DeviceWriteSignalMatrixToFile("signal.txt", d_whiteNoiseSignalMatrix, h_valuesPerSample, h_numberOfSamples, false);


	//2. Create a RFIM Struct
	//--------------------------
	RFIMMemoryStruct* RFIMStruct = RFIMMemoryStructCreate(h_valuesPerSample, h_numberOfSamples, 2);

	//Create space to store the filtered signal
	float* d_filteredSignal;
	cudaError_t cudaError = cudaMalloc(&d_filteredSignal, sizeof(float) * h_valuesPerSample * h_numberOfSamples);

	if(cudaError != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate memory signal\n");
	}

	//3. Run RFIM benchmark
	//--------------------------
	Benchmark(RFIMStruct, d_whiteNoiseSignalMatrix, d_filteredSignal, 10000, 1); //10000

	//RFIMRoutine(RFIMStruct, d_whiteNoiseSignalMatrix, d_filteredSignal);

	//TODO: Debug remove this
	/*
	Utility_DeviceWriteSignalMatrixToFile("meanVec.txt", RFIMStruct->d_meanVec, h_valuesPerSample, 1, false);
	Utility_DeviceWriteSignalMatrixToFile("upperTriangularCovariance.txt", RFIMStruct->d_upperTriangularCovarianceMatrix, h_valuesPerSample, h_valuesPerSample, false);
	Utility_DeviceWriteSignalMatrixToFile("lowerTriangularCovariance.txt", RFIMStruct->d_upperTriangularTransposedMatrix, h_valuesPerSample, h_valuesPerSample, false);
	Utility_DeviceWriteSignalMatrixToFile("covarianceMatrix.txt", RFIMStruct->d_fullSymmetricCovarianceMatrix, h_valuesPerSample, h_valuesPerSample, false);
	Utility_DeviceWriteSignalMatrixToFile("eigenvalues.txt", RFIMStruct->d_S, h_valuesPerSample, 1, false);
	Utility_DeviceWriteSignalMatrixToFile("eigenvectorMatrix.txt", RFIMStruct->d_U, h_valuesPerSample, h_valuesPerSample, false);
	Utility_DeviceWriteSignalMatrixToFile("filteredSignal.txt", d_filteredSignal, h_valuesPerSample, h_numberOfSamples, false);
	*/


	//4. Free everything
	//--------------------------
	//Free the RFIM Struct
	RFIMMemoryStructDestroy(RFIMStruct);
	cudaFree(d_whiteNoiseSignalMatrix);
	cudaFree(d_filteredSignal);


	return 0;
}
