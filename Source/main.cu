
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
#include <cublas.h>
#include <string>

#include <stdint.h>

#include "../Header/Kernels.h"
#include "../Header/UnitTests.h"
#include "../Header/CudaMacros.h"
#include "../Header/RFIMHelperFunctions.h"
#include "../Header/CudaUtilityFunctions.h"
#include "../Header/UtilityFunctions.h"

//TODO: Make the user hand in a cublasHandle to use in the RFIMHelper functions



int main(int argc, char **argv)
{
	//Run all the unit tests
	RunAllUnitTests();


	//1. Generate a signal on the device
	//----------------------------------

	uint64_t h_valuesPerSample = 26;
	uint64_t h_numberOfSamples = 1024;

	//Start cuda rand library
	curandGenerator_t rngGen;

	if( curandCreateGenerator(&rngGen, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS)
	{
		fprintf(stderr, "main: Unable to start cuRand library\n");
		exit(1);
	}

	//Set the RNG seed
	if((curandSetPseudoRandomGeneratorSeed(rngGen, 1)) != CURAND_STATUS_SUCCESS)
	{
		fprintf(stderr, "main: Unable to set the RNG Seed value\n");
		exit(1);
	}



	float* d_whiteNoiseSignalMatrix = Device_GenerateWhiteNoiseSignal(&rngGen, h_valuesPerSample, h_numberOfSamples);

	//Destroy the RNG
	if(curandDestroyGenerator(rngGen) != CURAND_STATUS_SUCCESS)
	{
		fprintf(stderr, "Device_GenerateWhiteNoiseSignal: Error in destroying the RNG generator \n");
		exit(1);
	}

	//----------------------------------

	//2.Calculate the covariance matrix of this signal
	//----------------------------------

	//Setup the cublas library
	cublasHandle_t cublasHandle;
	cublasCreate_v2(&cublasHandle);

	float* d_covarianceMatrix = Device_CalculateCovarianceMatrix(&cublasHandle, d_whiteNoiseSignalMatrix, h_valuesPerSample, h_numberOfSamples);


	//Destroy the cublas handle
	cublasDestroy_v2(cublasHandle);

	//----------------------------------

	//3. Graph the covariance matrix
	//----------------------------------

	//Transpose it to row-major (simplify writing to file)
	float* d_covarianceMatrixTranspose = Device_MatrixTranspose(d_covarianceMatrix, h_valuesPerSample, h_valuesPerSample);

	//Copy the signal to host memory
	float* h_covarianceMatrixTranspose = CudaUtility_CopySignalToHost(d_covarianceMatrixTranspose,
			h_valuesPerSample * h_valuesPerSample * sizeof(float));

	//Write the signal to file
	Utility_WriteSignalMatrixToFile(std::string("signal.txt"), h_covarianceMatrixTranspose, h_valuesPerSample, h_valuesPerSample);

	//Graph it via python on own computer!

	//----------------------------------


	//Free all memory
	//----------------------------------

	free(h_covarianceMatrixTranspose);

	cudaFree(d_whiteNoiseSignalMatrix);
	cudaFree(d_covarianceMatrix);
	cudaFree(d_covarianceMatrixTranspose);

	//----------------------------------

	return 0;
}
