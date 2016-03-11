
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
	float* d_signal; //row major matrix containing the generated signal, (sampleNumber x sampleElements) matrix
	float* d_covarianceMatrix; //row major in the end, this will be a (sampleElements x sampleElements) covariance matrix


	uint64_t sampleElements = 26; //The number of elements per sample (i.e. b1p1, b1p2, b2p1,b2p2) where b = beam, p = polarisation
	uint64_t sampleNumber = 1024; //Numbers of samples being used to compute the covariance matrix in this iteration
	uint64_t totalElements = sampleElements * sampleNumber; //The total number of elements in this signal
	uint64_t covarianceMatrixElements = upperTriangularLength(sampleElements); //The number of elements that will end up in the covariance matrix
	curandGenerator_t rngGen;


	//Allocate data and set data
	cudaMalloc(&d_signal, sizeof(float) * totalElements);
	cudaMalloc(&d_covarianceMatrix, sizeof(float) * covarianceMatrixElements);
	cudaMemset(d_covarianceMatrix, 0, sizeof(float) * covarianceMatrixElements);



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
	//Generate signal
	if(curandGenerateNormal(rngGen, d_signal, totalElements, 0.0f, 1.0f) != CURAND_STATUS_SUCCESS)
	{
		printf("Error at %s:%d\n",__FILE__,__LINE__);
		exit(1);
	}

	//Startup cublas
	cublasHandle_t cublasHandle;
	cublasCreate_v2(&cublasHandle);


	//Calculate the covariance matrix
	//-------------------------------
	//1. Calculate the outer product of the signal (sampleNumber x sampleElements) * (sampleElements x sampleNumber)
	//	AKA. signal * (signal)T, where T = transpose, which will give you a (sampleNumber x sampleNumber) matrix as a result

	//Take the outer product of the signal with itself
	float alpha = 1.0f;
	float beta = 1.0f;
	cublasSsyrk_v2(cublasHandle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, sampleElements, sampleNumber, &alpha, d_signal, sampleElements,
			&beta, d_covarianceMatrix, sampleElements);

	//2. Calculate the mean using cublas

	//3. Take the outer product of the mean with itself

	//4. Subtract the result of step 3 from the result of step 1. This will give you a covariance matrix

	//5. plot the covariance matrix


	//-------------------------------





	//Destroy the RNG
	if(curandDestroyGenerator(rngGen) != CURAND_STATUS_SUCCESS)
	{
		printf("Error at %s:%d\n",__FILE__,__LINE__);
		exit(1);
	}

	//Free memory
	cudaFree(d_signal);
	cudaFree(d_covarianceMatrix);

	//Free cublas
	cublasDestroy_v2(cublasHandle);
	*/
	return 0;
}





/*
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
*/
