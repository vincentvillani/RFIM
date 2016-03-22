
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
#include <cusolverDn.h>
#include <string>
#include <stdint.h>

#include "../Header/Kernels.h"
#include "../Header/UnitTests.h"
#include "../Header/CudaMacros.h"
#include "../Header/RFIMHelperFunctions.h"
#include "../Header/CudaUtilityFunctions.h"
#include "../Header/UtilityFunctions.h"


//TODO: Look at ways to reuse allocated memory if possible
//TODO: Make sure memory that can be used again, is still in a valid state after the first execution
//TODO: Move everything over to GEMM because of the solver?

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
		fprintf(stderr, "main: Error in destroying the RNG generator \n");
		exit(1);
	}

	//----------------------------------

	//2.Calculate the covariance matrix of this signal
	//----------------------------------

	//Setup the cublas library
	cublasHandle_t cublasHandle;
	cublasStatus_t cublasStatus = cublasCreate_v2(&cublasHandle);

	if(cublasStatus != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "Main: Error creating a cublas handle\n");
		exit(1);
	}

	//Calculate the covariance matrix
	float* d_triangularCovarianceMatrix = Device_CalculateCovarianceMatrix(&cublasHandle, d_whiteNoiseSignalMatrix, h_valuesPerSample, h_numberOfSamples);




	//----------------------------------

	//3. Graph the covariance matrix
	//----------------------------------

	//Transpose it to row-major (simplify writing to file)
	float* d_triangularCovarianceMatrixTranspose = Device_MatrixTranspose(d_triangularCovarianceMatrix, h_valuesPerSample, h_valuesPerSample);

	//Copy the signal to host memory
	float* h_triangularCovarianceMatrixTranspose = CudaUtility_CopySignalToHost(d_triangularCovarianceMatrixTranspose,
			h_valuesPerSample * h_valuesPerSample * sizeof(float));

	//Write the signal to file
	Utility_WriteSignalMatrixToFile(std::string("signal.txt"), h_triangularCovarianceMatrixTranspose, h_valuesPerSample, h_valuesPerSample);

	//Graph it via python on own computer!

	//----------------------------------


	//4. Calculate the eigenvectors and eigenvalues
	//----------------------------------

	//Create a cusolver handle
	cusolverDnHandle_t cusolverHandle;
	cusolverStatus_t cusolverStatus;

	cusolverStatus = cusolverDnCreate(&cusolverHandle);

	if(cusolverStatus != CUSOLVER_STATUS_SUCCESS)
	{
		fprintf(stderr, "main: error in creating a cusolver handle\n");
		exit(1);
	}

	//Create a full covariance matrix
	float* d_fullCovarianceMatrix = Device_FullSymmetricMatrix(&cublasHandle, d_triangularCovarianceMatrix,
			h_valuesPerSample);


	//Write the full covariance matrix to file, just to check
	float* h_fullCovarianceMatrix = CudaUtility_CopySignalToHost(d_fullCovarianceMatrix, sizeof(float) * h_valuesPerSample * h_valuesPerSample);
	Utility_WriteSignalMatrixToFile(std::string("fullCovarianceMatrix.txt"), h_fullCovarianceMatrix, h_valuesPerSample, h_valuesPerSample);


	//Allocate memory for the eigenvalue solver
	float* d_U;
	float* d_S;
	float* d_VT;
	float* d_Lworkspace;
	int workspaceLength;
	int* d_devInfo;

	cudaMalloc(&d_U, sizeof(float) * h_valuesPerSample * h_valuesPerSample);
	cudaMalloc(&d_S, sizeof(float) * h_valuesPerSample);
	cudaMalloc(&d_VT, sizeof(float) * h_valuesPerSample * h_valuesPerSample);
	cusolverDnSgesvd_bufferSize(cusolverHandle, h_valuesPerSample, h_valuesPerSample, &workspaceLength);
	cudaMalloc(&d_Lworkspace, workspaceLength);
	cudaMalloc(&d_devInfo, sizeof(int));


	//Run the solver
	Device_EigenvalueSolver(&cublasHandle, &cusolverHandle, d_fullCovarianceMatrix, d_U, d_S, d_VT, d_Lworkspace, NULL, workspaceLength, d_devInfo, h_valuesPerSample);


	//Copy the eigenvalues to the host
	float* h_eigenvalues = CudaUtility_CopySignalToHost(d_S, sizeof(float) * h_valuesPerSample);


	//print the values
	for(uint32_t i = 0; i < h_valuesPerSample; ++i)
	{
		printf("eigenvalue %d: %f\n", i, h_eigenvalues[i]);
	}

	//Write the eigenvalues to file
	Utility_WriteSignalMatrixToFile(std::string("eigenvalues.txt"), h_eigenvalues, h_valuesPerSample, 1);

	cusolverDnDestroy(cusolverHandle);

	free(h_eigenvalues);

	cudaFree(d_fullCovarianceMatrix);
	cudaFree(d_U);
	cudaFree(d_S);
	cudaFree(d_VT);
	cudaFree(d_Lworkspace);
	cudaFree(d_devInfo);

	//----------------------------------

	//Free all memory
	//----------------------------------

	free(h_triangularCovarianceMatrixTranspose);

	cudaFree(d_whiteNoiseSignalMatrix);
	cudaFree(d_triangularCovarianceMatrix);
	cudaFree(d_triangularCovarianceMatrixTranspose);

	//Destroy the cublas handle
	cublasStatus = cublasDestroy_v2(cublasHandle);

	if(cublasStatus != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "Main: Error destroying a cublas handle\n");
		exit(1);
	}

	//----------------------------------

	return 0;
}
