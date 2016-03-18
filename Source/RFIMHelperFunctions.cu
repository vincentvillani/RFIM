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

#include "../Header/CudaUtilityFunctions.h"


//Private helper functions
//--------------------------

float* CalculateMeanMatrix(cublasHandle_t* cublasHandle, const float* d_signalMatrix, uint64_t h_valuesPerSample, uint64_t h_numberOfSamples);


//--------------------------


float* Device_GenerateWhiteNoiseSignal(curandGenerator_t* rngGen, uint64_t h_valuesPerSample, uint64_t h_numberOfSamples)
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


	//Generate the signal!
	//Generate random numbers on the device
	//Generate random numbers using a normal distribution
	//Normal distribution should emulate white noise hopefully?
	//Generate signal
	if(curandGenerateNormal(*rngGen, d_signal, totalSignalLength, 0.0f, 1.0f) != CURAND_STATUS_SUCCESS)
	{
		fprintf(stderr, "Device_GenerateWhiteNoiseSignal: Error when generating the signal\n");
		exit(1);
	}


	//Return the generated signal that resides in DEVICE memory
	return d_signal;

}





float* Device_CalculateCovarianceMatrix(const float* d_signalMatrix, uint64_t h_valuesPerSample, uint64_t h_numberOfSamples)
{
	//d_signalMatrix should be column-major as CUBLAS is column-major library (indexes start at 1 also)
	//Remember to take that into account!

	//Setup
	//--------------------------------

	float* d_covarianceMatrix;


	//Setup the cublas library
	//TODO: In the future maybe there should be a shared handle for the cublas library
	cublasHandle_t cublasHandle;
	cublasCreate_v2(&cublasHandle);

	//--------------------------------


	//Calculate the meanMatrix of the signal
	//--------------------------------

	//At this point in time d_covarianceMatrix is actually the mean matrix
	//This is done so I can get better performance out of the cublas API
	d_covarianceMatrix = CalculateMeanMatrix(&cublasHandle, d_signalMatrix, h_valuesPerSample, h_numberOfSamples);

	//--------------------------------


	//Calculate the covariance matrix
	//-------------------------------
	//1. Calculate the outer product of the signal (sampleElements x sampleNumber) * ( sampleNumber x sampleElements)
	//	AKA. signal * (signal)T, where T = transpose, which will give you a (sampleNumber x sampleNumber) matrix as a result

	//Take the outer product of the signal with itself
	float alpha = 1.0f / h_numberOfSamples;
	float beta = -1.0f;

	cublasStatus_t cublasError;
	cublasError = cublasSsyrk_v2(cublasHandle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, h_valuesPerSample, h_numberOfSamples,
			&alpha, d_signalMatrix, h_valuesPerSample, &beta, d_covarianceMatrix, h_valuesPerSample);

	if(cublasError != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "Device_CalculateCovarianceMatrix: error calculating the covariance matrix\n");
		exit(1);
	}


	//Destroy the cublas handle
	cublasDestroy_v2(cublasHandle);

	return d_covarianceMatrix;
}










//Private functions implementation
//----------------------------------

float* CalculateMeanMatrix(cublasHandle_t* cublasHandle, const float* d_signalMatrix, uint64_t h_valuesPerSample, uint64_t h_numberOfSamples)
{

	/*
	//TODO: DEBUG - REMOVE
	float* h_signal = CudaUtility_CopySignalToHost(d_signalMatrix, 6 * sizeof(float));
	for(int i = 0; i < 6; ++i)
	{
		printf("h_signal %d: %f\n", i, h_signal[i]);
	}
	//-----------------------
	*/

	//Setup the one matrix
	//1 x n matrix or n x 1 matrix (doesn't matter in this case) containing only ones
	//--------------------------------------
	float* h_oneMatrix;
	float* d_oneMatrix;

	uint64_t oneMatrixByteSize = sizeof(float) * h_numberOfSamples;

	//Setup memory
	h_oneMatrix = (float*)malloc(oneMatrixByteSize);
	if(cudaMalloc(&d_oneMatrix, oneMatrixByteSize) != cudaSuccess)
	{
		fprintf(stderr, "CalculateMeanMatrix: error allocating %llu bytes for d_oneMatrix\n", oneMatrixByteSize);
		exit(1);
	}

	//Set the matrix on the host
	for(uint64_t i = 0; i < h_numberOfSamples; ++i)
	{
		h_oneMatrix[i] = 1.0f;
	}

	//Copy it over to the device
	d_oneMatrix = CudaUtility_CopySignalToDevice(h_oneMatrix, oneMatrixByteSize);

	//--------------------------------------

	/*
	//TODO: DEBUG - REMOVE
	h_oneMatrix = CudaUtility_CopySignalToHost(d_oneMatrix, oneMatrixByteSize);
	for(int i = 0; i < h_numberOfSamples; ++i)
	{
		printf("oneMatrix %d: %f\n", i, h_oneMatrix[i]);
	}
	//-----------------------
	*/



	//Setup the mean vec
	//
	float* d_meanVec;
	uint64_t meanVecByteSize = sizeof(float) * h_valuesPerSample;

	if(cudaMalloc(&d_meanVec, meanVecByteSize) != cudaSuccess)
	{
		fprintf(stderr, "CalculateMeanMatrix: error allocating %llu bytes for the d_meanVec\n", meanVecByteSize);
		exit(1);
	}

	//Setup the meanMatrix
	//--------------------------------------
	float* d_meanMatrix;
	uint64_t meanMatrixByteSize = h_valuesPerSample * h_valuesPerSample * sizeof(float);

	if( cudaMalloc(&d_meanMatrix, meanMatrixByteSize) != cudaSuccess)
	{
		fprintf(stderr, "CalculateMeanMatrix: error allocating %llu bytes for the d_meanMatrix\n", meanMatrixByteSize);
		exit(1);
	}

	if(cudaMemset(d_meanMatrix, 0, meanMatrixByteSize) != cudaSuccess)
	{
		fprintf(stderr, "CalculateMeanMatrix: error when setting d_meanMatrix to zero\n");
		exit(1);
	}

	//--------------------------------------


	//Calculate d_meanVec
	//d_meanVec = d_oneMatrix (1 x h_numberOfSamples) * d_signal (transposed) (h_numberOfSamples x h_valuesPerSample ) matrix = 1 * h_valuesPerSample matrix
	//This each of the beams added up. It adds up the columns of transposed d_signal
	//---------------------------
	cublasStatus_t cublasError;


	float alpha = 1.0f / h_numberOfSamples;
	float beta = 0;

	cublasError = cublasSgemm_v2(*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 1, h_valuesPerSample, h_numberOfSamples,
			&alpha, d_oneMatrix, 1, d_signalMatrix, h_valuesPerSample, &beta, d_meanVec, 1);

	if(cublasError != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "CalculateMeanMatrix: An error occured while computing d_meanVec\n");
		exit(1);
	}

	//--------------------------------------

	/*
	//TODO: DEBUG - REMOVE
	float* h_meanVec = (float*)malloc(meanVecByteSize);
	h_meanVec = CudaUtility_CopySignalToHost(d_meanVec, meanVecByteSize);
	for(int i = 0; i < h_valuesPerSample; ++i)
	{
		printf("meanVec %d: %f\n", i, h_meanVec[i]);
	}
	//-----------------------
	*/



	//Calculate mean matrix
	//mean matrix = outer product of the transposed d_meanVec with itself
	//d_meanMatrix = d_meanVec_Transposed (h_valuesPerSample x 1) * d_meanVec (1 x h_valuesPerSample)
	//--------------------------------------

	alpha = 1.0f;

	cublasError = cublasSsyrk_v2(*cublasHandle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, h_valuesPerSample, 1,
			&alpha, d_meanVec, 1, &beta, d_meanMatrix, h_valuesPerSample);

	if(cublasError != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "CalculateMeanMatrix: An error occured while computing d_meanMatrix\n");
		exit(1);
	}

	//--------------------------------------

	//Free d_oneMatrix and d_meanVec
	cudaFree(d_oneMatrix);
	cudaFree(d_meanVec);

	free(h_oneMatrix);

	return d_meanMatrix;
}



float* Device_MatrixTranspose(const float* d_matrix, uint64_t rowNum, uint64_t colNum)
{

	float* d_transposedMatrix;

	cudaError_t cudaError;
	cudaError = cudaMalloc(&d_transposedMatrix, rowNum * colNum * sizeof(float));

	if(cudaError != cudaSuccess)
	{
		fprintf(stderr, "Device_MatrixTranspose: Failed to allocate %llu bytes\n", rowNum * colNum * sizeof(float));
		exit(1);
	}


	cublasHandle_t cublasHandle;
	cublasCreate_v2(&cublasHandle);

	cublasStatus_t cublasStatus;

	float alpha = 1;
	float beta = 0;

	//cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, &alpha, d_matrix, M, &beta, d_matrix, N, d_matrixT, N);


	cublasStatus = cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, colNum, rowNum,
			&alpha, d_matrix, rowNum,
			&beta, d_matrix, rowNum,
			d_transposedMatrix, colNum);


	if(cublasStatus != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "Device_InplaceMatrixTranspose: Transposition of the matrix failed!\n");
		exit(1);
	}

	cublasDestroy_v2(cublasHandle);

	return d_transposedMatrix;
}



float* DEBUG_CALCULATE_MEAN_MATRIX(float* d_signalMatrix, uint64_t h_valuesPerSample, uint64_t h_numberOfSamples)
{
	cublasHandle_t cublasHandle;
	cublasCreate_v2(&cublasHandle);

	float* d_meanMatrix = CalculateMeanMatrix(&cublasHandle, d_signalMatrix, h_valuesPerSample, h_numberOfSamples);

	cublasDestroy_v2(cublasHandle);

	return d_meanMatrix;
}


