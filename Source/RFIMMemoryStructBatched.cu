/*
 * RFIMMemoryStructBatched.cu
 *
 *  Created on: 26 Apr 2016
 *      Author: vincentvillani
 */

#include "../Header/RFIMMemoryStructBatched.h"

#include <stdio.h>

#include "../Header/CudaUtilityFunctions.h"


RFIMMemoryStructBatched* RFIMMemoryStructBatchedCreate(uint64_t h_valuesPerSample, uint64_t h_numberOfSamples, uint64_t h_dimensionToReduce,
		uint64_t h_batchSize, uint64_t h_numberOfCUDAStreams)
{
	RFIMMemoryStructBatched* result;
	cudaMallocHost(&result, sizeof(RFIMMemoryStructBatched));




	//Set signal attributes
	//------------------------
	result->h_valuesPerSample = h_valuesPerSample;
	result->h_numberOfSamples = h_numberOfSamples;
	result->h_eigenVectorDimensionsToReduce = h_dimensionToReduce;
	result->h_batchSize = h_batchSize;
	result->h_cudaStreamsLength = h_numberOfCUDAStreams;



	//Setup library handles
	//------------------------
	cudaMallocHost(&(result->cublasHandle), sizeof(cublasHandle_t));
	cudaMallocHost(&(result->cusolverHandle), sizeof(cusolverDnHandle_t));

	cublasStatus_t cublasStatus;
	cusolverStatus_t cusolverStatus;

	//Create the contexts for each library
	cublasStatus = cublasCreate_v2( result->cublasHandle );
	cusolverStatus = cusolverDnCreate( result->cusolverHandle );

	//Check the contexts started up ok
	if(cublasStatus != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "RFIMMemory::RFIMMemory(): Error starting cublas context\n");
		exit(1);
	}

	if(cusolverStatus != CUSOLVER_STATUS_SUCCESS)
	{
		fprintf(stderr, "RFIMMemory::RFIMMemory(): Error starting cusolver context\n");
		exit(1);
	}


	//Allocate space for the cudaSteams
	cudaMallocHost(&(result->h_cudaStreams), sizeof(cudaStream_t) * h_numberOfCUDAStreams);
	for(uint64_t i = 0; i < h_numberOfCUDAStreams; ++i)
	{
		cudaStreamCreate(result->h_cudaStreams + i);
	}




	//Setup the one vec, we use the same memory over and over again, it should never change
	//------------------------
	uint64_t oneVecLength = h_numberOfSamples;
	uint64_t oneVecByteSize = sizeof(float) * oneVecLength;


	float* h_oneVec;
	cudaMallocHost(&h_oneVec, oneVecByteSize);
	cudaMalloc(&(result->d_oneVec), oneVecByteSize);

	//Fill the one vec with ones
	for(uint64_t i = 0; i < oneVecLength; ++i)
	{
		h_oneVec[i] = 1;
	}

	//copy the ones over and free the host memory
	cudaMemcpy(result->d_oneVec, h_oneVec, oneVecByteSize, cudaMemcpyHostToDevice);
	cudaFreeHost(h_oneVec);


	//Setup the batched pointers to the device memory
	//Each pointer will point to the same point in memory as offset is set to zero
	result->d_oneVecBatched = CudaUtility_createBatchedDevicePointers(result->d_oneVec, 0, h_batchSize);




	//Setup the mean vec
	//------------------------
	uint64_t meanVecLength = h_valuesPerSample * h_batchSize;
	uint64_t meanVecByteSize = sizeof(float) * meanVecLength;

	result->h_meanVecBatchOffset = h_valuesPerSample;

	cudaMalloc(&(result->d_meanVec), meanVecByteSize);

	//Setup the batched pointers to meanVec memory
	result->d_meanVecBatched = CudaUtility_createBatchedDevicePointers(result->d_meanVec,
			result->h_meanVecBatchOffset, h_batchSize);





	//Setup the covariance matrix
	//------------------------
	uint64_t covarianceMatrixLength = h_valuesPerSample * h_valuesPerSample * h_batchSize;
	uint64_t covarianceMatrixByteSize = sizeof(float) * covarianceMatrixLength;

	result->h_covarianceMatrixBatchOffset = h_valuesPerSample * h_valuesPerSample;

	cudaMalloc(&(result->d_covarianceMatrix), covarianceMatrixByteSize);

	//Setup the batched pointers to covarianceMatrix memory
	result->d_covarianceMatrixBatched = CudaUtility_createBatchedDevicePointers(result->d_covarianceMatrix,
			result->h_covarianceMatrixBatchOffset, h_batchSize);



	return result;

}




void RFIMMemoryStructDestroy(RFIMMemoryStructBatched* RFIMStruct)
{
	//Free device memory
	cudaFree(RFIMStruct->d_oneVec);
	cudaFree(RFIMStruct->d_oneVecBatched);

	cudaFree(RFIMStruct->d_meanVec);
	cudaFree(RFIMStruct->d_meanVecBatched);

	cudaFree(RFIMStruct->d_covarianceMatrix);
	cudaFree(RFIMStruct->d_covarianceMatrixBatched);
}
