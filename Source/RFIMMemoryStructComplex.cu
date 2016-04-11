/*
 * RFIMMemoryStructComplex.cu
 *
 *  Created on: 11 Apr 2016
 *      Author: vincentvillani
 */

#include <stdio.h>

#include "../Header/RFIMMemoryStructComplex.h"


RFIMMemoryStructComplex* RFIMMemoryStructComplexCreate(uint64_t h_valuesPerSample, uint64_t h_numberOfSamples, uint64_t h_dimensionToReduce,
		uint64_t h_batchSize, uint64_t h_numberOfCUDAStreams)
{


	RFIMMemoryStructComplex* result;
	cudaMallocHost(&result, sizeof(RFIMMemoryStructComplex));


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
	uint64_t oneVecByteSize = sizeof(cuComplex) * oneVecLength;


	cuComplex* h_oneVec;
	cudaMallocHost(&h_oneVec, oneVecByteSize);
	cudaMalloc(&(result->d_oneVec), oneVecByteSize);

	//Fill the one vec with ones
	for(uint64_t i = 0; i < oneVecLength; ++i)
	{
		h_oneVec[i] = make_cuComplex(1, 0);
	}

	//copy the ones over and free the host memory
	cudaMemcpy(result->d_oneVec, h_oneVec, oneVecByteSize, cudaMemcpyHostToDevice);
	cudaFreeHost(h_oneVec);





	//Setup the mean vec
	//------------------------
	uint64_t meanVecLength = h_valuesPerSample * h_batchSize;
	uint64_t meanVecByteSize = sizeof(cuComplex) * meanVecLength;

	result->h_meanVecBatchOffset = h_valuesPerSample;

	cudaMalloc(&(result->d_meanVec), meanVecByteSize);





	//Setup the covariance matrix
	//------------------------
	uint64_t covarianceMatrixLength = h_valuesPerSample * h_valuesPerSample * h_batchSize;
	uint64_t covarianceMatrixByteSize = sizeof(cuComplex) * covarianceMatrixLength;

	result->h_covarianceMatrixBatchOffset = h_valuesPerSample * h_valuesPerSample;

	cudaMalloc(&(result->d_covarianceMatrix), covarianceMatrixByteSize);




	//Setup the eigenvector/value variables
	//------------------------

	//U and VT
	uint64_t singleULength = h_valuesPerSample * h_valuesPerSample;
	uint64_t ULength = singleULength * h_batchSize;
	uint64_t UByteSize = sizeof(cuComplex) * ULength;

	cudaMalloc(&(result->d_U), UByteSize);
	cudaMalloc(&(result->d_VT), UByteSize); //VT is the same size as U

	result->h_UBatchOffset = singleULength;
	result->h_VTBatchOffset = singleULength;


	//S
	uint64_t singleSLength = h_valuesPerSample;
	uint64_t SLength = h_valuesPerSample * h_batchSize;
	uint64_t SByteLength = sizeof(cuComplex) * SLength;

	cudaMalloc(&(result->d_S), SByteLength);

	result->h_SBatchOffset = singleSLength;


	//Eigenworking space length
	result->h_singleEigWorkingSpaceByteSize = 0;

	cusolverStatus = cusolverDnSgesvd_bufferSize(*result->cusolverHandle, h_valuesPerSample, h_valuesPerSample, &(result->h_singleEigWorkingSpaceByteSize));

	//Check if it went well
	if(cusolverStatus != CUSOLVER_STATUS_SUCCESS)
	{
		fprintf(stderr, "RFIMMemory::RFIMMemory(): Error finding eigenvalue working buffer size\n");
		exit(1);
	}


	uint64_t totalEigenvalueWorkingspace = result->h_singleEigWorkingSpaceByteSize * h_batchSize;

	//Allocate memory for the eigen working space
	cudaMalloc(&(result->d_eigenWorkingSpace), totalEigenvalueWorkingspace);

	result->h_eigenWorkingSpaceBatchOffset = result->h_singleEigWorkingSpaceByteSize / sizeof(int);


	//DevInfo
	uint64_t devInfoSingleLength = 1;
	uint64_t devInfoLength = devInfoSingleLength * h_batchSize;
	uint64_t devInfoByteSize = sizeof(int) * devInfoLength;

	cudaMalloc(&(result->d_devInfo), devInfoByteSize);
	cudaMallocHost(&(result->h_devInfo), devInfoByteSize);

	result->h_devInfoBatchOffset = devInfoSingleLength;






	//Projected signal
	//------------------------
	uint64_t projectedSignalSingleLength = h_valuesPerSample * h_numberOfSamples;
	uint64_t projectedSignalLength = projectedSignalSingleLength * h_batchSize;
	uint64_t projectedSignalByteSize = sizeof(cuComplex) * projectedSignalLength;

	cudaMalloc(&(result->d_projectedSignalMatrix), projectedSignalByteSize);

	result->h_projectedSignalBatchOffset = projectedSignalSingleLength;



	cudaError_t error = cudaDeviceSynchronize();

	if(error != cudaSuccess)
	{
		fprintf(stderr, "RFIMMemoryStructCreate: An error occurred, we probably ran out of device memory...\n");
		exit(1);
	}


	return result;
}

void RFIMMemoryStructComplexDestroy(RFIMMemoryStructComplex* RFIMStruct)
{
	//Free device memory
	cudaFree(RFIMStruct->d_oneVec);
	cudaFree(RFIMStruct->d_meanVec);
	cudaFree(RFIMStruct->d_covarianceMatrix);

	cudaFree(RFIMStruct->d_U);
	cudaFree(RFIMStruct->d_S);
	cudaFree(RFIMStruct->d_VT);
	cudaFree(RFIMStruct->d_eigenWorkingSpace);
	cudaFree(RFIMStruct->d_devInfo);

	cudaFree(RFIMStruct->d_projectedSignalMatrix);


	//Free host memory
	cudaFreeHost(RFIMStruct->h_devInfo);



	//Destroy the cuda library contexts
	cublasDestroy_v2(*RFIMStruct->cublasHandle);
	cusolverDnDestroy(*RFIMStruct->cusolverHandle);


	cudaFreeHost(RFIMStruct->cublasHandle);
	cudaFreeHost(RFIMStruct->cusolverHandle);

	//Destroy the cuda streams
	for(uint64_t i = 0; i < RFIMStruct->h_cudaStreamsLength; ++i)
	{
		cudaStreamDestroy(RFIMStruct->h_cudaStreams[i]);
	}
	cudaFreeHost(RFIMStruct->h_cudaStreams);

	//Deallocate the struct memory on the host
	cudaFreeHost(RFIMStruct);
}
