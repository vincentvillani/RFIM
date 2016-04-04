
#include "../Header/RFIMMemoryStruct.h"

#include "../Header/CudaUtilityFunctions.h"

#include <stdio.h>


RFIMMemoryStruct* RFIMMemoryStructCreate(uint64_t h_valuesPerSample, uint64_t h_numberOfSamples, uint64_t h_dimensionToReduce,
		uint64_t h_batchSize, uint32_t threadIndex)
{

	RFIMMemoryStruct* result;
	cudaMallocHost(&result, sizeof(RFIMMemoryStruct));

	cudaMallocHost(&(result->cublasHandle), sizeof(cublasHandle_t));

	cudaMallocHost(&(result->cusolverHandle), sizeof(cusolverDnHandle_t));

	cublasStatus_t cublasStatus;
	cusolverStatus_t cusolverStatus;

	//Start up the cudaStream
	cudaStreamCreateWithFlags(&result->cudaStream, cudaStreamNonBlocking);

	//Create the contexts for each library
	cublasStatus = cublasCreate_v2( result->cublasHandle );
	cusolverStatus = cusolverDnCreate( result->cusolverHandle );

	//Have the library handles execute on this newly created stream
	cublasSetStream_v2(*result->cublasHandle, result->cudaStream);
	cusolverDnSetStream(*result->cusolverHandle, result->cudaStream);

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


	//Set signal attributes
	result->h_valuesPerSample = h_valuesPerSample;
	result->h_numberOfSamples = h_numberOfSamples;
	result->h_eigenVectorDimensionsToReduce = h_dimensionToReduce;
	result->h_batchSize = h_batchSize;
	result->h_threadId = threadIndex;



	//Setup the one vec
	//------------------------
	uint64_t oneVecByteSize = sizeof(float) * h_numberOfSamples;

	float* h_oneVec;
	cudaMallocHost(&h_oneVec, oneVecByteSize);

	float** h_oneVecPointerArray;
	cudaMallocHost(&h_oneVecPointerArray, sizeof(float*) * h_batchSize);


	//Fill the one vec with ones
	for(uint64_t i = 0; i < h_numberOfSamples; ++i)
	{
		h_oneVec[i] = 1;
	}


	//Set each pointer to point to the same array
	for(uint64_t i = 0; i < h_batchSize; ++i)
	{
		h_oneVecPointerArray[i] = h_oneVec;
	}



	uint64_t meanVecByteSize = sizeof(float) * h_valuesPerSample;
	uint64_t covarianceMatrixByteSize = sizeof(float) * h_valuesPerSample * h_valuesPerSample;
	uint64_t UByteSize = sizeof(float) * h_valuesPerSample * h_valuesPerSample;
	uint64_t SByteSize = sizeof(float) * h_valuesPerSample;
	uint64_t VTByteSize = sizeof(float) * h_valuesPerSample * h_valuesPerSample;

	//Ask cusolver for the needed buffer size
	result->h_eigWorkingSpaceLength = 0;
	cusolverStatus = cusolverDnSgesvd_bufferSize(*result->cusolverHandle, h_valuesPerSample, h_valuesPerSample, &(result->h_eigWorkingSpaceLength));
	//Check if it went well
	if(cusolverStatus != CUSOLVER_STATUS_SUCCESS)
	{
		fprintf(stderr, "RFIMMemory::RFIMMemory(): Error finding eigenvalue working buffer size\n");
		exit(1);
	}
	uint64_t projectedSignalMatrixByteSize = sizeof(float) * h_valuesPerSample * h_numberOfSamples;



	//Allocate 2D pointers on the device
	result->d_oneVec = CudaUtility_BatchAllocateDeviceArrays(h_batchSize, oneVecByteSize, &(result->cudaStream));
	CudaUtility_BatchCopyArraysHostToDevice(result->d_oneVec, h_oneVecPointerArray, h_batchSize, oneVecByteSize, &(result->cudaStream)); //Copy the oneVec data to the 2D array

	result->d_meanVec = CudaUtility_BatchAllocateDeviceArrays(h_batchSize, meanVecByteSize, &(result->cudaStream));

	result->d_covarianceMatrix = CudaUtility_BatchAllocateDeviceArrays(h_batchSize, covarianceMatrixByteSize, &(result->cudaStream));
	result->d_U = CudaUtility_BatchAllocateDeviceArrays(h_batchSize, UByteSize, &(result->cudaStream));
	result->d_S = CudaUtility_BatchAllocateDeviceArrays(h_batchSize, SByteSize, &(result->cudaStream));
	result->d_VT = CudaUtility_BatchAllocateDeviceArrays(h_batchSize, VTByteSize, &(result->cudaStream));
	result->d_eigWorkingSpace = CudaUtility_BatchAllocateDeviceArrays(h_batchSize, result->h_eigWorkingSpaceLength, &(result->cudaStream));
	cudaMalloc(&(result->d_devInfo), sizeof(int) * h_batchSize);
	//result->h_devInfoValues = (int*)malloc(sizeof(int) * h_batchSize);
	cudaMallocHost(&(result->h_devInfoValues), sizeof(int) * h_batchSize);
	result->d_projectedSignalMatrix = CudaUtility_BatchAllocateDeviceArrays(h_batchSize, projectedSignalMatrixByteSize, &(result->cudaStream));



	//Allocate space for the pointers to device memory, this is used to speed up the eigenvector solver part of the RFIM
	uint64_t pointersArrayByteSize = sizeof(float*) * h_batchSize;

	cudaMallocHost(&(result->h_covarianceMatrixDevicePointers), pointersArrayByteSize);
	cudaMallocHost(&(result->h_UDevicePointers), pointersArrayByteSize); //Allocate pinned memory for use with async memcpy
	cudaMallocHost(&(result->h_SDevicePointers), pointersArrayByteSize);
	cudaMallocHost(&(result->h_VTDevicePointers), pointersArrayByteSize);
	cudaMallocHost(&(result->h_eigWorkingSpaceDevicePointers), pointersArrayByteSize);


	//Copy the pointers to device memory over to the host memory
	cudaMemcpyAsync(result->h_covarianceMatrixDevicePointers, result->d_covarianceMatrix, pointersArrayByteSize, cudaMemcpyDeviceToHost, result->cudaStream);
	cudaMemcpyAsync(result->h_UDevicePointers, result->d_U, pointersArrayByteSize, cudaMemcpyDeviceToHost, result->cudaStream);
	cudaMemcpyAsync(result->h_SDevicePointers, result->d_S, pointersArrayByteSize, cudaMemcpyDeviceToHost, result->cudaStream);
	cudaMemcpyAsync(result->h_VTDevicePointers, result->d_VT, pointersArrayByteSize, cudaMemcpyDeviceToHost, result->cudaStream);
	cudaMemcpyAsync(result->h_eigWorkingSpaceDevicePointers, result->d_eigWorkingSpace, pointersArrayByteSize, cudaMemcpyDeviceToHost, result->cudaStream);


	//Wait for all memcopies, memsets etc to occur
	cudaStreamSynchronize(result->cudaStream);

	//Free memory
	//-----------------------------
	cudaFreeHost(h_oneVec);
	cudaFreeHost(h_oneVecPointerArray);


	//Check for errors
	cudaError_t cudaError = cudaGetLastError();

	if(cudaError != cudaSuccess)
	{
		fprintf(stderr, "RFIMMemoryStructCreate: Probably failed to allocate memory\n");
		exit(1);
	}


	return result;
}



void RFIMMemoryStructDestroy(RFIMMemoryStruct* RFIMStruct)
{


	CudaUtility_BatchDeallocateDeviceArrays(RFIMStruct->d_oneVec, RFIMStruct->h_batchSize, &(RFIMStruct->cudaStream));
	CudaUtility_BatchDeallocateDeviceArrays(RFIMStruct->d_meanVec, RFIMStruct->h_batchSize, &(RFIMStruct->cudaStream));
	CudaUtility_BatchDeallocateDeviceArrays(RFIMStruct->d_covarianceMatrix, RFIMStruct->h_batchSize, &(RFIMStruct->cudaStream));
	CudaUtility_BatchDeallocateDeviceArrays(RFIMStruct->d_U, RFIMStruct->h_batchSize, &(RFIMStruct->cudaStream));
	CudaUtility_BatchDeallocateDeviceArrays(RFIMStruct->d_S, RFIMStruct->h_batchSize, &(RFIMStruct->cudaStream));
	CudaUtility_BatchDeallocateDeviceArrays(RFIMStruct->d_VT, RFIMStruct->h_batchSize, &(RFIMStruct->cudaStream));
	CudaUtility_BatchDeallocateDeviceArrays(RFIMStruct->d_eigWorkingSpace, RFIMStruct->h_batchSize, &(RFIMStruct->cudaStream));
	cudaFree(RFIMStruct->d_devInfo);
	CudaUtility_BatchDeallocateDeviceArrays(RFIMStruct->d_projectedSignalMatrix, RFIMStruct->h_batchSize, &(RFIMStruct->cudaStream));

	//Free the host pointers to device memory
	cudaFreeHost(RFIMStruct->h_covarianceMatrixDevicePointers);
	cudaFreeHost(RFIMStruct->h_UDevicePointers);
	cudaFreeHost(RFIMStruct->h_SDevicePointers);
	cudaFreeHost(RFIMStruct->h_VTDevicePointers);
	cudaFreeHost(RFIMStruct->h_eigWorkingSpaceDevicePointers);
	cudaFreeHost(RFIMStruct->h_devInfoValues);

	//Destroy the cuda library contexts
	cublasDestroy_v2(*RFIMStruct->cublasHandle);
	cusolverDnDestroy(*RFIMStruct->cusolverHandle);


	cudaFreeHost(RFIMStruct->cublasHandle);
	cudaFreeHost(RFIMStruct->cusolverHandle);

	cudaStreamDestroy(RFIMStruct->cudaStream);

	//Deallocate the struct memory on the host
	cudaFreeHost(RFIMStruct);

}


