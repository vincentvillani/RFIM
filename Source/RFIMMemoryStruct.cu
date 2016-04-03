
#include "../Header/RFIMMemoryStruct.h"

#include "../Header/CudaUtilityFunctions.h"

#include <stdio.h>


RFIMMemoryStruct* RFIMMemoryStructCreate(uint32_t h_valuesPerSample, uint32_t h_numberOfSamples, uint32_t h_dimensionToReduce, uint32_t h_batchSize)
{
	RFIMMemoryStruct* result = (RFIMMemoryStruct*)malloc(sizeof(RFIMMemoryStruct));

	result->cublasHandle = (cublasHandle_t*)malloc(sizeof(cublasHandle_t));
	result->cusolverHandle = (cusolverDnHandle_t*)malloc(sizeof(cusolverDnHandle_t));

	cublasStatus_t cublasStatus;
	cusolverStatus_t cusolverStatus;

	//Create the contexts for each library
	cublasStatus = cublasCreate_v2( result->cublasHandle );
	cusolverStatus = cusolverDnCreate( result->cusolverHandle );

	//Check the contexts started up ok
	if(cublasStatus != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "RFIMMemory::RFIMMemory(): Error starting cublas context\n");
		//exit(1);
	}

	if(cusolverStatus != CUSOLVER_STATUS_SUCCESS)
	{
		fprintf(stderr, "RFIMMemory::RFIMMemory(): Error starting cusolver context\n");
		//exit(1);
	}


	//Set signal attributes
	result->h_valuesPerSample = h_valuesPerSample;
	result->h_numberOfSamples = h_numberOfSamples;
	result->h_eigenVectorDimensionsToReduce = h_dimensionToReduce;
	result->h_batchSize = h_batchSize;



	//Setup the one vec
	//------------------------
	uint32_t oneVecByteSize = sizeof(float) * h_numberOfSamples;
	float* h_oneVec = (float*)malloc(oneVecByteSize);
	float** h_oneVecPointerArray = (float**)malloc(sizeof(float*) * h_batchSize);

	//printf("0\n");

	//Fill the one vec with ones
	for(uint32_t i = 0; i < h_numberOfSamples; ++i)
	{
		h_oneVec[i] = 1;
	}

	//printf("0.5\n");

	//Set each pointer to point to the same array
	for(uint32_t i = 0; i < h_batchSize; ++i)
	{
		h_oneVecPointerArray[i] = h_oneVec;
	}

	//printf("0.75\n");


	//Allocate one array on the device, everything in the pointer array will point to this
	//float* d_oneVec;
	//cudaMalloc(&d_oneVec, oneVecByteSize);
	//cudaMemcpy(d_oneVec, h_oneVec, oneVecByteSize, cudaMemcpyHostToDevice);



	uint64_t meanVecByteSize = sizeof(float) * h_valuesPerSample;
	uint64_t covarianceMatrixByteSize = sizeof(float) * h_valuesPerSample * h_valuesPerSample;
	uint64_t UByteSize = sizeof(float) * h_valuesPerSample * h_valuesPerSample;
	uint64_t SByteSize = sizeof(float) * h_valuesPerSample;
	uint64_t VTByteSize = sizeof(float) * h_valuesPerSample * h_valuesPerSample;
	//uint32_t devInfoByteSize = sizeof(int);
	//Ask cusolver for the needed buffer size
	result->h_eigWorkingSpaceLength = 0;
	cusolverStatus = cusolverDnSgesvd_bufferSize(*result->cusolverHandle, h_valuesPerSample, h_valuesPerSample, &(result->h_eigWorkingSpaceLength));
	//Check if it went well
	if(cusolverStatus != CUSOLVER_STATUS_SUCCESS)
	{
		fprintf(stderr, "RFIMMemory::RFIMMemory(): Error finding eigenvalue working buffer size\n");
		//exit(1);
	}
	uint32_t projectedSignalMatrixByteSize = sizeof(float) * ((h_valuesPerSample - result->h_eigenVectorDimensionsToReduce) * h_numberOfSamples);

	//printf("1\n");

	//Allocate 2D pointers on the device
	result->d_oneVec = CudaUtility_BatchAllocateDeviceArrays(h_batchSize, oneVecByteSize);
	CudaUtility_BatchCopyArraysHostToDevice(result->d_oneVec, h_oneVecPointerArray, h_batchSize, oneVecByteSize); //Copy the oneVec data to the 2D array

	result->d_meanVec = CudaUtility_BatchAllocateDeviceArrays(h_batchSize, meanVecByteSize);

	result->d_covarianceMatrix = CudaUtility_BatchAllocateDeviceArrays(h_batchSize, covarianceMatrixByteSize);
	result->d_U = CudaUtility_BatchAllocateDeviceArrays(h_batchSize, UByteSize);
	result->d_S = CudaUtility_BatchAllocateDeviceArrays(h_batchSize, SByteSize);
	result->d_VT = CudaUtility_BatchAllocateDeviceArrays(h_batchSize, VTByteSize);
	result->d_eigWorkingSpace = CudaUtility_BatchAllocateDeviceArrays(h_batchSize, result->h_eigWorkingSpaceLength);
	cudaMalloc(&result->d_devInfo, sizeof(int) * h_batchSize);
	result->h_devInfoValues = (int*)malloc(sizeof(int) * h_batchSize);



	//Allocate space for the pointers to device memory, this is used to speed up the eigenvector solver part of the RFIM
	uint64_t pointersArrayByteSize = sizeof(float*) * h_batchSize;

	result->h_covarianceMatrixDevicePointers = (float**)malloc(pointersArrayByteSize);
	result->h_UDevicePointers = (float**)malloc(pointersArrayByteSize);
	result->h_SDevicePointers = (float**)malloc(pointersArrayByteSize);
	result->h_VTDevicePointers = (float**)malloc(pointersArrayByteSize);
	result->h_eigWorkingSpaceDevicePointers = (float**)malloc(pointersArrayByteSize);
	//result->h_devInfoDevicePointers = (int**)malloc(pointersArrayByteSize);

	//Copy the pointers to device memory over to the host memory
	cudaMemcpy(result->h_covarianceMatrixDevicePointers, result->d_covarianceMatrix, pointersArrayByteSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(result->h_UDevicePointers, result->d_U, pointersArrayByteSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(result->h_SDevicePointers, result->d_S, pointersArrayByteSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(result->h_VTDevicePointers, result->d_VT, pointersArrayByteSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(result->h_eigWorkingSpaceDevicePointers, result->d_eigWorkingSpace, pointersArrayByteSize, cudaMemcpyDeviceToHost);
	//cudaMemcpy(result->h_devInfoDevicePointers, result->d_devInfo, pointersArrayByteSize, cudaMemcpyDeviceToHost);

	//Allocate space for the h_devInfoValues
	//result->h_devInfoValues = (int*)malloc(sizeof(int) * h_batchSize);


	//printf("2\n");


	//Free memory
	//-----------------------------
	free(h_oneVec);
	free(h_oneVecPointerArray);

	//printf("3\n");

	return result;
}



void RFIMMemoryStructDestroy(RFIMMemoryStruct* RFIMStruct)
{
	//Destroy the cuda library contexts
	cublasDestroy_v2(*RFIMStruct->cublasHandle);
	cusolverDnDestroy(*RFIMStruct->cusolverHandle);

	free(RFIMStruct->cublasHandle);
	free(RFIMStruct->cusolverHandle);

	CudaUtility_BatchDeallocateDeviceArrays(RFIMStruct->d_oneVec, RFIMStruct->h_batchSize);
	CudaUtility_BatchDeallocateDeviceArrays(RFIMStruct->d_meanVec, RFIMStruct->h_batchSize);
	CudaUtility_BatchDeallocateDeviceArrays(RFIMStruct->d_covarianceMatrix, RFIMStruct->h_batchSize);
	CudaUtility_BatchDeallocateDeviceArrays(RFIMStruct->d_U, RFIMStruct->h_batchSize);
	CudaUtility_BatchDeallocateDeviceArrays(RFIMStruct->d_S, RFIMStruct->h_batchSize);
	CudaUtility_BatchDeallocateDeviceArrays(RFIMStruct->d_VT, RFIMStruct->h_batchSize);
	CudaUtility_BatchDeallocateDeviceArrays(RFIMStruct->d_eigWorkingSpace, RFIMStruct->h_batchSize);
	cudaFree(RFIMStruct->d_devInfo);

	//Free the host pointers to device memory
	free(RFIMStruct->h_covarianceMatrixDevicePointers);
	free(RFIMStruct->h_UDevicePointers);
	free(RFIMStruct->h_SDevicePointers);
	free(RFIMStruct->h_VTDevicePointers);
	free(RFIMStruct->h_eigWorkingSpaceDevicePointers);
	free(RFIMStruct->h_devInfoValues);

	//Deallocate the struct memory on the host
	free(RFIMStruct);

}


