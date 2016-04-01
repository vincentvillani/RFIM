/*
 * CudaUtilityFunctions.cu
 *
 *  Created on: 17/03/2016
 *      Author: vincentvillani
 */


#include "../Header/CudaUtilityFunctions.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

//Copies data from the host to the device and returns a device pointer
void CudaUtility_CopySignalToDevice(float* h_signal, float** d_destination, uint64_t signalByteSize)
{
	cudaError_t cudaError;

	cudaError = cudaMemcpy(*d_destination, h_signal, signalByteSize, cudaMemcpyHostToDevice);

	//copy data from host to device
	if( cudaError != cudaSuccess)
	{
		fprintf(stderr, "CopySignalToDevice: error copying memory to the device\n");
		fprintf(stderr, "Cuda error code: %s\n", cudaGetErrorString(cudaError));
		//exit(1);
	}

}

//Copies data from the device to the host and returns a host pointer
void CudaUtility_CopySignalToHost(float* d_signal, float** h_destination, uint64_t signalByteSize)
{
	cudaError_t cudaError;

	cudaError = cudaMemcpy(*h_destination, d_signal, signalByteSize, cudaMemcpyDeviceToHost);
	//copy data from host to device
	if( cudaError != cudaSuccess)
	{
		fprintf(stderr, "CopySignalToHost: error copying memory to the host\n");
		fprintf(stderr, "Cuda error code: %s\n", cudaGetErrorString(cudaError));
		//exit(1);
	}

}


float** CudaUtility_BatchAllocateDeviceArrays(uint32_t numberOfArrays, uint64_t arrayByteSize)
{
	//Allocate space for the pointers
	float** h_resultDevicePointers = (float**)malloc(sizeof(float*) * numberOfArrays);
	float** d_result;
	cudaMalloc(&d_result, sizeof(float*) * numberOfArrays);


	//Allocate space for each pointer and copy it's address into h_resultDevicePointers + i
	for(uint32_t i = 0; i < numberOfArrays; ++i)
	{
		//Allocate each array's memory and store pointers to it on the host
		cudaMalloc(&(h_resultDevicePointers[i]), arrayByteSize);
	}

	//Copy all the pointers into device memory
	cudaMemcpy(d_result, h_resultDevicePointers, sizeof(float*) * numberOfArrays, cudaMemcpyHostToDevice);

	free(h_resultDevicePointers);

	return d_result;
}



void CudaUtility_BatchDeallocateDeviceArrays(float** d_arrays, uint32_t numberOfArrays)
{
	//Copy the pointers to the host
	float** h_arraysDevicePointers = (float**)malloc(sizeof(float*) * numberOfArrays);
	cudaMemcpy(h_arraysDevicePointers, d_arrays, sizeof(float*) * numberOfArrays, cudaMemcpyDeviceToHost);

	for(uint32_t i = 0; i < numberOfArrays; ++i)
	{
		//Free each array
		cudaFree(h_arraysDevicePointers[i]);
	}

	//Free the device pointers
	cudaFree(d_arrays);

	//Free memory on the host
	free(h_arraysDevicePointers);
}


void CudaUtility_BatchCopyArraysHostToDevice(float** d_arrays, float** h_arrays, uint32_t numberOfArrays, uint64_t arrayByteSize)
{
	uint64_t pointersArrayByteSize = sizeof(float*) * numberOfArrays;

	//Copy the device pointers to the host
	float** h_devicePointers = (float**)malloc(pointersArrayByteSize);
	cudaMemcpy(h_devicePointers, d_arrays, pointersArrayByteSize, cudaMemcpyDeviceToHost);

	for(uint32_t i = 0; i < numberOfArrays; ++i)
	{
		//Copy the actual data across to each pointer
		cudaMemcpy(h_devicePointers[i], h_arrays[i], arrayByteSize, cudaMemcpyHostToDevice);
	}

	free(h_devicePointers);
}

