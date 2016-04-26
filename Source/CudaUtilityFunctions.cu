/*
 * CudaUtilityFunction.cpp
 *
 *  Created on: 26 Apr 2016
 *      Author: vincentvillani
 */

#include "../Header/CudaUtilityFunctions.h"

#include <cuda.h>


float** CudaUtility_createBatchedDevicePointers(float* d_basePointer, uint64_t h_offset, uint64_t h_length)
{

	uint64_t resultBatchedPointersByteSize = sizeof(float*) * h_length;

	float** h_resultBatchedPointers;
	cudaMallocHost(&h_resultBatchedPointers, resultBatchedPointersByteSize);

	for(uint64_t i = 0; i < h_length; ++i)
	{
		h_resultBatchedPointers[i] = d_basePointer + (i * h_offset);
	}

	//Copy the pointers to the device
	float** d_resultBatchedPointers;
	cudaMalloc(&d_resultBatchedPointers, resultBatchedPointersByteSize);
	cudaMemcpy(d_resultBatchedPointers, h_resultBatchedPointers, resultBatchedPointersByteSize, cudaMemcpyHostToDevice);

	//Free the host pointers
	cudaFreeHost(h_resultBatchedPointers);

	return d_resultBatchedPointers;
}
