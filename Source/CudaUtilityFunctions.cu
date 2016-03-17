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
float* Utility_CopySignalToDevice(float* h_signal, uint64_t signalByteSize)
{
	float* d_result;

	//Allocate memory
	if( cudaMalloc(&d_result, signalByteSize) != cudaSuccess )
	{
		fprintf(stderr, "CopySignalToDevice: error allocating %llu bytes of memory\n", signalByteSize);
		exit(1);
	}

	//copy data from host to device
	if( cudaMemcpy(d_result, h_signal, signalByteSize, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		fprintf(stderr, "CopySignalToDevice: error copying memory to the device\n");
		exit(1);
	}

	return d_result;
}

//Copies data from the device to the host and returns a host pointer
float* Utility_CopySignalToHost(float* d_signal, uint64_t signalByteSize)
{
	float* h_result;

	//Allocate memory
	h_result = (float*)malloc(signalByteSize);

	if(h_result == NULL)
	{
		fprintf(stderr, "CopySignalToHost: error allocating %llu bytes of memory\n", signalByteSize);
		exit(1);
	}

	//copy data from host to device
	if( cudaMemcpy(h_result, d_signal, signalByteSize, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		fprintf(stderr, "CopySignalToHost: error copying memory to the host\n");
		exit(1);
	}

	return h_result;
}

