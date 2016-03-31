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

