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

	//copy data from host to device
	if( cudaMemcpy(*d_destination, h_signal, signalByteSize, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		fprintf(stderr, "CopySignalToDevice: error copying memory to the device\n");
		//exit(1);
	}

}

//Copies data from the device to the host and returns a host pointer
void CudaUtility_CopySignalToHost(float* d_signal, float** h_destination, uint64_t signalByteSize)
{
	//copy data from host to device
	if( cudaMemcpy(*h_destination, d_signal, signalByteSize, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		fprintf(stderr, "CopySignalToHost: error copying memory to the host\n");
		//exit(1);
	}

}

