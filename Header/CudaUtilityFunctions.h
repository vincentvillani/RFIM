/*
 * CudaUtilityFunctions.h
 *
 *  Created on: 17/03/2016
 *      Author: vincentvillani
 */

#ifndef CUDAUTILITYFUNCTIONS_H_
#define CUDAUTILITYFUNCTIONS_H_

#include <stdint.h>

#include <cuda.h>

//Copies data from the host to the device and returns a device pointer
void CudaUtility_CopySignalToDevice(float* h_signal, float** d_destination, uint64_t signalByteSize, cudaStream_t* cudaStream);

//Copies data from the device to the host and returns a host pointer
void CudaUtility_CopySignalToHost(float* d_signal, float** h_destination, uint64_t signalByteSize, cudaStream_t* cudaStream);

float** CudaUtility_BatchAllocateDeviceArrays(uint32_t numberOfArrays, uint64_t arrayByteSize, cudaStream_t* cudaStream);
void CudaUtility_BatchDeallocateDeviceArrays(float** d_arrays, uint32_t numberOfArrays, cudaStream_t* cudaStream);

float** CudaUtility_BatchAllocateHostArrays(uint32_t numberOfArrays, uint64_t arrayByteSize);
void CudaUtility_BatchDeallocateHostArrays(float** h_arrays, uint32_t numberOfArrays);

void CudaUtility_BatchCopyArraysHostToDevice(float** d_arrays, float** h_arrays, uint32_t numberOfArrays, uint64_t arrayByteSize, cudaStream_t* cudaStream);
void CudaUtility_BatchCopyArraysDeviceToHost(float** d_arrays, float** h_arrays, uint32_t numberOfArrays, uint64_t arrayByteSize, cudaStream_t* cudaStream);

#endif /* CUDAUTILITYFUNCTIONS_H_ */
