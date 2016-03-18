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
float* Utility_CopySignalToDevice(float* h_signal, uint64_t signalByteSize);

//Copies data from the device to the host and returns a host pointer
float* Utility_CopySignalToHost(float* d_signal, uint64_t signalByteSize);



#endif /* CUDAUTILITYFUNCTIONS_H_ */
