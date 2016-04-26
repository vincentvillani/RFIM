/*
 * CudaUtilityFunctions.h
 *
 *  Created on: 26 Apr 2016
 *      Author: vincentvillani
 */

#ifndef CUDAUTILITYFUNCTIONS_H_
#define CUDAUTILITYFUNCTIONS_H_

#include <stdint.h>

//Creates an array of pointers to pointers with a length of 'length'.
//Each pointer in the resulting array points to memory in d_basePointer
float** CudaUtility_createBatchedDevicePointers(float* d_basePointer, uint64_t h_offset, uint64_t h_length);



#endif /* CUDAUTILITYFUNCTIONS_H_ */
