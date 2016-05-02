/*
 * RFIM.h
 *
 *  Created on: 17/03/2016
 *      Author: vincentvillani
 */

#ifndef RFIM_H_
#define RFIM_H_

#include <stdint.h>

#include <cuComplex.h>

#include "RFIMMemoryStruct.h"
#include "RFIMMemoryStructBatched.h"
#include "RFIMMemoryStructComplex.h"
#include "RFIMMemoryStructCPU.h"



//Does RFIM mitigation and returns the filtered signal matrix device pointer
void RFIMRoutine(RFIMMemoryStruct* RFIMStruct, float* d_columnMajorSignalMatrices, float* d_columnMajorFilteredSignalMatrices);
void RFIMRoutineBatched(RFIMMemoryStructBatched* RFIMStruct, float** d_columnMajorSignalMatrices, float** d_columnMajorFilteredSignalMatrices);
void RFIMRoutineComplex(RFIMMemoryStructComplex* RFIMStruct, cuComplex* d_columnMajorSignalMatrices, cuComplex* d_columnMajorFilteredSignalMatrices);
void RFIMRoutineHost(RFIMMemoryStructCPU* RFIMStruct, float* h_columnMajorSignalMatrices, float* h_columnMajorFilteredSignalMatrices);

#endif /* RFIM_H_ */
