/*
 * RFIM.h
 *
 *  Created on: 17/03/2016
 *      Author: vincentvillani
 */

#ifndef RFIM_H_
#define RFIM_H_

#include <stdint.h>

#include "RFIMMemoryStruct.h"
#include "RFIMHelperFunctions.h"


//Does RFIM mitigation and returns the filtered signal matrix device pointer
float* RFIM(RFIMMemoryStruct* RFIMStruct, float* d_columnMajorSignalMatrix, uint32_t h_valuesPerSample, uint32_t h_numberOfSamples);

#endif /* RFIM_H_ */
