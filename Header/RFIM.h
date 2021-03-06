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



//Does RFIM mitigation and returns the filtered signal matrix device pointer
void RFIMRoutine(RFIMMemoryStruct* RFIMStruct, float* d_columnMajorSignalMatrix, float* d_columnMajorFilteredSignalMatrix);

#endif /* RFIM_H_ */
