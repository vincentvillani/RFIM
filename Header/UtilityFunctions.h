/*
 * UtilityFunctions.h
 *
 *  Created on: 17/03/2016
 *      Author: vincentvillani
 */

#ifndef UTILITYFUNCTIONS_H_
#define UTILITYFUNCTIONS_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>

#include <string>

//Write a device signal matrix to a file and graph it
void Utility_GraphData(float* d_signalMatrix, uint64_t rows, uint64_t columns, bool transpose);

//Write a host signal matrix to a file
void Utility_WriteSignalMatrixToFile(const std::string filename, const float* h_rowMajorSignalMatrix, uint64_t rows, uint64_t columns);





#endif /* UTILITYFUNCTIONS_H_ */
