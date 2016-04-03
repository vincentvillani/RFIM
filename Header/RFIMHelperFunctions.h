/*
 * RFIMHelperFunctions.h
 *
 *  Created on: 17/03/2016
 *      Author: vincentvillani
 */

#ifndef RFIMHELPERFUNCTIONS_H_
#define RFIMHELPERFUNCTIONS_H_

#include <stdint.h>
#include <cuda.h>
#include <curand.h>
#include <cublas.h>
#include <cusolverDn.h>

#include "../Header/RFIMMemoryStruct.h"


float* Device_GenerateWhiteNoiseSignal(curandGenerator_t* rngGen, uint64_t h_valuesPerSample, uint64_t h_numberOfSamples);
void Device_CalculateMeanMatrices(RFIMMemoryStruct* RFIMStruct, float** d_signalMatrices);
void Device_CalculateCovarianceMatrix(RFIMMemoryStruct* RFIMStruct, float** d_signalMatrices);
void Device_EigenvalueSolver(RFIMMemoryStruct* RFIMStruct);
/*
void Device_EigenReductionAndFiltering(RFIMMemoryStruct* RFIMStruct, float* d_originalSignalMatrix, float* d_filteredSignal);
void Device_MatrixTranspose(cublasHandle_t* cublasHandle, const float* d_matrix, float* d_matrixTransposed, uint64_t rowNum, uint64_t colNum);
*/




#endif /* RFIMHELPERFUNCTIONS_H_ */
