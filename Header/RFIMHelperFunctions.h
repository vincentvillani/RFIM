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
#include "../Header/RFIMMemoryStructBatched.h"
#include "../Header/RFIMMemoryStructComplex.h"
#include "../Header/RFIMMemoryStructCPU.h"


float* Device_GenerateWhiteNoiseSignal(curandGenerator_t* rngGen, uint64_t h_valuesPerSample, uint64_t h_numberOfSamples, uint64_t h_batchSize);
float* Device_GenerateWhiteNoiseSignal(curandGenerator_t* rngGen, uint64_t h_valuesPerSample, uint64_t h_numberOfSamples,
		uint64_t h_batchSize, uint64_t h_threadNum);
//float* Device_GenerateWhiteNoiseSignal(curandGenerator_t* rngGen, uint64_t h_valuesPerSample, uint64_t h_numberOfSamples,
//		uint64_t h_batchSize, uint64_t h_threadNum, float mean, float stdDev);
cuComplex* Device_GenerateWhiteNoiseSignalComplex(curandGenerator_t* rngGen, uint64_t h_valuesPerSample, uint64_t h_numberOfSamples,
		uint64_t h_batchSize, uint64_t h_threadNum);

void Device_CalculateMeanMatrices(RFIMMemoryStruct* RFIMStruct, float* d_signalMatrices);
void Device_CalculateMeanMatricesBatched(RFIMMemoryStructBatched* RFIMStruct, float** d_signalMatrices);
void Device_CalculateMeanMatricesComplex(RFIMMemoryStructComplex* RFIMStruct, cuComplex* d_signalMatrices);
void Host_CalculateMeanMatrices(RFIMMemoryStructCPU* RFIMStruct, float* h_signalMatrices);

void Device_CalculateCovarianceMatrix(RFIMMemoryStruct* RFIMStruct, float* d_signalMatrices);
void Device_CalculateCovarianceMatrixBatched(RFIMMemoryStructBatched* RFIMStruct, float** d_signalMatrices);
void Device_CalculateCovarianceMatrixComplex(RFIMMemoryStructComplex* RFIMStruct, cuComplex* d_signalMatrices);

void Device_EigenvalueSolver(RFIMMemoryStruct* RFIMStruct);
void Device_EigenvalueSolverBatched(RFIMMemoryStructBatched* RFIMStruct);
void Device_EigenvalueSolverComplex(RFIMMemoryStructComplex* RFIMStruct);

void Device_EigenReductionAndFiltering(RFIMMemoryStruct* RFIMStruct, float* d_originalSignalMatrices, float* d_filteredSignals);
void Device_EigenReductionAndFilteringBatched(RFIMMemoryStructBatched* RFIMStruct, float** d_originalSignalMatrices, float** d_filteredSignals);
void Device_EigenReductionAndFilteringComplex(RFIMMemoryStructComplex* RFIMStruct, cuComplex* d_originalSignalMatrices, cuComplex* d_filteredSignals);



//void Device_MatrixTranspose(cublasHandle_t* cublasHandle, const float* d_matrix, float* d_matrixTransposed, uint64_t rowNum, uint64_t colNum);





#endif /* RFIMHELPERFUNCTIONS_H_ */
