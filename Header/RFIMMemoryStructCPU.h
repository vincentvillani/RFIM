/*
 * RFIMMemoryStructCPU.h
 *
 *  Created on: 28 Apr 2016
 *      Author: vincentvillani
 */

#ifndef RFIMMEMORYSTRUCTCPU_H_
#define RFIMMEMORYSTRUCTCPU_H_

#include <stdint.h>

typedef struct RFIMMemoryStructCPU
{

	//Signal attributes, these need to be set before use
	uint64_t h_valuesPerSample;
	uint64_t h_numberOfSamples;
	uint64_t h_batchSize;
	uint64_t h_eigenVectorDimensionsToReduce;



	//As a user you should be able to ignore everything below here
	//-------------------------------------------------------------

	//Mean working memory
	float* h_oneVec; //A vector filled with ones, to calculate the mean

	float* h_meanVec;
	uint64_t h_meanVecBatchOffset;

	//Covariance matrix working memory
	float* h_covarianceMatrix;
	uint64_t h_covarianceMatrixBatchOffset;


	//Eigenvector/value working memory
	float* h_U;
	uint64_t h_UBatchOffset;

	float* h_S;
	uint64_t h_SBatchOffset;

	float* h_VT;
	uint64_t h_VTBatchOffset;


	float* h_projectedSignalMatrix;
	uint64_t h_projectedSignalBatchOffset;



}RFIMMemoryStructCPU;


RFIMMemoryStructCPU* RFIMMemoryStructCreateCPU(uint64_t h_valuesPerSample, uint64_t h_numberOfSamples, uint64_t h_dimensionToReduce,
		uint64_t h_batchSize);
void RFIMMemoryStructDestroy(RFIMMemoryStructCPU* RFIMStruct);



#endif /* RFIMMEMORYSTRUCTCPU_H_ */