/*
 * RFIMMemoryStructCPU.cpp
 *
 *  Created on: 28 Apr 2016
 *      Author: vincentvillani
 */

#include "../Header/RFIMMemoryStructCPU.h"



#include <mkl.h>

RFIMMemoryStructCPU* RFIMMemoryStructCreateCPU(uint64_t h_valuesPerSample, uint64_t h_numberOfSamples, uint64_t h_dimensionToReduce,
		uint64_t h_batchSize)
{

	RFIMMemoryStructCPU* result = (RFIMMemoryStructCPU*)malloc(sizeof(RFIMMemoryStructCPU));



	//Set signal attributes
	//------------------------
	result->h_valuesPerSample = h_valuesPerSample;
	result->h_numberOfSamples = h_numberOfSamples;
	result->h_eigenVectorDimensionsToReduce = h_dimensionToReduce;
	result->h_batchSize = h_batchSize;





	//Setup the one vec, we use the same memory over and over again, it should never change
	//------------------------
	uint64_t oneVecLength = h_numberOfSamples;
	uint64_t oneVecByteSize = sizeof(float) * oneVecLength;

	result->h_oneVec = (float*)malloc(oneVecByteSize);

	//Fill the one vec with ones
	for(uint64_t i = 0; i < oneVecLength; ++i)
	{
		result->h_oneVec[i] = 1;
	}


	//Mean vec
	uint64_t meanVecLength = h_valuesPerSample * h_batchSize;
	uint64_t meanVecByteSize = sizeof(float) * meanVecLength;

	result->h_meanVec = (float*)malloc(meanVecByteSize);
	result->h_meanVecBatchOffset = h_valuesPerSample;



	//Setup the covariance matrix
	//------------------------
	uint64_t covarianceMatrixLength = h_valuesPerSample * h_valuesPerSample * h_batchSize;
	uint64_t covarianceMatrixByteSize = sizeof(float) * covarianceMatrixLength;
	result->h_covarianceMatrixBatchOffset = h_valuesPerSample * h_valuesPerSample;

	result->h_covarianceMatrix = (float*)malloc(covarianceMatrixByteSize);



	//Eigenvector stuff
	//------------------------
	uint64_t singleULength = h_valuesPerSample * h_valuesPerSample;
	uint64_t ULength = singleULength * h_batchSize;
	uint64_t UByteSize = sizeof(float) * ULength;

	result->h_U = (float*)malloc(UByteSize);
	result->h_VT = (float*)malloc(UByteSize); //VT is the same size as U

	result->h_UBatchOffset = singleULength;
	result->h_VTBatchOffset = singleULength;


	//S
	uint64_t singleSLength = h_valuesPerSample;
	uint64_t SLength = h_valuesPerSample * h_batchSize;
	uint64_t SByteLength = sizeof(float) * SLength;

	result->h_S = (float*)malloc(SByteLength);
	result->h_SBatchOffset = singleSLength;


	//Projected signal
	//------------------------
	uint64_t projectedSignalSingleLength = h_valuesPerSample * h_numberOfSamples;
	uint64_t projectedSignalLength = projectedSignalSingleLength * h_batchSize;
	uint64_t projectedSignalByteSize = sizeof(float) * projectedSignalLength;
	result->h_projectedSignalBatchOffset = projectedSignalSingleLength;

	result->h_projectedSignalMatrix = (float*)malloc(projectedSignalByteSize);

	return result;
}



void RFIMMemoryStructDestroy(RFIMMemoryStructCPU* RFIMStruct)
{
	free(RFIMStruct->h_oneVec);
	free(RFIMStruct->h_meanVec);

	free(RFIMStruct->h_covarianceMatrix);

	free(RFIMStruct->h_U);
	free(RFIMStruct->h_VT);
	free(RFIMStruct->h_S);

	free(RFIMStruct->h_projectedSignalMatrix);

	free(RFIMStruct);
}
