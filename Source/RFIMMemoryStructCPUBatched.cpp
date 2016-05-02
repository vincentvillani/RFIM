#include "../Header/RFIMMemoryStructCPUBatched.h"

#include <stdlib.h>

RFIMMemoryStructCPUBatched* RFIMMemoryStructCreateCPUBatched(uint64_t h_valuesPerSample, uint64_t h_numberOfSamples, uint64_t h_dimensionToReduce,
		uint64_t h_batchSize)
{
	RFIMMemoryStructCPUBatched* result = (RFIMMemoryStructCPUBatched*)malloc(sizeof(RFIMMemoryStructCPUBatched));



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

		//Setup the batch pointers
		//All pointers point to the same point in memory
		result->h_oneVecBatched = (float**)malloc(sizeof(float*) * h_batchSize);
		for(uint64_t i = 0; i < h_batchSize; ++i)
		{
			result->h_oneVecBatched[i] = result->h_oneVec;
		}


		//Mean vec
		uint64_t meanVecLength = h_valuesPerSample * h_batchSize;
		uint64_t meanVecByteSize = sizeof(float) * meanVecLength;

		result->h_meanVec = (float*)malloc(meanVecByteSize);
		result->h_meanVecBatchOffset = h_valuesPerSample;

		//Setup the batched pointers
		result->h_oneVecBatched = (float**)malloc(sizeof(float*) * h_batchSize);
		for(uint64_t i = 0; i < h_batchSize; ++i)
		{
			result->h_oneVecBatched[i] = result->h_meanVec + (i * result->h_meanVecBatchOffset);
		}



		//Setup the covariance matrix
		//------------------------
		uint64_t covarianceMatrixLength = h_valuesPerSample * h_valuesPerSample * h_batchSize;
		uint64_t covarianceMatrixByteSize = sizeof(float) * covarianceMatrixLength;
		result->h_covarianceMatrixBatchOffset = h_valuesPerSample * h_valuesPerSample;

		result->h_covarianceMatrix = (float*)malloc(covarianceMatrixByteSize);

		//Setup the batched pointers
		result->h_covarianceMatrixBatched = (float**)malloc(sizeof(float*) * h_batchSize);
		for(uint64_t i = 0; i < h_batchSize; ++i)
		{
			result->h_covarianceMatrixBatched[i] = result->h_covarianceMatrix +
					(i * result->h_covarianceMatrixBatchOffset);
		}

		return result;
}



void RFIMMemoryStructDestroy(RFIMMemoryStructCPUBatched* RFIMStruct)
{
	free(RFIMStruct->h_oneVec);
	free(RFIMStruct->h_oneVecBatched);
	free(RFIMStruct->h_meanVec);
	free(RFIMStruct->h_meanVecBatched);

	free(RFIMStruct->h_covarianceMatrix);
	free(RFIMStruct->h_covarianceMatrixBatched);


	free(RFIMStruct);
}
