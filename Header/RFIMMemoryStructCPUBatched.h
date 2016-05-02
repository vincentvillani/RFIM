#ifndef RFIM_MEMORY_STRUCT_CPU_BATCHED
#define RFIM_MEMORY_STRUCT_CPU_BATCHED


#include <stdint.h>

typedef struct RFIMMemoryStructCPUBatched
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
	float** h_oneVecBatched;

	float* h_meanVec;
	float** h_meanVecBatched;
	uint64_t h_meanVecBatchOffset;

	//Covariance matrix working memory
	float* h_covarianceMatrix;
	float** h_covarianceMatrixBatched;
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



}RFIMMemoryStructCPUBatched;


RFIMMemoryStructCPUBatched* RFIMMemoryStructCreateCPUBatched(uint64_t h_valuesPerSample, uint64_t h_numberOfSamples, uint64_t h_dimensionToReduce,
		uint64_t h_batchSize);
void RFIMMemoryStructDestroy(RFIMMemoryStructCPUBatched* RFIMStruct);

#endif
