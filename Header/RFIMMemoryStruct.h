/*
 * RFIMMemoryStruct.h
 *
 *  Created on: 22 Mar 2016
 *      Author: vincentvillani
 */

#ifndef RFIMMEMORYSTRUCT_H_
#define RFIMMEMORYSTRUCT_H_

#include <cublas.h>
#include <cusolverDn.h>
#include <stdint.h>

typedef struct RFIMMemoryStruct
{

	//Signal attributes, these need to be set before use
	uint64_t h_valuesPerSample;
	uint64_t h_numberOfSamples;
	uint64_t h_batchSize;
	uint64_t h_threadId;
	uint64_t h_eigenVectorDimensionsToReduce;



	//As a user you should be able to ignore everything below here
	//-------------------------------------------------------------

	//Mean working memory
	float* d_oneVec; //A vector filled with ones, to calculate the mean

	float* d_meanVec;
	uint64_t h_meanVecBatchOffset;

	//Covariance matrix working memory
	float* d_covarianceMatrix;
	uint64_t h_covarianceMatrixBatchOffset;


	//Eigenvector/value working memory
	float* d_U;
	uint64_t h_UBatchOffset;

	float* d_S;
	uint64_t h_SBatchOffset;

	float* d_VT;
	uint64_t h_VTBatchOffset;


	int h_singleEigWorkingSpaceByteSize;
	float* d_eigenWorkingSpace;
	uint64_t h_eigenWorkingSpaceBatchOffset;


	int* d_devInfo;
	int* h_devInfo;
	uint64_t h_devInfoBatchOffset;

	/*
	float** d_projectedSignalMatrix;
	*/

	//Library handles
	cublasHandle_t* cublasHandle;
	cusolverDnHandle_t* cusolverHandle;
	cudaStream_t* h_cudaStreams;
	uint64_t h_cudaStreamsLength;


}RFIMMemoryStruct;


RFIMMemoryStruct* RFIMMemoryStructCreate(uint64_t h_valuesPerSample, uint64_t h_numberOfSamples, uint64_t h_dimensionToReduce,
		uint64_t h_batchSize, uint64_t h_numberOfCUDAStreams);
void RFIMMemoryStructDestroy(RFIMMemoryStruct* RFIMStruct);


#endif /* RFIMMEMORYSTRUCT_H_ */
