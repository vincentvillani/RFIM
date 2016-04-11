/*
 * RFIMMemoryStructComplex.h
 *
 *  Created on: 11 Apr 2016
 *      Author: vincentvillani
 */

#ifndef RFIMMEMORYSTRUCTCOMPLEX_H_
#define RFIMMEMORYSTRUCTCOMPLEX_H_


#include <stdint.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

typedef struct RFIMMemoryStructComplex
{

	//Signal attributes, these need to be set before use
	uint64_t h_valuesPerSample;
	uint64_t h_numberOfSamples;
	uint64_t h_batchSize;
	uint64_t h_eigenVectorDimensionsToReduce;



	//As a user you should be able to ignore everything below here
	//-------------------------------------------------------------

	//Mean working memory
	cuComplex* d_oneVec; //A vector filled with ones, to calculate the mean

	cuComplex* d_meanVec;
	uint64_t h_meanVecBatchOffset;

	//Covariance matrix working memory
	cuComplex* d_covarianceMatrix;
	uint64_t h_covarianceMatrixBatchOffset;


	//Eigenvector/value working memory
	cuComplex* d_U;
	uint64_t h_UBatchOffset;

	float* d_S;
	uint64_t h_SBatchOffset;

	cuComplex* d_VT;
	uint64_t h_VTBatchOffset;


	int h_singleEigWorkingSpaceByteSize;
	cuComplex* d_eigenWorkingSpace;
	uint64_t h_eigenWorkingSpaceBatchOffset;

	float* h_rWork;
	uint64_t h_rWorkBatchOffset;


	int* d_devInfo;
	int* h_devInfo;
	uint64_t h_devInfoBatchOffset;


	cuComplex* d_projectedSignalMatrix;
	uint64_t h_projectedSignalBatchOffset;


	//Library handles
	cublasHandle_t* cublasHandle;
	cusolverDnHandle_t* cusolverHandle;
	cudaStream_t* h_cudaStreams;
	uint64_t h_cudaStreamsLength;


}RFIMMemoryStructComplex;


RFIMMemoryStructComplex* RFIMMemoryStructComplexCreate(uint64_t h_valuesPerSample, uint64_t h_numberOfSamples, uint64_t h_dimensionToReduce,
		uint64_t h_batchSize, uint64_t h_numberOfCUDAStreams);
void RFIMMemoryStructComplexDestroy(RFIMMemoryStructComplex* RFIMStruct);



#endif /* RFIMMEMORYSTRUCTCOMPLEX_H_ */
