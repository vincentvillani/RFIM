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
	uint32_t h_threadId;



	//As a user you should be able to ignore everything below here
	//-------------------------------------------------------------

	//Mean working memory
	float** d_oneVec; //A vector filled with ones, to calculate the mean
	float** d_meanVec;

	//Covariance matrix working memory
	float** d_covarianceMatrix;
	float** h_covarianceMatrixDevicePointers;

	//Eigenvector/value working memory
	float** d_U;
	float** h_UDevicePointers;
	//float** h_UDeviceOffsetPointers; //This is used to remove the eigenvector columns

	float** d_S;
	float** h_SDevicePointers;

	float** d_VT;
	float** h_VTDevicePointers;

	float** d_eigWorkingSpace;
	float** h_eigWorkingSpaceDevicePointers;

	int h_eigWorkingSpaceLength;

	int* d_devInfo;
	int* h_devInfoValues;

	uint64_t h_eigenVectorDimensionsToReduce;

	float** d_projectedSignalMatrix;

	//Library handles
	cublasHandle_t* cublasHandle;
	cusolverDnHandle_t* cusolverHandle;
	cudaStream_t cudaStream;


}RFIMMemoryStruct;


RFIMMemoryStruct* RFIMMemoryStructCreate(uint64_t h_valuesPerSample, uint64_t h_numberOfSamples, uint64_t h_dimensionToReduce, uint64_t h_batchSize, uint32_t threadId);
void RFIMMemoryStructDestroy(RFIMMemoryStruct* RFIMStruct);


#endif /* RFIMMEMORYSTRUCT_H_ */
