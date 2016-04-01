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
	uint32_t h_valuesPerSample;
	uint32_t h_numberOfSamples;
	uint32_t h_batchSize;




	//As a user you should be able to ignore everything below here
	//-------------------------------------------------------------


	//Mean working memory
	float** d_oneVec; //A vector filled with ones, to calculate the mean
	float** d_meanVec;

	//Covariance matrix working memory
	float** d_covarianceMatrix;

	//Eigenvector/value working memory
	float** d_U;
	float** d_S;
	float** d_VT;
	float** d_eigWorkingSpace;
	int h_eigWorkingSpaceLength;
	int** d_devInfo;

	uint32_t h_eigenVectorDimensionsToReduce;

	float** d_projectedSignalMatrix;

	//Library handles
	cublasHandle_t* cublasHandle;
	cusolverDnHandle_t* cusolverHandle;


}RFIMMemoryStruct;


RFIMMemoryStruct* RFIMMemoryStructCreate(uint32_t h_valuesPerSample, uint32_t h_numberOfSamples, uint32_t h_dimensionToReduce, uint32_t h_batchSize);
void RFIMMemoryStructDestroy(RFIMMemoryStruct* RFIMStruct);


#endif /* RFIMMEMORYSTRUCT_H_ */
