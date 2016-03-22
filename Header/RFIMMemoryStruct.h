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

	//Signals
	//float* d_originalSignal; //The original signal on the device
	//float* d_filteredSignal; //The filtered signal as a result of RFIM

	//Signal attributes, these need to be set before use
	uint32_t h_valuesPerSample;
	uint32_t h_numberOfSamples;




	//As a user you should be able to ignore everything below here
	//-------------------------------------------------------------


	//Mean working memory
	float* d_oneVec; //A vector filled with ones, to calculate the mean
	float* d_meanVec;
	float* d_meanMatrix;

	//Covariance matrix working memory
	float* d_upperTriangularCovarianceMatrix;
	float* d_upperTriangularTransposedMatrix;
	float* d_fullSymmetricCovarianceMatrix;

	//Eigenvector/value working memory
	float* d_U;
	float* d_S;
	float* d_VT;
	float* d_eigWorkingSpace;
	int h_eigWorkingSpaceLength;
	int* d_devInfo;

	float* d_reducedEigenVecMatrix;
	float* d_reducedEigenVecMatrixTranspose;
	float* d_reducedEigenMatrixOuterProduct;
	uint32_t h_eigenVectorDimensionsToReduce;

	//Library handles
	cublasHandle_t* cublasHandle;
	cusolverDnHandle_t* cusolverHandle;


}RFIMMemoryStruct;


RFIMMemoryStruct* RFIMMemoryStructCreate(uint32_t h_valuesPerSample, uint32_t h_numberOfSamples);
void RFIMMemoryStructDestroy(RFIMMemoryStruct* RFIMStruct);


#endif /* RFIMMEMORYSTRUCT_H_ */
