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
	float** h_covarianceMatrixDevicePointers;

	//Eigenvector/value working memory
	float** d_U;
	float** h_UDevicePointers;

	float** d_S;
	float** h_SDevicePointers;

	float** d_VT;
	float** h_VTDevicePointers;

	float** d_eigWorkingSpace;
	float** h_eigWorkingSpaceDevicePointers;

	int h_eigWorkingSpaceLength;

	int* d_devInfo;
	//int** h_devInfoDevicePointers; //Space to copy the d_devInfo devicePointers into
	int* h_devInfoValues;

	uint32_t h_eigenVectorDimensionsToReduce;

	float** d_projectedSignalMatrix;

	//Library handles
	cublasHandle_t* cublasHandle;
	cusolverDnHandle_t* cusolverHandle;


}RFIMMemoryStruct;


RFIMMemoryStruct* RFIMMemoryStructCreate(uint32_t h_valuesPerSample, uint32_t h_numberOfSamples, uint32_t h_dimensionToReduce, uint32_t h_batchSize);
void RFIMMemoryStructDestroy(RFIMMemoryStruct* RFIMStruct);


#endif /* RFIMMEMORYSTRUCT_H_ */
