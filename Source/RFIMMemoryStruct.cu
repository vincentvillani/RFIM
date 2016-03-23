
#include "../Header/RFIMMemoryStruct.h"

#include "../Header/CudaUtilityFunctions.h"

#include <stdio.h>


RFIMMemoryStruct* RFIMMemoryStructCreate(uint32_t h_valuesPerSample, uint32_t h_numberOfSamples)
{
	RFIMMemoryStruct* result = (RFIMMemoryStruct*)malloc(sizeof(RFIMMemoryStruct));

	result->cublasHandle = (cublasHandle_t*)malloc(sizeof(cublasHandle_t));
	result->cusolverHandle = (cusolverDnHandle_t*)malloc(sizeof(cusolverDnHandle_t));

	cublasStatus_t cublasStatus;
	cusolverStatus_t cusolverStatus;

	//Create the contexts for each library
	cublasStatus = cublasCreate_v2( result->cublasHandle );
	cusolverStatus = cusolverDnCreate( result->cusolverHandle );

	//Check the contexts started up ok
	if(cublasStatus != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "RFIMMemory::RFIMMemory(): Error starting cublas context\n");
		//exit(1);
	}

	if(cusolverStatus != CUSOLVER_STATUS_SUCCESS)
	{
		fprintf(stderr, "RFIMMemory::RFIMMemory(): Error starting cusolver context\n");
		//exit(1);
	}


	//Set signal attributes
	result->h_valuesPerSample = h_valuesPerSample;
	result->h_numberOfSamples = h_numberOfSamples;

	//Allocate memory for the filtered signal
	//Set the original signal to NULL originally
	//result->d_originalSignal = NULL; //The RFIM routine will exit if this is set to NULL upon entry
	//cudaMalloc(&(result->d_filteredSignal), sizeof(float) * h_valuesPerSample * h_numberOfSamples);


	//Setup the mean working memory
	float* h_oneVec = (float*)malloc(sizeof(float) * h_valuesPerSample);

	//Fill the one vec with ones
	for(uint32_t i = 0; i < h_valuesPerSample; ++i)
	{
		h_oneVec[i] = 1;
	}


	cudaMalloc(&result->d_oneVec, sizeof(float) * h_valuesPerSample);
	cudaMemset(result->d_oneVec, 0, sizeof(float) * h_valuesPerSample);

	//Copy the one vec to the device
	CudaUtility_CopySignalToDevice(h_oneVec, &result->d_oneVec,  sizeof(float) * h_valuesPerSample);

	//Free the host memory, don't need it anymore
	free(h_oneVec);



	//Allocate working space for the other mean
	cudaMalloc(&(result->d_meanVec), sizeof(float) * h_valuesPerSample);
	cudaMemset(result->d_meanVec, 0, sizeof(float) * h_valuesPerSample);
	//cudaMalloc(&(result->d_meanMatrix), sizeof(float) * h_valuesPerSample * h_valuesPerSample);


	//Allocate space for the covariance matrix
	cudaMalloc(&(result->d_upperTriangularCovarianceMatrix), sizeof(float) * h_valuesPerSample * h_valuesPerSample);
	cudaMemset(result->d_upperTriangularCovarianceMatrix, 0, sizeof(float) * h_valuesPerSample * h_valuesPerSample);

	cudaMalloc(&(result->d_upperTriangularTransposedMatrix), sizeof(float) * h_valuesPerSample * h_valuesPerSample);
	cudaMemset(result->d_upperTriangularTransposedMatrix, 0, sizeof(float) * h_valuesPerSample * h_valuesPerSample);

	cudaMalloc(&(result->d_fullSymmetricCovarianceMatrix), sizeof(float) * h_valuesPerSample * h_valuesPerSample);
	cudaMemset(result->d_fullSymmetricCovarianceMatrix, 0, sizeof(float) * h_valuesPerSample * h_valuesPerSample);



	//Allocate working space for the eigenvector/value solver
	cudaMalloc(&(result->d_U), sizeof(float) * h_valuesPerSample * h_valuesPerSample);
	cudaMemset(result->d_U, 0, sizeof(float) * h_valuesPerSample * h_valuesPerSample);


	cudaMalloc(&(result->d_S), sizeof(float) * h_valuesPerSample);
	cudaMemset(result->d_S, 0, sizeof(float) * h_valuesPerSample);


	cudaMalloc(&(result->d_VT), sizeof(float) * h_valuesPerSample * h_valuesPerSample);
	cudaMemset(result->d_VT, 0, sizeof(float) * h_valuesPerSample * h_valuesPerSample);

	cudaMalloc(&(result->d_devInfo), sizeof(int));
	cudaMemset(result->d_devInfo, 0, sizeof(int));


	//Ask cusolver for the needed buffer size
	result->h_eigWorkingSpaceLength = 0;

	cusolverStatus = cusolverDnSgesvd_bufferSize(*result->cusolverHandle, h_valuesPerSample, h_valuesPerSample, &(result->h_eigWorkingSpaceLength));

	//Check if it went well
	if(cusolverStatus != CUSOLVER_STATUS_SUCCESS)
	{
		fprintf(stderr, "RFIMMemory::RFIMMemory(): Error finding eigenvalue working buffer size\n");
		//exit(1);
	}

	//Allocate memory for it
	cudaMalloc( &(result->d_eigWorkingSpace), result->h_eigWorkingSpaceLength);
	cudaMemset(result->d_eigWorkingSpace, 0, result->h_eigWorkingSpaceLength);


	//Eigenvectors dimensions to reduce, chosen arbitrarily for now
	//TODO: Come back to this. This will probably change
	result->h_eigenVectorDimensionsToReduce = 2;

	//Allocate memory for the reduced Eigenvector matrix and it's transpose
	cudaMalloc(&(result->d_reducedEigenVecMatrix), sizeof(float) * h_valuesPerSample *
			(h_valuesPerSample - result->h_eigenVectorDimensionsToReduce));
	cudaMemset(result->d_reducedEigenVecMatrix, 0, sizeof(float) * h_valuesPerSample *
			(h_valuesPerSample - result->h_eigenVectorDimensionsToReduce));



	cudaMalloc(&(result->d_reducedEigenVecMatrixTranspose), sizeof(float) * h_valuesPerSample *
			(h_valuesPerSample - result->h_eigenVectorDimensionsToReduce));
	cudaMemset(result->d_reducedEigenVecMatrixTranspose, 0, sizeof(float) * h_valuesPerSample *
			(h_valuesPerSample - result->h_eigenVectorDimensionsToReduce));

	//Outer product returns the matrix back to it's original dimensionality
	cudaMalloc(&(result->d_reducedEigenMatrixOuterProduct), sizeof(float) * h_valuesPerSample * h_valuesPerSample);
	cudaMemset(result->d_reducedEigenMatrixOuterProduct, 0, sizeof(float) * h_valuesPerSample *
				(h_valuesPerSample - result->h_eigenVectorDimensionsToReduce));


	return result;
}



void RFIMMemoryStructDestroy(RFIMMemoryStruct* RFIMStruct)
{
	//Destroy the cuda library contexts
	cublasDestroy_v2(*RFIMStruct->cublasHandle);
	cusolverDnDestroy(*RFIMStruct->cusolverHandle);

	free(RFIMStruct->cublasHandle);
	free(RFIMStruct->cusolverHandle);

	//Deallocate the mean working memory
	cudaFree(RFIMStruct->d_oneVec);
	cudaFree(RFIMStruct->d_meanVec);
	//cudaFree(RFIMStruct->d_meanMatrix);

	//Deallocate covariance working memory
	cudaFree(RFIMStruct->d_upperTriangularCovarianceMatrix);
	cudaFree(RFIMStruct->d_upperTriangularTransposedMatrix);
	cudaFree(RFIMStruct->d_fullSymmetricCovarianceMatrix);

	//Deallocate eigenvector/value working memory
	cudaFree(RFIMStruct->d_U);
	cudaFree(RFIMStruct->d_S);
	cudaFree(RFIMStruct->d_VT);
	cudaFree(RFIMStruct->d_devInfo);
	cudaFree(RFIMStruct->d_eigWorkingSpace);

	cudaFree(RFIMStruct->d_reducedEigenVecMatrix);
	cudaFree(RFIMStruct->d_reducedEigenVecMatrixTranspose);
	cudaFree(RFIMStruct->d_reducedEigenMatrixOuterProduct);



	//Deallocate the struct memory on the host
	free(RFIMStruct);

}


