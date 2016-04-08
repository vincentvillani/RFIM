/*
 * RFIM.cu
 *
 *  Created on: 17/03/2016
 *      Author: vincentvillani
 */

#include "../Header/RFIM.h"

#include "../Header/CudaUtilityFunctions.h"
#include "../Header/UtilityFunctions.h"
#include "../Header/RFIMHelperFunctions.h"

#include <stdio.h>

void RFIMRoutine(RFIMMemoryStruct* RFIMStruct, float* d_columnMajorSignalMatrices, float* d_columnMajorFilteredSignalMatrices)
{


	//If we reduce everything, we will have nothing left...
	if(RFIMStruct->h_eigenVectorDimensionsToReduce >= RFIMStruct->h_valuesPerSample)
	{
		fprintf(stderr, "RFIMStruct->h_eigenVectorDimensionsToReduce >= RFIMStruct->h_valuesPerSample\n");
		exit(1);
	}

	//Calculate covariance matrix for this signal
	Device_CalculateCovarianceMatrix(RFIMStruct, d_columnMajorSignalMatrices);


	//TODO: Debug!
	cudaDeviceSynchronize();

	//Write the matrix to a file
	float* h_covarMatrix;
	uint64_t matrixByteSize = sizeof(float) * RFIMStruct->h_valuesPerSample *  RFIMStruct->h_valuesPerSample;

	cudaMallocHost(&h_covarMatrix, matrixByteSize);
	cudaMemcpy(h_covarMatrix, RFIMStruct->d_covarianceMatrix, matrixByteSize, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	Utility_WriteSignalMatrixToFile("signalCovarianceMatrix.txt", h_covarMatrix, RFIMStruct->h_valuesPerSample,  RFIMStruct->h_valuesPerSample);

	cudaDeviceSynchronize();
	//exit(1);
	//-----------------


	//Calculate the eigenvectors/values
	Device_EigenvalueSolver(RFIMStruct);


	//TODO: Debug!
	cudaDeviceSynchronize();

	//Write the matrix to a file
	float* h_S;
	float* h_U;
	uint64_t SByteSize = sizeof(float) * RFIMStruct->h_valuesPerSample;
	uint64_t UByteSize = sizeof(float) * RFIMStruct->h_valuesPerSample * RFIMStruct->h_valuesPerSample;

	cudaMallocHost(&h_S, SByteSize);
	cudaMallocHost(&h_U, UByteSize);
	cudaMemcpy(h_S, RFIMStruct->d_S, SByteSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_U, RFIMStruct->d_U, UByteSize, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	Utility_WriteSignalMatrixToFile("eigenvalues.txt", h_S, RFIMStruct->h_valuesPerSample,  1);
	Utility_WriteSignalMatrixToFile("eigenvectors.txt", h_U, RFIMStruct->h_valuesPerSample,  RFIMStruct->h_valuesPerSample);

	cudaDeviceSynchronize();

	exit(1);



	//Project the signal against the reduced eigenvector matrix and back again to the original dimensions
	Device_EigenReductionAndFiltering(RFIMStruct, d_columnMajorSignalMatrices, d_columnMajorFilteredSignalMatrices);


	//Make sure all streams we used are done computing before we leave here
	//This is done to ensure some when these streams are used again, we don't override memory other streams may need
	//(some streams may overtake others and be working in a whole different RFIMRoutine iteration and overwrite needed memory)
	cudaError_t cudaError;
	for(uint64_t i = 0; i < RFIMStruct->h_cudaStreamsLength; ++i)
	{
		cudaError = cudaStreamSynchronize(RFIMStruct->h_cudaStreams[i]);

		if(cudaError != cudaSuccess)
		{
			fprintf(stderr, "RFIMRoutine: Something went wrong along the way...\n");
			exit(1);
		}
	}

}

