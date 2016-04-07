/*
 * UnitTests.cu
 *
 *  Created on: 10/03/2016
 *      Author: vincentvillani
 */


#include "../Header/UnitTests.h"

#include "../Header/CudaMacros.h"
#include "../Header/CudaUtilityFunctions.h"
#include "../Header/Kernels.h"
#include "../Header/RFIMHelperFunctions.h"
#include "../Header/UtilityFunctions.h"
#include "../Header/RFIMMemoryStruct.h"

#include <cublas.h>

#include <assert.h>
#include <cmath>
#include <string>


//Production tests
void MeanCublasProduction();
void CovarianceCublasProduction();
//void EigendecompProduction();
//void FilteringProduction();
//void TransposeProduction();
//void GraphProduction();





//-------------------------------------

//Production
//-------------------------------------

void MeanCublasProduction()
{


	uint32_t valuesPerSample = 3;
	uint32_t sampleNum = 2;
	uint32_t batchSize = 5;
	uint32_t numberOfCudaStreams = 16;

	RFIMMemoryStruct* RFIMStruct = RFIMMemoryStructCreate(valuesPerSample, sampleNum, 0, batchSize, numberOfCudaStreams);

	uint64_t signalLength = valuesPerSample * sampleNum * batchSize;
	uint64_t signalByteSize = sizeof(float) * signalLength;

	float* h_signal;
	cudaMallocHost(&h_signal, signalByteSize);

	//Set the host signal
	for(uint32_t i = 0; i < signalLength; ++i)
	{
		h_signal[i] = i + 1;
	}


	//Copy the signal over to the host
	float* d_signal;
	cudaMalloc(&d_signal, signalByteSize);
	cudaMemcpy(d_signal, h_signal, signalByteSize, cudaMemcpyHostToDevice);

	//cudaDeviceSynchronize();

	//Compute the mean matrix
	Device_CalculateMeanMatrices(RFIMStruct, d_signal);


	//Allocate space to store the result
	uint64_t meanMatricesLength = valuesPerSample * valuesPerSample * batchSize;
	uint64_t meanMatricesByteSize = sizeof(float) * meanMatricesLength;

	uint64_t singleMeanMatrixByteSize = sizeof(float) * valuesPerSample * valuesPerSample;
	uint64_t meanMatrixOffset = valuesPerSample * valuesPerSample;


	float* h_meanMatrices;
	cudaMallocHost(&h_meanMatrices, meanMatricesByteSize);

	uint32_t cudaStreamIterator = 0;

	//Copy the memory back to the host asyncly
	for(uint64_t i = 0; i < batchSize; ++i)
	{
		cudaMemcpyAsync(h_meanMatrices + (i * meanMatrixOffset), RFIMStruct->d_covarianceMatrix + (i * meanMatrixOffset),
				singleMeanMatrixByteSize, cudaMemcpyDeviceToHost, RFIMStruct->h_cudaStreams[cudaStreamIterator]);

		cudaStreamIterator += 1;
		if(cudaStreamIterator >= RFIMStruct->h_cudaStreamsLength)
		{
			cudaStreamIterator = 0;
		}
	}

	//Wait for all operations to complete
	cudaError_t cudaError = cudaDeviceSynchronize();

	if(cudaError != cudaSuccess)
	{
		fprintf(stderr, "Something has gone wrong in MeanCublasProduction unit test\n");
		exit(1);
	}


	//Print the results
	for(uint64_t i = 0; i < meanMatricesLength; ++i)
	{
		//printf("%llu: %f\n", i, h_meanMatrices[i]);
	}


	//Free all memory
	cudaFreeHost(h_signal);
	cudaFreeHost(h_meanMatrices);

	cudaFree(d_signal);

	RFIMMemoryStructDestroy(RFIMStruct);

	cudaError = cudaDeviceSynchronize();

	if(cudaError != cudaSuccess)
	{
		fprintf(stderr, "Something has gone wrong at the end of MeanCublasProduction unit test\n");
		exit(1);
	}

}




void CovarianceCublasProduction()
{

	uint32_t valuesPerSample = 3;
	uint32_t sampleNum = 2;
	uint32_t batchSize = 5;
	uint32_t numberOfCudaStreams = 16;

	RFIMMemoryStruct* RFIMStruct = RFIMMemoryStructCreate(valuesPerSample, sampleNum, 0, batchSize, numberOfCudaStreams);

	uint64_t signalLength = valuesPerSample * sampleNum * batchSize;
	uint64_t signalByteSize = sizeof(float) * signalLength;

	float* h_signal;
	cudaMallocHost(&h_signal, signalByteSize);

	//Set the host signal
	for(uint32_t i = 0; i < signalLength; ++i)
	{
		h_signal[i] = i + 1;
	}


	//Copy the signal over to the host
	float* d_signal;
	cudaMalloc(&d_signal, signalByteSize);
	cudaMemcpy(d_signal, h_signal, signalByteSize, cudaMemcpyHostToDevice);


	//Compute the covariance matrices
	Device_CalculateCovarianceMatrix(RFIMStruct, d_signal);



	//Copy the covariance matrices back to the host asyncly
	uint64_t covarianceMatricesLength = valuesPerSample * valuesPerSample * batchSize;
	uint64_t covarianceMatricesByteSize = sizeof(float) * covarianceMatricesLength;

	uint64_t meanMatrixOffset = valuesPerSample * valuesPerSample;
	uint64_t singleCovarianceMatrixByteSize = sizeof(float) * meanMatrixOffset;

	float* h_covarianceMatrices;
	cudaMallocHost(&h_covarianceMatrices, covarianceMatricesByteSize);

	uint64_t cudaStreamIterator = 0;

	//These memcpy streams should line up with the computation
	for(uint64_t i = 0; i < batchSize; ++i)
	{
		cudaMemcpyAsync(h_covarianceMatrices + (i * meanMatrixOffset),
				RFIMStruct->d_covarianceMatrix + (i * meanMatrixOffset),
				singleCovarianceMatrixByteSize, cudaMemcpyDeviceToHost, RFIMStruct->h_cudaStreams[cudaStreamIterator]);

		cudaStreamIterator += 1;
		if(cudaStreamIterator >= numberOfCudaStreams)
		{
			cudaStreamIterator = 0;
		}
	}


	//Wait for everything to complete
	cudaDeviceSynchronize();


	//Print the results
	for(uint64_t i = 0; i < covarianceMatricesLength; ++i)
	{
		if(h_covarianceMatrices[i] - 2.25f > 0.0000001f)
		{
			fprintf(stderr, "CovarianceCublasProduction unit test failed!\n");
			exit(1);
		}

		//printf("%llu: %f\n", i, h_covarianceMatrices[i]);
	}


	//Free all memory
	cudaFreeHost(h_signal);
	cudaFreeHost(h_covarianceMatrices);

	cudaFree(d_signal);

	RFIMMemoryStructDestroy(RFIMStruct);

}


/*

void EigendecompProduction()
{
	int valuesPerSample = 2;
	int batchSize = 20;
	int covarianceMatrixByteSize = sizeof(float) * valuesPerSample * valuesPerSample;
	int signalPointersArrayByteSize = sizeof(float*) * batchSize;


	RFIMMemoryStruct* RFIMStruct = RFIMMemoryStructCreate(valuesPerSample, valuesPerSample, 2, batchSize, 0);


	//Create small full covariance matrix
	float* h_fullSymmCovarianceMatrix; // = (float*)malloc( covarianceMatrixByteSize );
	float** h_fullSymmCovarianceMatrixPointersArray; // = (float**)malloc(signalPointersArrayByteSize);
	cudaMallocHost(&h_fullSymmCovarianceMatrix, covarianceMatrixByteSize);
	cudaMallocHost(&h_fullSymmCovarianceMatrixPointersArray, signalPointersArrayByteSize);

	h_fullSymmCovarianceMatrix[0] = 5.0f;
	h_fullSymmCovarianceMatrix[1] = 2.0f;
	h_fullSymmCovarianceMatrix[2] = 2.0f;
	h_fullSymmCovarianceMatrix[3] = 5.0f;

	//Set each pointer to point to h_fullSymmCovarianceMatrix
	for(uint32_t i = 0; i < batchSize; ++i)
	{
		h_fullSymmCovarianceMatrixPointersArray[i] = h_fullSymmCovarianceMatrix;
	}

	//Copy these covariance matrices to the device
	CudaUtility_BatchCopyArraysHostToDevice(RFIMStruct->d_covarianceMatrix, h_fullSymmCovarianceMatrixPointersArray, batchSize, covarianceMatrixByteSize,  &(RFIMStruct->cudaStream));

	Device_EigenvalueSolver(RFIMStruct);

	//Copy the results back
	float** h_SData = CudaUtility_BatchAllocateHostArrays(batchSize, sizeof(float) * valuesPerSample);
	float** h_UData = CudaUtility_BatchAllocateHostArrays(batchSize, sizeof(float) * valuesPerSample * valuesPerSample);

	CudaUtility_BatchCopyArraysDeviceToHost(RFIMStruct->d_S, h_SData, batchSize,  sizeof(float) * valuesPerSample,  &(RFIMStruct->cudaStream));
	CudaUtility_BatchCopyArraysDeviceToHost(RFIMStruct->d_U, h_UData, batchSize,  sizeof(float) * valuesPerSample * valuesPerSample,  &(RFIMStruct->cudaStream));


	float eigenvalueExpectedResults[2];
	eigenvalueExpectedResults[0] = 7.0f;
	eigenvalueExpectedResults[1] = 3.0f;

	float eigenvectorExpectedResults[4];
	eigenvectorExpectedResults[0] = -0.707107f;
	eigenvectorExpectedResults[1] = -0.707107f;
	eigenvectorExpectedResults[2] = -0.707107f;
	eigenvectorExpectedResults[3] = 0.707107f;

	//Check and print the results
	for(uint32_t i = 0; i < batchSize; ++i)
	{
		//eigenvalues
		for(uint32_t j = 0; j < valuesPerSample; ++j)
		{

			if(fabs(eigenvalueExpectedResults[j]) - fabs(h_SData[i][j]) > 0.000001f)
			{
				fprintf(stderr, "EigendecompProduction unit test failed. Eigenvalues are incorrect\n");
				exit(1);
			}


			//printf("Eigenvalue[%u][%u] = %f\n", i, j, h_SData[i][j]);
		}

		//eigenvectors
		for(uint32_t j = 0; j < valuesPerSample * valuesPerSample; ++j)
		{

			if(fabs(eigenvectorExpectedResults[j]) - fabs(h_UData[i][j]) > 0.000001f)
			{
				fprintf(stderr, "EigendecompProduction unit test failed. Eigenvectors are incorrect\n");
				exit(1);
			}


			//printf("Eigenvector[%u][%u] = %f\n", i, j, h_UData[i][j]);
		}

		//printf("\n");
	}


	//Free memory
	cudaFreeHost(h_fullSymmCovarianceMatrix);
	cudaFreeHost(h_fullSymmCovarianceMatrixPointersArray);

	CudaUtility_BatchDeallocateHostArrays(h_SData, batchSize);
	CudaUtility_BatchDeallocateHostArrays(h_UData, batchSize);

	RFIMMemoryStructDestroy(RFIMStruct);


}



//Doesn't actually prove that the filter itself works, just that the math functions are working as you would expected
//By removing 0 dimensions we should get the same signal back
void FilteringProduction()
{
	int valuesPerSample = 2;
	int batchSize = 5;

	int signalByteSize = sizeof(float) * valuesPerSample * valuesPerSample;


	//REDUCE NOTHING! This should give us back the same signal
	RFIMMemoryStruct* RFIM = RFIMMemoryStructCreate(valuesPerSample, valuesPerSample, 0, batchSize, 0);


	//Create small full covariance matrix
	float** h_signalPointers; // = (float**)malloc(sizeof(float*) * batchSize);
	float* h_signal; // = (float*)malloc( signalByteSize );
	cudaMallocHost(&h_signalPointers, sizeof(float*) * batchSize);
	cudaMallocHost(&h_signal, signalByteSize);

	h_signal[0] = 1.0f;
	h_signal[1] = 2.0f;
	h_signal[2] = 7.0f;
	h_signal[3] = -8.0f;

	//Set each pointer to h_signal
	for(uint32_t i = 0; i < batchSize; ++i)
	{
		h_signalPointers[i] = h_signal;
	}


	//Copy signal to the device
	float** d_signalPointers = CudaUtility_BatchAllocateDeviceArrays(batchSize, signalByteSize,  &(RFIM->cudaStream));
	CudaUtility_BatchCopyArraysHostToDevice(d_signalPointers, h_signalPointers, batchSize, signalByteSize,  &(RFIM->cudaStream));



	//Calculate the covariance matrix
	Device_CalculateCovarianceMatrix(RFIM, d_signalPointers);

	//Calculate the eigenvectors
	Device_EigenvalueSolver(RFIM);

	//Setup the signal output
	float** d_filteredSignals = CudaUtility_BatchAllocateDeviceArrays(batchSize, signalByteSize,  &(RFIM->cudaStream));



	//Do the projection
	Device_EigenReductionAndFiltering(RFIM, d_signalPointers, d_filteredSignals);


	//Copy the signal back to the host
	float** h_filteredSignals = CudaUtility_BatchAllocateHostArrays(batchSize, signalByteSize);
	CudaUtility_BatchCopyArraysDeviceToHost(d_filteredSignals, h_filteredSignals, batchSize, signalByteSize,  &(RFIM->cudaStream));

	bool failed = false;


	for(uint32_t i = 0; i < batchSize; ++i)
	{
		//Make sure we got the same signal back
		for(uint32_t j = 0; j < valuesPerSample * valuesPerSample; ++j)
		{
			//print the signal
			//printf("Orig[%u][%u]: %f, filt[%u][%u]: %f\n", i, j, h_signalPointers[i][j], i, j, h_filteredSignals[i][j]);

			if(fabs(h_signalPointers[i][j]) - fabs(h_filteredSignals[i][j]) > 0.0000001f)
			{
				failed = true;
			}
		}

		//printf("\n");
	}

	if(failed)
	{
		fprintf(stderr, "FilteringProduction: Unit test failed!\n");
		exit(1);
	}


	//Free all memory
	cudaFreeHost(h_signal);
	cudaFreeHost(h_signalPointers);

	CudaUtility_BatchDeallocateDeviceArrays(d_signalPointers, batchSize,  &(RFIM->cudaStream));
	CudaUtility_BatchDeallocateDeviceArrays(d_filteredSignals, batchSize,  &(RFIM->cudaStream));
	CudaUtility_BatchDeallocateHostArrays(h_filteredSignals, batchSize);

	RFIMMemoryStructDestroy(RFIM);

}

*/




void RunAllUnitTests()
{
	MeanCublasProduction();
	CovarianceCublasProduction();
	//EigendecompProduction();
	//FilteringProduction();

	printf("All tests passed!\n");

}

