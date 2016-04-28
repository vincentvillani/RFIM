/*
 * UnitTests.cu
 *
 *  Created on: 10/03/2016
 *      Author: vincentvillani
 */


#include "../Header/UnitTests.h"

#include "../Header/CudaMacros.h"
#include "../Header/Kernels.h"
#include "../Header/RFIMHelperFunctions.h"
#include "../Header/UtilityFunctions.h"
#include "../Header/RFIMMemoryStruct.h"
#include "../Header/RFIMMemoryStructComplex.h"
#include "../Header/RFIMMemoryStructCPU.h"
#include "../Header/RFIM.h"
#include "../Header/CudaUtilityFunctions.h"

#include <cublas_v2.h>

#include <assert.h>
#include <cmath>
#include <string>
#include <thread>
#include <vector>


//Production tests
void MeanCublasProduction();
void MeanCublasBatchedProduction();
void MeanCublasComplexProduction();
void MeanCublasProductionCPU();

void CovarianceCublasProduction();
void CovarianceCublasBatchedProduction();
void CovarianceCublasComplexProduction();
void CovarianceHostProduction();

void EigendecompProduction();
void EigendecompBatchedProduction();
void EigendecompComplexProduction();

void FilteringProduction();
void FilteringProductionComplex();

void RoundTripNoReduction();
void RoundTripNoReductionBatched();
void RoundTripNoReductionComplex();

void MemoryLeakTest();
void MemoryLeakTestComplex();


void RFIMTest();


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
		printf("Unbatched[%llu]: %f\n", i, h_meanMatrices[i]);
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



void MeanCublasBatchedProduction()
{
	uint32_t valuesPerSample = 3;
	uint32_t sampleNum = 2;
	uint32_t batchSize = 5;
	uint32_t numberOfCudaStreams = 1;

	RFIMMemoryStructBatched* RFIMStruct = RFIMMemoryStructBatchedCreate(valuesPerSample, sampleNum, 0, batchSize, numberOfCudaStreams);

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

	uint64_t signalBatchOffset = valuesPerSample * sampleNum;

	//Create the batched pointers to the signal
	float** d_signalBatched = CudaUtility_createBatchedDevicePointers(d_signal, signalBatchOffset, batchSize);

	//Compute the mean vector
	Device_CalculateMeanMatricesBatched(RFIMStruct, d_signalBatched);

	//Copy the results to the host and check them
	float* h_meanVec;
	uint64_t meanVecLength = valuesPerSample * batchSize;
	uint64_t meanVecByteSize = meanVecLength * sizeof(float);
	cudaMallocHost(&h_meanVec, meanVecByteSize);
	cudaMemcpy(h_meanVec, RFIMStruct->d_meanVec, meanVecByteSize, cudaMemcpyDeviceToHost);

	//print the results
	for(uint64_t i = 0; i < meanVecLength; ++i)
	{
		//printf("MeanVecBatched %llu: %f\n", i, h_meanVec[i]);
	}



	//Copy the mean matrix over to the device
	float* h_meanMatrix;
	uint64_t meanMatrixLength = valuesPerSample * valuesPerSample * batchSize;
	uint64_t meanMatrixByteSize = sizeof(float) * meanMatrixLength;
	cudaMallocHost(&h_meanMatrix, meanMatrixByteSize);
	cudaMemcpy(h_meanMatrix, RFIMStruct->d_covarianceMatrix, meanMatrixByteSize, cudaMemcpyDeviceToHost);


	for(uint64_t i = 0; i < meanMatrixLength; ++i)
	{
		printf("MeanMatrix[%llu]: %f\n", i, h_meanMatrix[i]);
	}



	//Free everything
	cudaFreeHost(h_signal);
	cudaFreeHost(h_meanVec);
	cudaFreeHost(h_meanMatrix);

	cudaFree(d_signal);
	cudaFree(d_signalBatched);

	RFIMMemoryStructDestroy(RFIMStruct);
}




void MeanCublasComplexProduction()
{
	uint32_t valuesPerSample = 3;
	uint32_t sampleNum = 2;
	uint32_t batchSize = 5;
	uint32_t numberOfCudaStreams = 16;

	RFIMMemoryStructComplex* RFIMStruct = RFIMMemoryStructComplexCreate(valuesPerSample, sampleNum, 0, batchSize, numberOfCudaStreams);

	uint64_t signalLength = valuesPerSample * sampleNum * batchSize;
	uint64_t signalByteSize = sizeof(cuComplex) * signalLength;

	cuComplex* h_signal;
	cudaMallocHost(&h_signal, signalByteSize);

	//Set the host signal
	for(uint32_t i = 0; i < signalLength; ++i)
	{
		h_signal[i] = make_cuComplex(i + 1, i + 1);
	}


	//Copy the signal over to the host
	cuComplex* d_signal;
	cudaMalloc(&d_signal, signalByteSize);
	cudaMemcpy(d_signal, h_signal, signalByteSize, cudaMemcpyHostToDevice);

	//cudaDeviceSynchronize();

	//Compute the mean matrix
	Device_CalculateMeanMatricesComplex(RFIMStruct, d_signal);


	//Allocate space to store the result
	uint64_t meanMatricesLength = valuesPerSample * valuesPerSample * batchSize;
	uint64_t meanMatricesByteSize = sizeof(cuComplex) * meanMatricesLength;

	uint64_t singleMeanMatrixByteSize = sizeof(cuComplex) * valuesPerSample * valuesPerSample;
	uint64_t meanMatrixOffset = valuesPerSample * valuesPerSample;


	cuComplex* h_meanMatrices;
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
		//printf("%llu: real:%f, imag:%f\n", i, h_meanMatrices[i].x, h_meanMatrices[i].y);
	}


	//Free all memory
	cudaFreeHost(h_signal);
	cudaFreeHost(h_meanMatrices);

	cudaFree(d_signal);

	RFIMMemoryStructComplexDestroy(RFIMStruct);

	cudaError = cudaDeviceSynchronize();

	if(cudaError != cudaSuccess)
	{
		fprintf(stderr, "Something has gone wrong at the end of MeanCublasComplexProduction unit test\n");
		exit(1);
	}

}




void MeanCublasProductionCPU()
{
	uint32_t valuesPerSample = 3;
	uint32_t sampleNum = 2;
	uint32_t batchSize = 5;

	RFIMMemoryStructCPU* RFIMStruct = RFIMMemoryStructCreateCPU(valuesPerSample, sampleNum, 0, batchSize);

	uint64_t signalLength = valuesPerSample * sampleNum * batchSize;
	uint64_t signalByteSize = sizeof(float) * signalLength;

	float* h_signal = (float*)malloc(signalByteSize);

	//Set the host signal
	for(uint32_t i = 0; i < signalLength; ++i)
	{
		h_signal[i] = i + 1;
	}


	//Compute the mean vector
	Host_CalculateMeanMatrices(RFIMStruct, h_signal);

	uint64_t meanMatrixLength = valuesPerSample * valuesPerSample * batchSize;

	//print the results
	for(uint64_t i = 0; i < meanMatrixLength; ++i)
	{
		printf("CPU meanVec[%llu]: %f\n", i, RFIMStruct->h_covarianceMatrix[i]);
	}


	//Free everything
	free(h_signal);

	RFIMMemoryStructDestroy(RFIMStruct);
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





void CovarianceCublasBatchedProduction()
{
	uint32_t valuesPerSample = 3;
	uint32_t sampleNum = 2;
	uint32_t batchSize = 5;
	uint32_t numberOfCudaStreams = 1;

	RFIMMemoryStructBatched* RFIMStruct = RFIMMemoryStructBatchedCreate(valuesPerSample, sampleNum, 0, batchSize, numberOfCudaStreams);

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


	//Make batched pointers to the signal
	float** d_signalBatched;
	uint64_t signalOffset = valuesPerSample * sampleNum;
	d_signalBatched = CudaUtility_createBatchedDevicePointers(d_signal, signalOffset, batchSize);


	//Calculate the covariance matrices
	Device_CalculateCovarianceMatrixBatched(RFIMStruct, d_signalBatched);


	//Copy the covariance matrix over to the host
	float* h_covarianceMatrix;
	uint64_t covarianceMatrixLength = valuesPerSample * valuesPerSample * batchSize;
	uint64_t covarianceMatrixByteSize = sizeof(float) * covarianceMatrixLength;
	cudaMallocHost(&h_covarianceMatrix, covarianceMatrixByteSize);
	cudaMemcpy(h_covarianceMatrix, RFIMStruct->d_covarianceMatrix, covarianceMatrixByteSize, cudaMemcpyDeviceToHost);


	//Print the result
	for(uint64_t i = 0; i < covarianceMatrixLength; ++i)
	{
		printf("CovarianceMatrix[%llu]: %f\n", i, h_covarianceMatrix[i]);
	}


	//Free all memory
	cudaFree(d_signal);
	cudaFree(d_signalBatched);

	cudaFreeHost(h_signal);
	cudaFreeHost(h_covarianceMatrix);

	RFIMMemoryStructDestroy(RFIMStruct);

}






void CovarianceCublasComplexProduction()
{
	uint32_t valuesPerSample = 3;
	uint32_t sampleNum = 2;
	uint32_t batchSize = 5;
	uint32_t numberOfCudaStreams = 16;

	RFIMMemoryStructComplex* RFIMStruct = RFIMMemoryStructComplexCreate(valuesPerSample, sampleNum, 0, batchSize, numberOfCudaStreams);

	uint64_t signalLength = valuesPerSample * sampleNum * batchSize;
	uint64_t signalByteSize = sizeof(cuComplex) * signalLength;

	cuComplex* h_signal;
	cudaMallocHost(&h_signal, signalByteSize);

	//Set the host signal
	for(uint32_t i = 0; i < signalLength; ++i)
	{
		h_signal[i] = make_cuComplex(i + 1, i + 1);
	}


	//Copy the signal over to the host
	cuComplex* d_signal;
	cudaMalloc(&d_signal, signalByteSize);
	cudaMemcpy(d_signal, h_signal, signalByteSize, cudaMemcpyHostToDevice);


	//Compute the covariance matrices
	Device_CalculateCovarianceMatrixComplex(RFIMStruct, d_signal);



	//Copy the covariance matrices back to the host asyncly
	uint64_t covarianceMatricesLength = valuesPerSample * valuesPerSample * batchSize;
	uint64_t covarianceMatricesByteSize = sizeof(cuComplex) * covarianceMatricesLength;

	uint64_t meanMatrixOffset = valuesPerSample * valuesPerSample;
	uint64_t singleCovarianceMatrixByteSize = sizeof(cuComplex) * meanMatrixOffset;

	cuComplex* h_covarianceMatrices;
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


		if(h_covarianceMatrices[i].x - 4.5f > 0.0000001f || h_covarianceMatrices[i].y > 0.0000001f)
		{
			fprintf(stderr, "CovarianceCublasProductionComplex unit test failed!\n");
			exit(1);
		}


		//printf("%llu: real:%f, imag:%f\n", i, h_covarianceMatrices[i].x, h_covarianceMatrices[i].y);
	}


	//Free all memory
	cudaFreeHost(h_signal);
	cudaFreeHost(h_covarianceMatrices);

	cudaFree(d_signal);

	RFIMMemoryStructComplexDestroy(RFIMStruct);
}




void CovarianceHostProduction()
{
	uint32_t valuesPerSample = 3;
	uint32_t sampleNum = 2;
	uint32_t batchSize = 5;

	RFIMMemoryStructCPU* RFIMStruct = RFIMMemoryStructCreateCPU(valuesPerSample, sampleNum, 0, batchSize);

	uint64_t signalLength = valuesPerSample * sampleNum * batchSize;
	uint64_t signalByteSize = sizeof(float) * signalLength;

	float* h_signal = (float*)malloc(signalByteSize);

	//Set the host signal
	for(uint32_t i = 0; i < signalLength; ++i)
	{
		h_signal[i] = i + 1;
	}


	//calculate the covariance matrix
	Host_CalculateCovarianceMatrix(RFIMStruct, h_signal);


	uint64_t covarianceMatrixLength = valuesPerSample * valuesPerSample * batchSize;

	//print the result
	for(uint64_t i = 0; i < covarianceMatrixLength; ++i)
	{
		printf("CovarianceMatrixHost[%llu]: %f\n", i, RFIMStruct->h_covarianceMatrix[i]);
	}


	//Free everything
	free(h_signal);

	RFIMMemoryStructDestroy(RFIMStruct);

}





void EigendecompProduction()
{

	uint64_t valuesPerSample = 2;
	uint64_t numberOfSamples = 2;
	uint64_t batchSize = 20;
	uint64_t numberOfCudaStreams = 16;
	uint64_t singleCovarianceMatrixLength = valuesPerSample * valuesPerSample;
	uint64_t covarianceMatrixLength = singleCovarianceMatrixLength * batchSize;
	uint64_t covarianceMatrixByteSize = sizeof(float) * covarianceMatrixLength;


	RFIMMemoryStruct* RFIMStruct = RFIMMemoryStructCreate(valuesPerSample, valuesPerSample, numberOfSamples, batchSize, numberOfCudaStreams);


	float* h_covarianceMatrices;
	cudaMallocHost(&h_covarianceMatrices, covarianceMatrixByteSize);


	//Set the matrices
	for(uint64_t i = 0; i < batchSize; ++i)
	{
		float* currentCovarianceMatrix = h_covarianceMatrices + (i * singleCovarianceMatrixLength);

		currentCovarianceMatrix[0] = 5.0f;
		currentCovarianceMatrix[1] = 2.0f;
		currentCovarianceMatrix[2] = 2.0f;
		currentCovarianceMatrix[3] = 5.0f;

	}


	//Copy the matrices over to the host
	cudaMemcpy(RFIMStruct->d_covarianceMatrix, h_covarianceMatrices, covarianceMatrixByteSize, cudaMemcpyHostToDevice);




	//Compute the eigenvectors/values
	Device_EigenvalueSolver(RFIMStruct);



	//copy the values back one by one
	uint64_t singleSLength = valuesPerSample;
	uint64_t singleSByteSize = sizeof(float) * singleSLength;
	uint64_t SLength = singleSLength * batchSize;
	uint64_t SByteSize = sizeof(float) * SLength;

	uint64_t singleULength = valuesPerSample * valuesPerSample;
	uint64_t singleUByteSize = sizeof(float) * singleULength;
	uint64_t ULength = singleULength * batchSize;
	uint64_t UByteSize = sizeof(float) * ULength;


	float* h_S;
	float* h_U;

	cudaMallocHost(&h_S, SByteSize);
	cudaMallocHost(&h_U, UByteSize);


	uint64_t cudaIterator = 0;


	//Streams should match up with the computation of each eigenvector/value
	for(uint64_t i  = 0; i < batchSize; ++i)
	{

		cudaMemcpyAsync(h_S + (i * singleSLength), RFIMStruct->d_S + (i * singleSLength),
				singleSByteSize, cudaMemcpyDeviceToHost,
				RFIMStruct->h_cudaStreams[cudaIterator]);


		cudaMemcpyAsync(h_U + (i * singleULength), RFIMStruct->d_U + (i * singleULength),
				singleUByteSize, cudaMemcpyDeviceToHost,
				RFIMStruct->h_cudaStreams[cudaIterator]);

		cudaIterator += 1;
		if(cudaIterator >= numberOfCudaStreams)
		{
			cudaIterator = 0;
		}
	}

	//Wait for everything to finish
	cudaDeviceSynchronize();



	float eigenvalueExpectedResults[2];
	eigenvalueExpectedResults[0] = 7.0f;
	eigenvalueExpectedResults[1] = 3.0f;

	float eigenvectorExpectedResults[4];
	eigenvectorExpectedResults[0] = -0.707107f;
	eigenvectorExpectedResults[1] = -0.707107f;
	eigenvectorExpectedResults[2] = -0.707107f;
	eigenvectorExpectedResults[3] = 0.707107f;



	//Check the results
	//Eigenvalues
	for(uint64_t i = 0; i < batchSize; ++i)
	{
		float* currentS = h_S + (i * singleSLength);

		bool failed = false;

		if(fabs(currentS[0]) - fabs(eigenvalueExpectedResults[0]) > 0.000001f)
		{
			failed = true;
		}

		if(fabs(currentS[1]) - fabs(eigenvalueExpectedResults[1]) > 0.000001f)
		{
			failed = true;
		}

		/*
		for(uint64_t j = 0; j < 2; ++j)
		{
			printf("Eigenvalue[%llu][%llu] = %f\n", i, j, currentS[j]);
		}
		*/

		if(failed)
		{
			fprintf(stderr, "EigendecompProduction unit test: eigenvalues are not being computed properly!\n");
			exit(1);
		}
	}


	//Check eigenvectors
	for(uint64_t i = 0; i < batchSize; ++i)
	{

		float* currentU = h_U + (i * singleULength);

		bool failed = false;

		if(fabs(currentU[0]) - fabs(eigenvectorExpectedResults[0]) > 0.000001f)
		{
			failed = true;
		}
		if(fabs(currentU[1]) - fabs(eigenvectorExpectedResults[1]) > 0.000001f)
		{
			failed = true;
		}
		if(fabs(currentU[2]) - fabs(eigenvectorExpectedResults[2]) > 0.000001f)
		{
			failed = true;
		}
		if(fabs(currentU[3]) - fabs(eigenvectorExpectedResults[3]) > 0.000001f)
		{
			failed = true;
		}

		/*
		for(uint64_t j = 0; j < 4; ++j)
		{
			printf("Eigenvector[%llu][%llu] = %f\n", i, j, currentU[j]);
		}
		*/

		if(failed)
		{
			fprintf(stderr, "EigendecompProduction unit test: eigenvectors are not being computed properly!\n");
			exit(1);
		}
	}


	//Free all the memory
	cudaFreeHost(h_covarianceMatrices);
	cudaFreeHost(h_S);
	cudaFreeHost(h_U);

	RFIMMemoryStructDestroy(RFIMStruct);
}






void EigendecompBatchedProduction()
{
	uint64_t valuesPerSample = 2;
	uint64_t numberOfSamples = 2;
	uint64_t batchSize = 20;
	uint64_t numberOfCudaStreams = 1;
	uint64_t singleCovarianceMatrixLength = valuesPerSample * valuesPerSample;
	uint64_t covarianceMatrixLength = singleCovarianceMatrixLength * batchSize;
	uint64_t covarianceMatrixByteSize = sizeof(float) * covarianceMatrixLength;


	RFIMMemoryStructBatched* RFIMStruct = RFIMMemoryStructBatchedCreate(valuesPerSample, valuesPerSample,
			numberOfSamples, batchSize, numberOfCudaStreams);


	float* h_covarianceMatrices;
	cudaMallocHost(&h_covarianceMatrices, covarianceMatrixByteSize);


	//Set the matrices
	for(uint64_t i = 0; i < batchSize; ++i)
	{
		float* currentCovarianceMatrix = h_covarianceMatrices + (i * singleCovarianceMatrixLength);

		currentCovarianceMatrix[0] = 5.0f;
		currentCovarianceMatrix[1] = 2.0f;
		currentCovarianceMatrix[2] = 2.0f;
		currentCovarianceMatrix[3] = 5.0f;

	}


	//Copy the matrices over to the host
	cudaMemcpy(RFIMStruct->d_covarianceMatrix, h_covarianceMatrices, covarianceMatrixByteSize, cudaMemcpyHostToDevice);




	//Compute the eigenvectors/values
	Device_EigenvalueSolverBatched(RFIMStruct);


	//Copy the results back to the host
	float* h_S;
	float* h_U;

	uint64_t h_SLength = valuesPerSample * batchSize;
	uint64_t h_SByteSize = sizeof(float) * h_SLength;

	uint64_t h_ULength = valuesPerSample * valuesPerSample * batchSize;
	uint64_t h_UByteSize = sizeof(float) * h_ULength;

	cudaMallocHost(&h_S, h_SByteSize);
	cudaMallocHost(&h_U, h_UByteSize);

	cudaMemcpy(h_S, RFIMStruct->d_S, h_SByteSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_U, RFIMStruct->d_U, h_UByteSize, cudaMemcpyDeviceToHost);


	float eigenvalueExpectedResults[2];
	eigenvalueExpectedResults[0] = 7.0f;
	eigenvalueExpectedResults[1] = 3.0f;

	float eigenvectorExpectedResults[4];
	eigenvectorExpectedResults[0] = -0.707107f;
	eigenvectorExpectedResults[1] = -0.707107f;
	eigenvectorExpectedResults[2] = -0.707107f;
	eigenvectorExpectedResults[3] = 0.707107f;


	uint64_t singleSLength = valuesPerSample;
	uint64_t singleULength = valuesPerSample * valuesPerSample;


	//Check the results
	//Eigenvalues
	for(uint64_t i = 0; i < batchSize; ++i)
	{
		float* currentS = h_S + (i * singleSLength);

		bool failed = false;

		if(fabs(currentS[0]) - fabs(eigenvalueExpectedResults[0]) > 0.000001f)
		{
			failed = true;
		}

		if(fabs(currentS[1]) - fabs(eigenvalueExpectedResults[1]) > 0.000001f)
		{
			failed = true;
		}


		for(uint64_t j = 0; j < 2; ++j)
		{
			printf("Eigenvalue[%llu][%llu] = %f\n", i, j, currentS[j]);
		}


		if(failed)
		{
			fprintf(stderr, "EigendecompProduction unit test: eigenvalues are not being computed properly!\n");
			exit(1);
		}
	}


	//Check eigenvectors
	for(uint64_t i = 0; i < batchSize; ++i)
	{

		float* currentU = h_U + (i * singleULength);

		bool failed = false;

		if(fabs(currentU[0]) - fabs(eigenvectorExpectedResults[0]) > 0.000001f)
		{
			failed = true;
		}
		if(fabs(currentU[1]) - fabs(eigenvectorExpectedResults[1]) > 0.000001f)
		{
			failed = true;
		}
		if(fabs(currentU[2]) - fabs(eigenvectorExpectedResults[2]) > 0.000001f)
		{
			failed = true;
		}
		if(fabs(currentU[3]) - fabs(eigenvectorExpectedResults[3]) > 0.000001f)
		{
			failed = true;
		}


		for(uint64_t j = 0; j < 4; ++j)
		{
			printf("Eigenvector[%llu][%llu] = %f\n", i, j, currentU[j]);
		}


		if(failed)
		{
			fprintf(stderr, "EigendecompProduction unit test: eigenvectors are not being computed properly!\n");
			exit(1);
		}
	}



	//Free all memory
	cudaFreeHost(h_covarianceMatrices);
	cudaFreeHost(h_S);
	cudaFreeHost(h_U);

	RFIMMemoryStructDestroy(RFIMStruct);

}





void EigendecompComplexProduction()
{
	uint64_t valuesPerSample = 2;
	uint64_t numberOfSamples = 3;
	uint64_t batchSize = 22;
	uint64_t numberOfCudaStreams = 16;
	uint64_t singleCovarianceMatrixLength = valuesPerSample * valuesPerSample;
	uint64_t covarianceMatrixLength = singleCovarianceMatrixLength * batchSize;
	uint64_t covarianceMatrixByteSize = sizeof(cuComplex) * covarianceMatrixLength;


	RFIMMemoryStructComplex* RFIMStruct = RFIMMemoryStructComplexCreate(valuesPerSample, numberOfSamples, 2,
			batchSize, numberOfCudaStreams);


	cuComplex* h_covarianceMatrices;
	cudaMallocHost(&h_covarianceMatrices, covarianceMatrixByteSize);


	//Set the matrices
	for(uint64_t i = 0; i < batchSize; ++i)
	{
		cuComplex* currentCovarianceMatrix = h_covarianceMatrices + (i * singleCovarianceMatrixLength);

		currentCovarianceMatrix[0] = make_cuComplex(5.0f, 0);
		currentCovarianceMatrix[1] = make_cuComplex(2.0f, 0);
		currentCovarianceMatrix[2] = make_cuComplex(2.0f, 0);
		currentCovarianceMatrix[3] = make_cuComplex(5.0f, 0);

	}


	//Copy the matrices over to the host
	cudaMemcpy(RFIMStruct->d_covarianceMatrix, h_covarianceMatrices, covarianceMatrixByteSize, cudaMemcpyHostToDevice);




	//Compute the eigenvectors/values
	Device_EigenvalueSolverComplex(RFIMStruct);



	//copy the values back one by one
	uint64_t singleSLength = valuesPerSample;
	uint64_t singleSByteSize = sizeof(float) * singleSLength;
	uint64_t SLength = singleSLength * batchSize;
	uint64_t SByteSize = sizeof(float) * SLength;

	uint64_t singleULength = valuesPerSample * valuesPerSample;
	uint64_t singleUByteSize = sizeof(cuComplex) * singleULength;
	uint64_t ULength = singleULength * batchSize;
	uint64_t UByteSize = sizeof(cuComplex) * ULength;


	float* h_S;
	cuComplex* h_U;

	cudaMallocHost(&h_S, SByteSize);
	cudaMallocHost(&h_U, UByteSize);


	uint64_t cudaIterator = 0;


	//Streams should match up with the computation of each eigenvector/value
	for(uint64_t i  = 0; i < batchSize; ++i)
	{

		cudaMemcpyAsync(h_S + (i * singleSLength), RFIMStruct->d_S + (i * singleSLength),
				singleSByteSize, cudaMemcpyDeviceToHost,
				RFIMStruct->h_cudaStreams[cudaIterator]);


		cudaMemcpyAsync(h_U + (i * singleULength), RFIMStruct->d_U + (i * singleULength),
				singleUByteSize, cudaMemcpyDeviceToHost,
				RFIMStruct->h_cudaStreams[cudaIterator]);

		cudaIterator += 1;
		if(cudaIterator >= numberOfCudaStreams)
		{
			cudaIterator = 0;
		}
	}



	//Wait for everything to finish
	cudaDeviceSynchronize();



	float eigenvalueExpectedResults[2];
	eigenvalueExpectedResults[0] = 7.0f;
	eigenvalueExpectedResults[1] = 3.0f;

	cuComplex eigenvectorExpectedResults[4];
	eigenvectorExpectedResults[0] = make_cuComplex(-0.707107f, 0);
	eigenvectorExpectedResults[1] = make_cuComplex(-0.707107f, 0);
	eigenvectorExpectedResults[2] = make_cuComplex(-0.707107f, 0);
	eigenvectorExpectedResults[3] = make_cuComplex(0.707107f, 0);



	//Check the results
	//Eigenvalues
	for(uint64_t i = 0; i < batchSize; ++i)
	{
		float* currentS = h_S + (i * singleSLength);


		bool failed = false;

		if(fabs(currentS[0]) - fabs(eigenvalueExpectedResults[0]) > 0.000001f)
		{
			failed = true;
		}

		if(fabs(currentS[1]) - fabs(eigenvalueExpectedResults[1]) > 0.000001f)
		{
			failed = true;
		}



		for(uint64_t j = 0; j < 2; ++j)
		{
			//printf("Eigenvalue[%llu][%llu] = %f\n", i, j, currentS[j]);
		}



		if(failed)
		{
			fprintf(stderr, "EigendecompProductionComplex unit test: eigenvalues are not being computed properly!\n");
			exit(1);
		}

	}


	//Check eigenvectors
	for(uint64_t i = 0; i < batchSize; ++i)
	{

		cuComplex* currentU = h_U + (i * singleULength);


		bool failed = false;

		if(fabs(currentU[0].x) - fabs(eigenvectorExpectedResults[0].x) > 0.000001f ||
				fabs(currentU[0].y) - fabs(eigenvectorExpectedResults[0].y) > 0.000001f)
		{
			failed = true;
		}
		if(fabs(currentU[1].x) - fabs(eigenvectorExpectedResults[1].x) > 0.000001f ||
				fabs(currentU[1].y) - fabs(eigenvectorExpectedResults[1].y) > 0.000001f)
		{
			failed = true;
		}
		if(fabs(currentU[2].x) - fabs(eigenvectorExpectedResults[2].x) > 0.000001f ||
				fabs(currentU[2].y) - fabs(eigenvectorExpectedResults[2].y) > 0.000001f)
		{
			failed = true;
		}
		if(fabs(currentU[3].x) - fabs(eigenvectorExpectedResults[3].x) > 0.000001f ||
				fabs(currentU[3].y) - fabs(eigenvectorExpectedResults[3].y) > 0.000001f)
		{
			failed = true;
		}


		for(uint64_t j = 0; j < 4; ++j)
		{
			//printf("Eigenvector[%llu][%llu] = real: %f, imag: %f\n", i, j, currentU[j].x, currentU[j].y);
		}



		if(failed)
		{
			fprintf(stderr, "EigendecompProduction unit test: eigenvectors are not being computed properly!\n");
			exit(1);
		}

	}


	//Free all the memory
	cudaFreeHost(h_covarianceMatrices);
	cudaFreeHost(h_S);
	cudaFreeHost(h_U);

	RFIMMemoryStructComplexDestroy(RFIMStruct);
}




//Doesn't actually prove that the filter itself works, just that the math functions are working as you would expected
//By removing 0 dimensions we should get the same signal back
void FilteringProduction()
{

	uint64_t valuesPerSample = 2;
	uint64_t numberOfSamples = 3; //THIS MAY MAKE THE UNIT TEST FAIL!?
	uint64_t dimensionsToReduce = 0;
	uint64_t batchSize = 20;
	uint64_t numberOfCudaStreams = 16;


	RFIMMemoryStruct* RFIMStruct = RFIMMemoryStructCreate(valuesPerSample, numberOfSamples, dimensionsToReduce, batchSize, numberOfCudaStreams);


	uint64_t singleSignalLength = valuesPerSample * numberOfSamples;
	uint64_t signalLength = singleSignalLength * batchSize;
	uint64_t signalByteSize = sizeof(float) * signalLength;

	float* h_signal;
	cudaMallocHost(&h_signal, signalByteSize);


	//Set the signal
	for(uint64_t i = 0; i < batchSize; ++i)
	{
		float* currentSignal = h_signal + (i * singleSignalLength);

		currentSignal[0] = 1.0f;
		currentSignal[1] = 2.0f;
		currentSignal[2] = 7.0f;
		currentSignal[3] = -8.0f;
	}


	//Copy the signal over to the device
	float* d_signal;
	cudaMalloc(&d_signal, signalByteSize);
	cudaMemcpy(d_signal, h_signal, signalByteSize, cudaMemcpyHostToDevice);


	//Calculate the covarianceMatrices
	Device_CalculateCovarianceMatrix(RFIMStruct, d_signal);


	//Calculate the eigenvectors/values
	Device_EigenvalueSolver(RFIMStruct);



	//Allocate memory for the filtered signal
	float* d_filteredSignal;
	cudaMalloc(&d_filteredSignal, signalByteSize);


	//Do the projection/reprojection
	Device_EigenReductionAndFiltering(RFIMStruct, d_signal, d_filteredSignal);



	//copy the result back to the host, one stream at a time

	float* h_filteredSignal;
	cudaMallocHost(&h_filteredSignal, signalByteSize);

	uint64_t cudaStreamIterator = 0;

	for(uint64_t i = 0; i < batchSize; ++i)
	{
		//Put in the request for the memory to be copied
		cudaMemcpyAsync(h_filteredSignal + (i * singleSignalLength),
				d_filteredSignal + (i * singleSignalLength),
				singleSignalLength * sizeof(float),
				cudaMemcpyDeviceToHost,
				RFIMStruct->h_cudaStreams[cudaStreamIterator]);

		//Iterate the streams
		cudaStreamIterator += 1;
		if(cudaStreamIterator >= RFIMStruct->h_cudaStreamsLength)
		{
			cudaStreamIterator = 0;
		}

	}


	//Wait for everything to be completed
	cudaError_t cudaError = cudaDeviceSynchronize();

	if(cudaError != cudaSuccess)
	{
		fprintf(stderr, "FilteringProduction: Something went wrong!\n");
		exit(1);
	}




	//print/check the results
	for(uint64_t i = 0; i < batchSize; ++i)
	{
		float* currentSignal = h_signal + (i * singleSignalLength);
		float* currentFilteredSignal = h_filteredSignal + (i * singleSignalLength);

		for(uint64_t j = 0; j < 4; ++j)
		{
			//printf("filteredSignal[%llu][%llu]: %f\n", i, j, currentFilteredSignal[j]);

			if(currentSignal[j] - currentFilteredSignal[j] > 0.000001f)
			{
				fprintf(stderr, "FilteringProduction unit test: results are different from expected!\n");
				fprintf(stderr, "Expected %f, Actual: %f\n", currentSignal[j], currentFilteredSignal[j]);
				exit(1);
			}
		}
	}



	//Free all memory
	cudaFreeHost(h_signal);
	cudaFreeHost(h_filteredSignal);

	cudaFree(d_signal);
	cudaFree(d_filteredSignal);

	RFIMMemoryStructDestroy(RFIMStruct);


}





void FilteringProductionComplex()
{
	uint64_t valuesPerSample = 2;
	uint64_t numberOfSamples = 3; //THIS MAY MAKE THE UNIT TEST FAIL!?
	uint64_t dimensionsToReduce = 0;
	uint64_t batchSize = 20;
	uint64_t numberOfCudaStreams = 16;


	RFIMMemoryStructComplex* RFIMStruct = RFIMMemoryStructComplexCreate(valuesPerSample, numberOfSamples, dimensionsToReduce, batchSize, numberOfCudaStreams);


	uint64_t singleSignalLength = valuesPerSample * numberOfSamples;
	uint64_t signalLength = singleSignalLength * batchSize;
	uint64_t signalByteSize = sizeof(cuComplex) * signalLength;

	cuComplex* h_signal;
	cudaMallocHost(&h_signal, signalByteSize);


	//Set the signal
	for(uint64_t i = 0; i < batchSize; ++i)
	{
		cuComplex* currentSignal = h_signal + (i * singleSignalLength);

		currentSignal[0] = make_cuComplex(1.0f, 4.0f);
		currentSignal[1] = make_cuComplex(3.0f, -9.0f);
		currentSignal[2] = make_cuComplex(2.0f, -2.0f);
		currentSignal[3] = make_cuComplex(7.0f, 2.0f);
	}



	//Copy the signal over to the device
	cuComplex* d_signal;
	cudaMalloc(&d_signal, signalByteSize);
	cudaMemcpy(d_signal, h_signal, signalByteSize, cudaMemcpyHostToDevice);


	//Calculate the covarianceMatrices
	Device_CalculateCovarianceMatrixComplex(RFIMStruct, d_signal);


	//Calculate the eigenvectors/values
	Device_EigenvalueSolverComplex(RFIMStruct);



	//Allocate memory for the filtered signal
	cuComplex* d_filteredSignal;
	cudaMalloc(&d_filteredSignal, signalByteSize);


	//Do the projection/reprojection
	Device_EigenReductionAndFilteringComplex(RFIMStruct, d_signal, d_filteredSignal);




	//copy the result back to the host, one stream at a time
	cuComplex* h_filteredSignal;
	cudaMallocHost(&h_filteredSignal, signalByteSize);

	uint64_t cudaStreamIterator = 0;

	for(uint64_t i = 0; i < batchSize; ++i)
	{
		//Put in the request for the memory to be copied
		cudaMemcpyAsync(h_filteredSignal + (i * singleSignalLength),
				d_filteredSignal + (i * singleSignalLength),
				singleSignalLength * sizeof(cuComplex),
				cudaMemcpyDeviceToHost,
				RFIMStruct->h_cudaStreams[cudaStreamIterator]);

		//Iterate the streams
		cudaStreamIterator += 1;
		if(cudaStreamIterator >= RFIMStruct->h_cudaStreamsLength)
		{
			cudaStreamIterator = 0;
		}

	}


	//Wait for everything to be completed
	cudaError_t cudaError = cudaDeviceSynchronize();

	if(cudaError != cudaSuccess)
	{
		fprintf(stderr, "FilteringProduction: Something went wrong!\n");
		exit(1);
	}



	//print/check the results
	for(uint64_t i = 0; i < batchSize; ++i)
	{
		cuComplex* currentSignal = h_signal + (i * singleSignalLength);
		cuComplex* currentFilteredSignal = h_filteredSignal + (i * singleSignalLength);

		for(uint64_t j = 0; j < 4; ++j)
		{
			//printf("filteredSignal[%llu][%llu]: real: %f, imag: %f\n", i, j,
			//		currentFilteredSignal[j].x, currentFilteredSignal[j].y);


			if(currentSignal[j].x - currentFilteredSignal[j].x > 0.000001f ||
					currentSignal[j].y - currentFilteredSignal[j].y > 0.000001f)
			{
				fprintf(stderr, "FilteringProduction unit test: results are different from expected!\n");
				fprintf(stderr, "Expected real: %f, imag: %f\nActual real: %f, imag: %f\n",
						currentSignal[j].x, currentSignal[j].y,
						currentFilteredSignal[j].x, currentFilteredSignal[j].y);
				exit(1);
			}

		}
	}



	//Free all memory
	cudaFreeHost(h_signal);
	cudaFreeHost(h_filteredSignal);

	cudaFree(d_signal);
	cudaFree(d_filteredSignal);

	RFIMMemoryStructComplexDestroy(RFIMStruct);


}





void RoundTripNoReduction()
{

	uint64_t h_valuesPerSample = 26;
	uint64_t h_numberOfSamples = 1024;
	uint64_t h_batchSize = 5;
	uint64_t h_dimensionsToReduce = 0;
	uint64_t h_numberOfCudaStreams = 2;

	RFIMMemoryStruct* RFIM = RFIMMemoryStructCreate(h_valuesPerSample, h_numberOfSamples, h_dimensionsToReduce, h_batchSize, h_numberOfCudaStreams);


	//Start up the RNG
	curandGenerator_t rngGen;

	if( curandCreateGenerator(&rngGen, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS)
	{
		fprintf(stderr, "main: Unable to start cuRand library\n");
		exit(1);
	}

	//Set the RNG seed
	if((curandSetPseudoRandomGeneratorSeed(rngGen, 1)) != CURAND_STATUS_SUCCESS)
	{
		fprintf(stderr, "main: Unable to set the RNG Seed value\n");
		exit(1);
	}

	cudaDeviceSynchronize();

	//Generate a signal
	uint64_t signalByteSize = sizeof(float) *  h_valuesPerSample * h_numberOfSamples * h_batchSize;
	float* d_signal = Device_GenerateWhiteNoiseSignal(&rngGen, h_valuesPerSample, h_numberOfSamples, h_batchSize);
	float* d_filteredSignal;
	cudaMalloc(&d_filteredSignal, signalByteSize);


	//Put it through RFIM Routine
	RFIMRoutine(RFIM, d_signal, d_filteredSignal);



	//Wait for everthing to complete / check errors
	cudaError_t cudaError = cudaDeviceSynchronize();

	if(cudaError != cudaSuccess)
	{
		fprintf(stderr, "RoundTripNoReduction(): Something has gone wrong!\n");
		exit(1);
	}



	//Check everything




	//Covariance matrix
	//Write to a text file and check against python result
	//TODO: Weird stuff is going on here...Ask Willem
	float* h_signal;
	cudaMallocHost(&h_signal, signalByteSize);
	cudaMemcpy(h_signal, d_signal, signalByteSize, cudaMemcpyDeviceToHost);

	Utility_WriteSignalMatrixToFile("RoundTripNoReductionSignal1.txt", h_signal, h_numberOfSamples, h_valuesPerSample);
	Utility_WriteSignalMatrixToFile("RoundTripNoReductionSignal2.txt", h_signal + (h_valuesPerSample * h_numberOfSamples), h_numberOfSamples, h_valuesPerSample);
	Utility_WriteSignalMatrixToFile("RoundTripNoReductionSignal3.txt", h_signal + (2 * h_valuesPerSample * h_numberOfSamples), h_numberOfSamples, h_valuesPerSample);
	Utility_WriteSignalMatrixToFile("RoundTripNoReductionSignal4.txt", h_signal + (3 * h_valuesPerSample * h_numberOfSamples), h_numberOfSamples, h_valuesPerSample);
	Utility_WriteSignalMatrixToFile("RoundTripNoReductionSignal5.txt", h_signal + (4 * h_valuesPerSample * h_numberOfSamples), h_numberOfSamples, h_valuesPerSample);



	float* h_covarianceMatrices;
	uint64_t covarianceByteSize = sizeof(float) * h_valuesPerSample * h_valuesPerSample * h_batchSize;
	cudaMallocHost(&h_covarianceMatrices, covarianceByteSize);
	cudaMemcpy(h_covarianceMatrices, RFIM->d_covarianceMatrix, covarianceByteSize, cudaMemcpyDeviceToHost);

	Utility_WriteSignalMatrixToFile("RoundTripNoReductionCovarianceMatrix1.txt", h_covarianceMatrices, h_valuesPerSample, h_valuesPerSample);
	Utility_WriteSignalMatrixToFile("RoundTripNoReductionCovarianceMatrix2.txt", h_covarianceMatrices + (h_valuesPerSample * h_valuesPerSample), h_valuesPerSample, h_valuesPerSample);
	Utility_WriteSignalMatrixToFile("RoundTripNoReductionCovarianceMatrix3.txt", h_covarianceMatrices + (2 * h_valuesPerSample * h_valuesPerSample), h_valuesPerSample, h_valuesPerSample);
	Utility_WriteSignalMatrixToFile("RoundTripNoReductionCovarianceMatrix4.txt", h_covarianceMatrices + (3 * h_valuesPerSample * h_valuesPerSample), h_valuesPerSample, h_valuesPerSample);
	Utility_WriteSignalMatrixToFile("RoundTripNoReductionCovarianceMatrix5.txt", h_covarianceMatrices + (4 * h_valuesPerSample * h_valuesPerSample), h_valuesPerSample, h_valuesPerSample);



	//Is the end result the same more or less?
	//Copy the end result back to the host
	float* h_filteredSignal;
	cudaMallocHost(&h_filteredSignal,  signalByteSize);
	cudaMemcpy(h_filteredSignal, d_filteredSignal, signalByteSize, cudaMemcpyDeviceToHost);

	uint64_t signalLength = h_valuesPerSample * h_numberOfSamples * h_batchSize;
	for(uint64_t i = 0; i < signalLength; ++i)
	{
		if(fabs(h_filteredSignal[i]) - fabs(h_signal[i]) > 0.00001f)
		{
			fprintf(stderr, "RoundTripNoReduction: Signal is not the same as filtered signal\nSignal[%llu] = %f\nFilteredSignal[%llu] = %f\n",
					i, h_signal[i], i, h_filteredSignal[i]);
			exit(1);
		}
	}



	//Free everything
	cudaFreeHost(h_signal);
	cudaFreeHost(h_covarianceMatrices);
	cudaFreeHost(h_filteredSignal);

	cudaFree(d_signal);
	cudaFree(d_filteredSignal);

	curandDestroyGenerator(rngGen);

	RFIMMemoryStructDestroy(RFIM);

}




void RoundTripNoReductionBatched()
{
	uint64_t h_valuesPerSample = 26;
	uint64_t h_numberOfSamples = 1024;
	uint64_t h_batchSize = 5;
	uint64_t h_dimensionsToReduce = 0;
	uint64_t h_numberOfCudaStreams = 2;

	RFIMMemoryStructBatched* RFIM = RFIMMemoryStructBatchedCreate(h_valuesPerSample, h_numberOfSamples,
			h_dimensionsToReduce, h_batchSize, h_numberOfCudaStreams);


	//Start up the RNG
	curandGenerator_t rngGen;

	if( curandCreateGenerator(&rngGen, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS)
	{
		fprintf(stderr, "main: Unable to start cuRand library\n");
		exit(1);
	}

	//Set the RNG seed
	if((curandSetPseudoRandomGeneratorSeed(rngGen, 1)) != CURAND_STATUS_SUCCESS)
	{
		fprintf(stderr, "main: Unable to set the RNG Seed value\n");
		exit(1);
	}

	cudaDeviceSynchronize();

	//Generate a signal
	uint64_t signalByteSize = sizeof(float) *  h_valuesPerSample * h_numberOfSamples * h_batchSize;
	float* d_signal = Device_GenerateWhiteNoiseSignal(&rngGen, h_valuesPerSample, h_numberOfSamples, h_batchSize);
	float* d_filteredSignal;
	cudaMalloc(&d_filteredSignal, signalByteSize);

	uint64_t singleSignalLength = h_valuesPerSample * h_numberOfSamples;

	float** d_signalBatched = CudaUtility_createBatchedDevicePointers(d_signal, singleSignalLength, h_batchSize);
	float** d_filteredBatched = CudaUtility_createBatchedDevicePointers(d_filteredSignal, singleSignalLength, h_batchSize);


	//Put it through RFIM Routine
	RFIMRoutineBatched(RFIM, d_signalBatched, d_filteredBatched);



	//Wait for everthing to complete / check errors
	cudaError_t cudaError = cudaDeviceSynchronize();

	if(cudaError != cudaSuccess)
	{
		fprintf(stderr, "RoundTripNoReduction(): Something has gone wrong!\n");
		exit(1);
	}



	//Check everything




	//Covariance matrix
	//Write to a text file and check against python result
	//TODO: Weird stuff is going on here...Ask Willem
	float* h_signal;
	cudaMallocHost(&h_signal, signalByteSize);
	cudaMemcpy(h_signal, d_signal, signalByteSize, cudaMemcpyDeviceToHost);

	/*

	Utility_WriteSignalMatrixToFile("RoundTripNoReductionSignal1.txt", h_signal, h_numberOfSamples, h_valuesPerSample);
	Utility_WriteSignalMatrixToFile("RoundTripNoReductionSignal2.txt", h_signal + (h_valuesPerSample * h_numberOfSamples), h_numberOfSamples, h_valuesPerSample);
	Utility_WriteSignalMatrixToFile("RoundTripNoReductionSignal3.txt", h_signal + (2 * h_valuesPerSample * h_numberOfSamples), h_numberOfSamples, h_valuesPerSample);
	Utility_WriteSignalMatrixToFile("RoundTripNoReductionSignal4.txt", h_signal + (3 * h_valuesPerSample * h_numberOfSamples), h_numberOfSamples, h_valuesPerSample);
	Utility_WriteSignalMatrixToFile("RoundTripNoReductionSignal5.txt", h_signal + (4 * h_valuesPerSample * h_numberOfSamples), h_numberOfSamples, h_valuesPerSample);



	float* h_covarianceMatrices;
	uint64_t covarianceByteSize = sizeof(float) * h_valuesPerSample * h_valuesPerSample * h_batchSize;
	cudaMallocHost(&h_covarianceMatrices, covarianceByteSize);
	cudaMemcpy(h_covarianceMatrices, RFIM->d_covarianceMatrix, covarianceByteSize, cudaMemcpyDeviceToHost);

	Utility_WriteSignalMatrixToFile("RoundTripNoReductionCovarianceMatrix1.txt", h_covarianceMatrices, h_valuesPerSample, h_valuesPerSample);
	Utility_WriteSignalMatrixToFile("RoundTripNoReductionCovarianceMatrix2.txt", h_covarianceMatrices + (h_valuesPerSample * h_valuesPerSample), h_valuesPerSample, h_valuesPerSample);
	Utility_WriteSignalMatrixToFile("RoundTripNoReductionCovarianceMatrix3.txt", h_covarianceMatrices + (2 * h_valuesPerSample * h_valuesPerSample), h_valuesPerSample, h_valuesPerSample);
	Utility_WriteSignalMatrixToFile("RoundTripNoReductionCovarianceMatrix4.txt", h_covarianceMatrices + (3 * h_valuesPerSample * h_valuesPerSample), h_valuesPerSample, h_valuesPerSample);
	Utility_WriteSignalMatrixToFile("RoundTripNoReductionCovarianceMatrix5.txt", h_covarianceMatrices + (4 * h_valuesPerSample * h_valuesPerSample), h_valuesPerSample, h_valuesPerSample);

	*/

	//Is the end result the same more or less?
	//Copy the end result back to the host
	float* h_filteredSignal;
	cudaMallocHost(&h_filteredSignal,  signalByteSize);
	cudaMemcpy(h_filteredSignal, d_filteredSignal, signalByteSize, cudaMemcpyDeviceToHost);

	uint64_t signalLength = h_valuesPerSample * h_numberOfSamples * h_batchSize;
	for(uint64_t i = 0; i < signalLength; ++i)
	{
		if(fabs(h_filteredSignal[i]) - fabs(h_signal[i]) > 0.00001f)
		{
			fprintf(stderr, "RoundTripNoReductionBatched: Signal is not the same as filtered signal\nSignal[%llu] = %f\nFilteredSignal[%llu] = %f\n",
					i, h_signal[i], i, h_filteredSignal[i]);
			exit(1);
		}
	}



	//Free everything
	cudaFreeHost(h_signal);
	//cudaFreeHost(h_covarianceMatrices);
	cudaFreeHost(h_filteredSignal);

	cudaFree(d_signal);
	cudaFree(d_signalBatched);
	cudaFree(d_filteredSignal);
	cudaFree(d_filteredBatched);


	curandDestroyGenerator(rngGen);

	RFIMMemoryStructDestroy(RFIM);
}





void RoundTripNoReductionComplex()
{

	uint64_t h_valuesPerSample = 26;
	uint64_t h_numberOfSamples = 1024;
	uint64_t h_batchSize = 5;
	uint64_t h_dimensionsToReduce = 0;
	uint64_t h_numberOfCudaStreams = 2;

	RFIMMemoryStructComplex* RFIM = RFIMMemoryStructComplexCreate(h_valuesPerSample, h_numberOfSamples,
			h_dimensionsToReduce, h_batchSize, h_numberOfCudaStreams);


	//Start up the RNG
	curandGenerator_t rngGen;

	if( curandCreateGenerator(&rngGen, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS)
	{
		fprintf(stderr, "main: Unable to start cuRand library\n");
		exit(1);
	}

	//Set the RNG seed
	if((curandSetPseudoRandomGeneratorSeed(rngGen, 1)) != CURAND_STATUS_SUCCESS)
	{
		fprintf(stderr, "main: Unable to set the RNG Seed value\n");
		exit(1);
	}

	cudaDeviceSynchronize();

	//Generate a signal
	uint64_t signalByteSize = sizeof(cuComplex) *  h_valuesPerSample * h_numberOfSamples * h_batchSize;
	cuComplex* d_signal = Device_GenerateWhiteNoiseSignalComplex(&rngGen, h_valuesPerSample, h_numberOfSamples, h_batchSize, 1);
	cuComplex* d_filteredSignal;
	cudaMalloc(&d_filteredSignal, signalByteSize);


	//Put it through RFIM Routine
	RFIMRoutineComplex(RFIM, d_signal, d_filteredSignal);



	//Wait for everthing to complete / check errors
	cudaError_t cudaError = cudaDeviceSynchronize();

	if(cudaError != cudaSuccess)
	{
		fprintf(stderr, "RoundTripNoReduction(): Something has gone wrong!\n");
		exit(1);
	}



	//Check everything




	//Covariance matrix
	//Write to a text file and check against python result
	//TODO: Weird stuff is going on here...Ask Willem
	cuComplex* h_signal;
	cudaMallocHost(&h_signal, signalByteSize);
	cudaMemcpy(h_signal, d_signal, signalByteSize, cudaMemcpyDeviceToHost);

	/*
	Utility_WriteSignalMatrixToFile("RoundTripNoReductionSignal1.txt", h_signal, h_numberOfSamples, h_valuesPerSample);
	Utility_WriteSignalMatrixToFile("RoundTripNoReductionSignal2.txt", h_signal + (h_valuesPerSample * h_numberOfSamples), h_numberOfSamples, h_valuesPerSample);
	Utility_WriteSignalMatrixToFile("RoundTripNoReductionSignal3.txt", h_signal + (2 * h_valuesPerSample * h_numberOfSamples), h_numberOfSamples, h_valuesPerSample);
	Utility_WriteSignalMatrixToFile("RoundTripNoReductionSignal4.txt", h_signal + (3 * h_valuesPerSample * h_numberOfSamples), h_numberOfSamples, h_valuesPerSample);
	Utility_WriteSignalMatrixToFile("RoundTripNoReductionSignal5.txt", h_signal + (4 * h_valuesPerSample * h_numberOfSamples), h_numberOfSamples, h_valuesPerSample);



	cuComplex* h_covarianceMatrices;
	uint64_t covarianceByteSize = sizeof(cuComplex) * h_valuesPerSample * h_valuesPerSample * h_batchSize;
	cudaMallocHost(&h_covarianceMatrices, covarianceByteSize);
	cudaMemcpy(h_covarianceMatrices, RFIM->d_covarianceMatrix, covarianceByteSize, cudaMemcpyDeviceToHost);

	Utility_WriteSignalMatrixToFile("RoundTripNoReductionCovarianceMatrix1.txt", h_covarianceMatrices, h_valuesPerSample, h_valuesPerSample);
	Utility_WriteSignalMatrixToFile("RoundTripNoReductionCovarianceMatrix2.txt", h_covarianceMatrices + (h_valuesPerSample * h_valuesPerSample), h_valuesPerSample, h_valuesPerSample);
	Utility_WriteSignalMatrixToFile("RoundTripNoReductionCovarianceMatrix3.txt", h_covarianceMatrices + (2 * h_valuesPerSample * h_valuesPerSample), h_valuesPerSample, h_valuesPerSample);
	Utility_WriteSignalMatrixToFile("RoundTripNoReductionCovarianceMatrix4.txt", h_covarianceMatrices + (3 * h_valuesPerSample * h_valuesPerSample), h_valuesPerSample, h_valuesPerSample);
	Utility_WriteSignalMatrixToFile("RoundTripNoReductionCovarianceMatrix5.txt", h_covarianceMatrices + (4 * h_valuesPerSample * h_valuesPerSample), h_valuesPerSample, h_valuesPerSample);
	*/

	//Is the end result the same more or less?
	//Copy the end result back to the host
	cuComplex* h_filteredSignal;
	cudaMallocHost(&h_filteredSignal,  signalByteSize);
	cudaMemcpy(h_filteredSignal, d_filteredSignal, signalByteSize, cudaMemcpyDeviceToHost);

	uint64_t signalLength = h_valuesPerSample * h_numberOfSamples * h_batchSize;
	for(uint64_t i = 0; i < signalLength; ++i)
	{
		if(fabs(h_filteredSignal[i].x) - fabs(h_signal[i].x) > 0.00001f ||
				fabs(h_filteredSignal[i].y) - fabs(h_signal[i].y) > 0.00001f)
		{
			fprintf(stderr,
					"RoundTripNoReduction: Signal is not the same as filtered signal\nSignal[%llu]: real %f, imag %f\nFilteredSignal[%llu]: real %f, imag %f\n",
					i, h_signal[i].x, h_signal[i].y, i, h_filteredSignal[i].x, h_filteredSignal[i].y);
			exit(1);
		}
	}



	//Free everything
	cudaFreeHost(h_signal);
	//cudaFreeHost(h_covarianceMatrices);
	cudaFreeHost(h_filteredSignal);

	cudaFree(d_signal);
	cudaFree(d_filteredSignal);

	curandDestroyGenerator(rngGen);

	RFIMMemoryStructComplexDestroy(RFIM);


}







void MemoryLeakTest()
{
	//Signal
	uint64_t h_valuesPerSample = 26;
	uint64_t h_numberOfSamples  = 1 << 10;
	uint64_t h_dimensionsToReduce = 0;
	uint64_t h_batchSize = 5;
	uint64_t h_numberOfCudaStreams = 8;
	uint64_t h_numberOfThreads = 4;


	//Start up the RNG
	curandGenerator_t rngGen;

	if( curandCreateGenerator(&rngGen, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS)
	{
		fprintf(stderr, "main: Unable to start cuRand library\n");
		exit(1);
	}

	//Set the RNG seed
	if((curandSetPseudoRandomGeneratorSeed(rngGen, 1)) != CURAND_STATUS_SUCCESS)
	{
		fprintf(stderr, "main: Unable to set the RNG Seed value\n");
		exit(1);
	}


	for(uint64_t i = 0; i < 10; ++i)
	{
		RFIMMemoryStruct** RFIMStructArray;
		cudaMallocHost(&RFIMStructArray, sizeof(RFIMMemoryStruct*) * h_numberOfThreads);

		//Allocate all the signal memory
		float* d_signal;
		float* d_filteredSignal;
		uint64_t signalThreadOffset = h_valuesPerSample * h_numberOfSamples * h_batchSize;
		uint64_t signalByteSize = sizeof(float) * h_valuesPerSample * h_numberOfSamples * h_batchSize * h_numberOfThreads;



		cudaMalloc(&d_filteredSignal, signalByteSize);

		d_signal = Device_GenerateWhiteNoiseSignal(&rngGen, h_valuesPerSample, h_numberOfSamples, h_batchSize, h_numberOfThreads);


		//Create a struct for each of the threads
		for(uint64_t currentThreadIndex = 0; currentThreadIndex < h_numberOfThreads; ++currentThreadIndex)
		{
			RFIMStructArray[currentThreadIndex] = RFIMMemoryStructCreate(h_valuesPerSample, h_numberOfSamples,
					h_dimensionsToReduce, h_batchSize, h_numberOfCudaStreams);

		}



		//Start a thread for each RFIMStruct
		//Allocate memory
		std::vector<std::thread*> threadVector;
		//cudaMallocHost(&threadArray, sizeof(std::thread) * h_numberOfThreads);



		//Start the threads
		for(uint64_t currentThreadIndex = 0; currentThreadIndex < h_numberOfThreads; ++currentThreadIndex)
		{
			threadVector.push_back(new std::thread(RFIMRoutine,
					RFIMStructArray[currentThreadIndex],
					d_signal + (currentThreadIndex * signalThreadOffset),
					d_filteredSignal + (currentThreadIndex * signalThreadOffset)));

			/*
			//Placement new, construct an object on already allocated memory
			std::thread* helloThread = new (threadArray + currentThreadIndex) std::thread(RFIMRoutine,
					std::ref(RFIMStructArray[currentThreadIndex]),
					d_signal + (currentThreadIndex * signalThreadOffset),
					d_filteredSignal + (currentThreadIndex * signalThreadOffset));
					*/

		}


		//Wait for all threads to complete
		for(uint64_t currentThreadIndex = 0; currentThreadIndex < h_numberOfThreads; ++currentThreadIndex)
		{
			threadVector[currentThreadIndex]->join();
		}



		//Compare the input and output results and see if they are the same
		float* h_signal;
		float* h_filteredSignal;
		cudaMallocHost(&h_signal, signalByteSize);
		cudaMallocHost(&h_filteredSignal, signalByteSize);
		cudaMemcpy(h_signal, d_signal, signalByteSize, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_filteredSignal, d_filteredSignal, signalByteSize, cudaMemcpyDeviceToHost);

		for(uint64_t signalIndex = 0; signalIndex < h_valuesPerSample * h_numberOfSamples * h_batchSize * h_numberOfThreads; ++signalIndex)
		{
			if(fabs(h_filteredSignal[signalIndex]) - fabs(h_signal[signalIndex]) > 0.00001f)
			{
				fprintf(stderr, "MemoryLeakTest: Signal is not the same as filtered signal\nSignal[%llu] = %f\nFilteredSignal[%llu] = %f\n",
						signalIndex, h_signal[signalIndex], signalIndex, h_filteredSignal[signalIndex]);
				exit(1);
			}
		}



		//Free each of the RFIMStructs
		for(uint64_t currentThreadIndex = 0; currentThreadIndex < h_numberOfThreads; ++currentThreadIndex)
		{
			RFIMMemoryStructDestroy(RFIMStructArray[currentThreadIndex]);

			std::thread* currentThread = threadVector[currentThreadIndex];
			delete currentThread;

		}



		cudaFreeHost(RFIMStructArray);
		cudaFreeHost(h_signal);
		cudaFreeHost(h_filteredSignal);
		//cudaFreeHost(threadArray);

		cudaFree(d_signal);
		cudaFree(d_filteredSignal);
	}

	curandDestroyGenerator(rngGen);
}



void MemoryLeakTestComplex()
{
	//Signal
	uint64_t h_valuesPerSample = 26;
	uint64_t h_numberOfSamples  = 1 << 10;
	uint64_t h_dimensionsToReduce = 0;
	uint64_t h_batchSize = 5;
	uint64_t h_numberOfCudaStreams = 8;
	uint64_t h_numberOfThreads = 4;


	//Start up the RNG
	curandGenerator_t rngGen;

	if( curandCreateGenerator(&rngGen, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS)
	{
		fprintf(stderr, "main: Unable to start cuRand library\n");
		exit(1);
	}

	//Set the RNG seed
	if((curandSetPseudoRandomGeneratorSeed(rngGen, 1)) != CURAND_STATUS_SUCCESS)
	{
		fprintf(stderr, "main: Unable to set the RNG Seed value\n");
		exit(1);
	}


	for(uint64_t i = 0; i < 10000; ++i)
	{
		RFIMMemoryStructComplex** RFIMStructArray;
		cudaMallocHost(&RFIMStructArray, sizeof(RFIMMemoryStructComplex*) * h_numberOfThreads);

		//Allocate all the signal memory
		cuComplex* d_signal;
		cuComplex* d_filteredSignal;
		uint64_t signalThreadOffset = h_valuesPerSample * h_numberOfSamples * h_batchSize;
		uint64_t signalByteSize = sizeof(cuComplex) * h_valuesPerSample * h_numberOfSamples * h_batchSize * h_numberOfThreads;



		cudaMalloc(&d_filteredSignal, signalByteSize);

		d_signal = Device_GenerateWhiteNoiseSignalComplex(&rngGen, h_valuesPerSample, h_numberOfSamples, h_batchSize, h_numberOfThreads);


		//Create a struct for each of the threads
		for(uint64_t currentThreadIndex = 0; currentThreadIndex < h_numberOfThreads; ++currentThreadIndex)
		{
			RFIMStructArray[currentThreadIndex] = RFIMMemoryStructComplexCreate(h_valuesPerSample, h_numberOfSamples,
					h_dimensionsToReduce, h_batchSize, h_numberOfCudaStreams);

		}



		//Start a thread for each RFIMStruct
		//Allocate memory
		std::vector<std::thread*> threadVector;
		//cudaMallocHost(&threadArray, sizeof(std::thread) * h_numberOfThreads);



		//Start the threads
		for(uint64_t currentThreadIndex = 0; currentThreadIndex < h_numberOfThreads; ++currentThreadIndex)
		{
			threadVector.push_back(new std::thread(RFIMRoutineComplex,
					RFIMStructArray[currentThreadIndex],
					d_signal + (currentThreadIndex * signalThreadOffset),
					d_filteredSignal + (currentThreadIndex * signalThreadOffset)));

			/*
			//Placement new, construct an object on already allocated memory
			std::thread* helloThread = new (threadArray + currentThreadIndex) std::thread(RFIMRoutine,
					std::ref(RFIMStructArray[currentThreadIndex]),
					d_signal + (currentThreadIndex * signalThreadOffset),
					d_filteredSignal + (currentThreadIndex * signalThreadOffset));
					*/

		}


		//Wait for all threads to complete
		for(uint64_t currentThreadIndex = 0; currentThreadIndex < h_numberOfThreads; ++currentThreadIndex)
		{
			threadVector[currentThreadIndex]->join();
		}



		//Compare the input and output results and see if they are the same
		cuComplex* h_signal;
		cuComplex* h_filteredSignal;
		cudaMallocHost(&h_signal, signalByteSize);
		cudaMallocHost(&h_filteredSignal, signalByteSize);
		cudaMemcpy(h_signal, d_signal, signalByteSize, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_filteredSignal, d_filteredSignal, signalByteSize, cudaMemcpyDeviceToHost);

		for(uint64_t signalIndex = 0; signalIndex < h_valuesPerSample * h_numberOfSamples * h_batchSize * h_numberOfThreads; ++signalIndex)
		{
			if(fabs(h_filteredSignal[signalIndex].x) - fabs(h_signal[signalIndex].x) > 0.00001f ||
					fabs(h_filteredSignal[signalIndex].y) - fabs(h_signal[signalIndex].y) > 0.00001f)
			{
				fprintf(stderr,
						"MemoryLeakTestComplex: Signal is not the same as filtered signal\nSignal[%llu]: real %f, imag %f\nFilteredSignal[%llu]: real %f, imag %f\n",
						i, h_signal[i].x, h_signal[i].y, i, h_filteredSignal[i].x, h_filteredSignal[i].y);
				exit(1);
			}
		}



		//Free each of the RFIMStructs
		for(uint64_t currentThreadIndex = 0; currentThreadIndex < h_numberOfThreads; ++currentThreadIndex)
		{
			RFIMMemoryStructComplexDestroy(RFIMStructArray[currentThreadIndex]);

			std::thread* currentThread = threadVector[currentThreadIndex];
			delete currentThread;

		}



		cudaFreeHost(RFIMStructArray);
		cudaFreeHost(h_signal);
		cudaFreeHost(h_filteredSignal);
		//cudaFreeHost(threadArray);

		cudaFree(d_signal);
		cudaFree(d_filteredSignal);
	}

	curandDestroyGenerator(rngGen);
}




void RFIMTest()
{
	//generate two streams of data, add a sine wave to it for a bit, and see what happens after it goes through this routine



	//Signal
	uint64_t h_valuesPerSample = 2;
	uint64_t h_numberOfSamples  = 1 << 15;
	uint64_t h_dimensionsToReduce = 1;
	uint64_t h_batchSize = 1;
	uint64_t h_numberOfCudaStreams = 1;
	uint64_t h_numberOfThreads = 1;


	//Start up the RNG
	curandGenerator_t rngGen;

	if( curandCreateGenerator(&rngGen, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS)
	{
		fprintf(stderr, "main: Unable to start cuRand library\n");
		exit(1);
	}

	//Set the RNG seed
	if((curandSetPseudoRandomGeneratorSeed(rngGen, 1)) != CURAND_STATUS_SUCCESS)
	{
		fprintf(stderr, "main: Unable to set the RNG Seed value\n");
		exit(1);
	}


	//Create an RFIMStruct
	RFIMMemoryStruct* RFIM = RFIMMemoryStructCreate(h_valuesPerSample, h_numberOfSamples,
			h_dimensionsToReduce, h_batchSize, h_numberOfCudaStreams);


	//Generate a signal
	uint64_t signalByteSize = sizeof(float) * h_valuesPerSample * h_numberOfSamples * h_batchSize * h_numberOfThreads;
	float* d_signal = Device_GenerateWhiteNoiseSignal(&rngGen, h_valuesPerSample, h_numberOfSamples,
			h_batchSize,  h_numberOfThreads);

	//Copy it to the host so I can add stuff to it
	float* h_signal;
	cudaMallocHost(&h_signal, signalByteSize);
	cudaMemcpy(h_signal, d_signal, signalByteSize, cudaMemcpyDeviceToHost);

	const float pi = 3.14159265359f;

	//Add a sine wave to each beam from sample zero to 4096
	//Sinewave has a frequency of 3 from 0 - 4096 and an amplitude of 3
	for(uint64_t i = 0; i < 4096; ++i)
	{
		//3 = frequency
		float sineValue = sinf( (2 * pi * 3 * i) / 4096);

		//Add sineValue to the existing noisy, uncorrelated signal
		h_signal[i * 2] += sineValue;
		h_signal[(i * 2) + 1] += sineValue;
	}


	//TODO: Write the signal to a file so it can be graphed
	Utility_WriteSignalMatrixToFile("RFIMBefore.txt", h_signal, h_numberOfSamples, h_valuesPerSample);


	//Copy this altered signal back to the device
	cudaMemcpy(d_signal, h_signal, signalByteSize, cudaMemcpyHostToDevice);

	//Run filtered signal
	float* d_filteredSignal;
	cudaMalloc(&d_filteredSignal, signalByteSize);

	//Run RFIM on it
	RFIMRoutine(RFIM, d_signal, d_filteredSignal);

	//Copy the results back to the host
	float* h_filteredSignal;
	cudaMallocHost(&h_filteredSignal, signalByteSize);
	cudaMemcpy(h_filteredSignal, d_filteredSignal, signalByteSize, cudaMemcpyDeviceToHost);

	//TODO: Write the results to a file
	Utility_WriteSignalMatrixToFile("RFIMAfter.txt", h_filteredSignal, h_numberOfSamples, h_valuesPerSample);


	//Free everything
	cudaFreeHost(h_signal);
	cudaFreeHost(h_filteredSignal);

	cudaFree(d_signal);
	cudaFree(d_filteredSignal);

	RFIMMemoryStructDestroy(RFIM);

	curandDestroyGenerator(rngGen);

}








void RunAllUnitTests()
{




	MeanCublasProduction();
	MeanCublasProductionCPU();

	CovarianceHostProduction();

	/*
	MeanCublasBatchedProduction();
	MeanCublasComplexProduction();



	CovarianceCublasProduction();
	CovarianceCublasBatchedProduction();
	CovarianceCublasComplexProduction();



	EigendecompProduction();
	EigendecompBatchedProduction();
	EigendecompComplexProduction();
	 */

	/*
	FilteringProduction();
	FilteringProductionComplex();


	RoundTripNoReduction();
	RoundTripNoReductionBatched();
	RoundTripNoReductionComplex();
	*/

	//MemoryLeakTest();


	//MemoryLeakTestComplex();


	//RFIMTest();

	printf("All tests passed!\n");

}

