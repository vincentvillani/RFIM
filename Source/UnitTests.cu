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
void EigendecompProduction();
void FilteringProduction();
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




void EigendecompProduction()
{

	uint64_t valuesPerSample = 2;
	uint64_t numberOfSamples = 3; //THIS MAY MAKE THE UNIT TEST FAIL!?
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






void RunAllUnitTests()
{
	MeanCublasProduction();
	CovarianceCublasProduction();
	EigendecompProduction();
	FilteringProduction();

	printf("All tests passed!\n");

}

