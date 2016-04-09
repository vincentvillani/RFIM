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
#include "../Header/RFIM.h"

#include <cublas.h>

#include <assert.h>
#include <cmath>
#include <string>
#include <thread>
#include <vector>


//Production tests
void MeanCublasProduction();
void CovarianceCublasProduction();
void EigendecompProduction();
void FilteringProduction();

void RoundTripNoReduction();
//void TransposeProduction();
//void GraphProduction();

void MemoryLeakTest();





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
		fprintf(stderr, "RoundTripNoReduction(): Something has done wrong!\n");
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
		if(fabs(h_filteredSignal[i]) - fabs(h_signal[i]) > 0.1f)
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


	for(uint64_t i = 0; i < 10000; ++i)
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
			if(fabs(h_filteredSignal[signalIndex]) - fabs(h_signal[signalIndex]) > 0.1f)
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




void RunAllUnitTests()
{

	MeanCublasProduction();
	CovarianceCublasProduction();
	EigendecompProduction();
	FilteringProduction();

	RoundTripNoReduction();


	MemoryLeakTest();

	printf("All tests passed!\n");

}

