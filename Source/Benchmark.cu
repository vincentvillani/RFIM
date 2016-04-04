
#include "../Header/Benchmark.h"
#include "../Header/RFIM.h"
#include "../Header/RFIMHelperFunctions.h"
#include "../Header/CudaUtilityFunctions.h"

#include <stdio.h>
#include <sys/time.h>
#include <curand.h>
#include <thread>
#include <vector>

double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}


void BenchmarkHelperFunction(RFIMMemoryStruct* RFIMStruct, float** d_columnMajorSignalMatrices, float** d_filteredSignalMatrices, uint64_t iterations)
{
	for(uint64_t i = 0; i < iterations; ++i)
	{
		RFIMRoutine(RFIMStruct, d_columnMajorSignalMatrices, d_filteredSignalMatrices);
	}

	printf("ThreadComplete\n");
}



void Benchmark()
{
	//Benchmark
	uint64_t iterations = 50;

	//Signal
	uint64_t h_valuesPerSample = 26;
	uint64_t h_numberOfSamples;
	uint64_t h_batchSize;
	uint64_t h_threadNum;


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


	//For each numberOfSamples value
	for(uint64_t i = 16; i < 22; ++i)
	{
		h_numberOfSamples = 1 << i;

		//printf("i: %llu\n", i);

		//For each batchSize
		for(uint64_t j = 0; j < 4; ++j)
		{

			h_batchSize = 1 << j;

			//printf("j: %llu\n", j);


			//For each threadNum
			for(uint64_t k = 0; k < 2; ++k)
			{
				h_threadNum = 1 << k;

				//printf("k: %llu\n", k);


				//generate the needed signals
				float*** d_signalMatrices; //= Device_GenerateWhiteNoiseSignal(&rngGen, h_valuesPerSample, h_numberOfSamples, h_batchSize, &RFIMStruct->cudaStream);
				float*** d_filteredSignalMatrices; //= CudaUtility_BatchAllocateDeviceArrays(h_batchSize, sizeof(float) * h_valuesPerSample * h_numberOfSamples, &RFIMStruct->cudaStream);
				RFIMMemoryStruct** RFIMStructPointers; //= (RFIMMemoryStruct*)malloc(sizeof(RFIMMemoryStruct*) * h_threadNum);

				//Allocate batches of signals for each thread and RFIMMemoryStructs
				cudaMallocHost(&d_signalMatrices, sizeof(float*) * h_threadNum);
				cudaMallocHost(&d_filteredSignalMatrices, sizeof(float*) * h_threadNum);
				cudaMallocHost(&RFIMStructPointers, sizeof(sizeof(RFIMMemoryStruct*) * h_threadNum));

				//printf("1.0\n");

				for(uint64_t currentSignalIndex = 0; currentSignalIndex < h_threadNum; ++currentSignalIndex)
				{
					RFIMStructPointers[currentSignalIndex] = RFIMMemoryStructCreate(h_valuesPerSample, h_numberOfSamples, 2,
							h_batchSize, currentSignalIndex);
					d_signalMatrices[currentSignalIndex] = Device_GenerateWhiteNoiseSignal(&rngGen, h_valuesPerSample,
							h_numberOfSamples, h_batchSize, &(RFIMStructPointers[currentSignalIndex]->cudaStream));
					d_filteredSignalMatrices[currentSignalIndex] = CudaUtility_BatchAllocateDeviceArrays(h_batchSize,
							sizeof(float) * h_valuesPerSample * h_numberOfSamples, &(RFIMStructPointers[currentSignalIndex]->cudaStream));
				}



				//Allocate memory for the threads
				std::vector<std::thread*> threadVector;


				//Start the timer
				double startTime = cpuSecond();

				//printf("1.1\n");

				//Start the appropriate number of threads
				for(uint64_t currentThreadIdx = 0; currentThreadIdx < h_threadNum; ++currentThreadIdx)
				{
					std::thread* newThread = new std::thread(BenchmarkHelperFunction, RFIMStructPointers[currentThreadIdx],
							d_signalMatrices[currentThreadIdx], d_filteredSignalMatrices[currentThreadIdx], iterations);
					threadVector.push_back(newThread);
				}


				//printf("1.2\n");

				//Join with all the threads
				for(uint64_t currentThreadIdx = 0; currentThreadIdx < h_threadNum; ++currentThreadIdx)
				{
					threadVector[currentThreadIdx]->join();
				}

				//printf("1.3\n");

				//Compute stats here
				//calculate total duration
				double totalDuration = cpuSecond() - startTime;

				//find the average time taken for each iteration
				double averageIterationTime = totalDuration / iterations;

				//Calculate the average samples processed per iteration in Mhz
				double averageHz = (h_numberOfSamples * h_batchSize * iterations * h_threadNum) / totalDuration;
				double averageMhz =  averageHz / 1000000.0;



				//Print the results
				printf("Signal: (%llu, %llu, %llu, %llu)\nIterations: %llu\nTotal time: %fs\nAverage time: %fs\nAverage Mhz: %f\n\n",
						h_valuesPerSample, h_numberOfSamples, h_batchSize, h_threadNum, iterations, totalDuration, averageIterationTime, averageMhz);



				//Deallocate thread memory, RFIMStruct and signal memory
				for(uint64_t currentThreadIdx = 0; currentThreadIdx < h_threadNum; ++currentThreadIdx)
				{
					delete threadVector[currentThreadIdx]; //Free the thread memory
					CudaUtility_BatchDeallocateDeviceArrays(d_signalMatrices[currentThreadIdx], h_batchSize, &(RFIMStructPointers[currentThreadIdx]->cudaStream));
					CudaUtility_BatchDeallocateDeviceArrays(d_filteredSignalMatrices[currentThreadIdx], h_batchSize, &(RFIMStructPointers[currentThreadIdx]->cudaStream));
					RFIMMemoryStructDestroy(RFIMStructPointers[currentThreadIdx]);
				}

				//Free pointers to signal pointers, RFIMStructs etc
				cudaFreeHost(d_signalMatrices);
				cudaFreeHost(d_filteredSignalMatrices);
				cudaFreeHost(RFIMStructPointers);

			}


		}
	}



	printf("Benchmark complete!!\n");

}
