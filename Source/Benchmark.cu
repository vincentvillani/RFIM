
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



void RFIMInstance(RFIMMemoryStruct* RFIM, float* d_signal, float* d_filteredSignal, uint64_t iterations)
{
	for(uint64_t i = 0; i < iterations; ++i)
	{
		RFIMRoutine(RFIM, d_signal, d_filteredSignal);
	}
}



void Benchmark()
{

	//Benchmark
	uint64_t iterations = 30;

	//Signal
	uint64_t h_valuesPerSample = 26;
	uint64_t h_numberOfSamples;
	uint64_t h_dimensionsToReduce = 2;
	uint64_t h_batchSize;
	uint64_t h_numberOfCudaStreams;
	uint64_t h_numberOfThreads;


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



		//For each batchSize
		for(uint64_t j = 0; j < 4; ++j)
		{

			h_batchSize = 1 << j;



			//For each numberOfStreams
			for(uint64_t k = 0; k < 7; ++k)
			{



				h_numberOfCudaStreams = 1 << k;




				for(uint64_t p = 0; p < 6; ++p)
				{

					h_numberOfThreads = 1 << p;


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
					std::thread* threadArray;
					cudaMallocHost(&threadArray, sizeof(std::thread) * h_numberOfThreads);




					//Start the timer
					double startTime = cpuSecond();

					//Start the threads
					for(uint64_t currentThreadIndex = 0; currentThreadIndex < h_numberOfThreads; ++currentThreadIndex)
					{
						//Placement new, construct an object on already allocated memory
						new (threadArray + currentThreadIndex) std::thread(RFIMInstance,
								std::ref(RFIMStructArray[currentThreadIndex]),
								d_signal + (currentThreadIndex * signalThreadOffset),
								d_filteredSignal + (currentThreadIndex * signalThreadOffset), iterations);
					}


					//Join with each of the threads
					for(uint64_t currentThreadIndex = 0; currentThreadIndex < h_numberOfThreads; ++currentThreadIndex)
					{
						threadArray[currentThreadIndex].join();
					}





					//Compute stats here
					//calculate total duration
					double totalDuration = cpuSecond() - startTime;

					//find the average time taken for each iteration
					double averageIterationTime = totalDuration / iterations;

					//TODO: *************** ADD THREAD NUM HERE!!!! ***************
					//Calculate the average samples processed per iteration in Mhz
					double averageHz = (h_numberOfSamples * h_batchSize * iterations * h_numberOfThreads) / totalDuration;
					double averageMhz =  averageHz / 1000000.0;



					//Print the results
					printf("Signal: (%llu, %llu, %llu, %llu, %llu)\nIterations: %llu\nTotal time: %fs\nAverage time: %fs\nAverage Mhz: %f\n\n",
							h_valuesPerSample, h_numberOfSamples, h_batchSize, h_numberOfCudaStreams, h_numberOfThreads, iterations, totalDuration, averageIterationTime, averageMhz);





					//Free each of the RFIMStructs
					for(uint64_t currentThreadIndex = 0; currentThreadIndex < h_numberOfThreads; ++currentThreadIndex)
					{
						RFIMMemoryStructDestroy(RFIMStructArray[currentThreadIndex]);
						threadArray[currentThreadIndex].~thread(); //Call the destructor
					}



					cudaFreeHost(RFIMStructArray);
					cudaFreeHost(threadArray);

					cudaFree(d_signal);
					cudaFree(d_filteredSignal);


				}



			}




		}
	}



	curandDestroyGenerator(rngGen);

	printf("Benchmark complete!!\n");

}
