
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





void Benchmark()
{

	//Benchmark
	uint64_t iterations = 1;

	//Signal
	uint64_t h_valuesPerSample = 26;
	uint64_t h_numberOfSamples;
	uint64_t h_dimensionsToReduce = 2;
	uint64_t h_batchSize;
	//uint64_t h_threadNum;
	uint64_t h_numberOfCudaStreams;


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
		for(uint64_t j = 0; j < 8; ++j)
		{

			h_batchSize = 1 << j;



			//For each numberOfStreams
			for(uint64_t k = 0; k < 7; ++k)
			{

				h_numberOfCudaStreams = 1 << k;



				RFIMMemoryStruct* RFIMStruct = RFIMMemoryStructCreate(h_valuesPerSample, h_numberOfSamples,
						h_dimensionsToReduce, h_batchSize, h_numberOfCudaStreams);

				float* d_signal = Device_GenerateWhiteNoiseSignal(&rngGen, h_valuesPerSample, h_numberOfSamples, h_batchSize);
				float* d_filteredSignal;
				cudaMalloc(&d_filteredSignal, sizeof(float) * h_valuesPerSample * h_numberOfSamples * h_batchSize);


				//Start the timer
				double startTime = cpuSecond();

				for(uint64_t currentIteration = 0; currentIteration < iterations; ++currentIteration)
				{
					RFIMRoutine(RFIMStruct, d_signal, d_filteredSignal);
					//printf("iterations done\n", j);
				}

				//Don't need to do this as each RFIMRoutine will do this itself
				/*
				//Wait for everything to finish
				cudaError_t cudaError = cudaDeviceSynchronize();

				if(cudaError != cudaSuccess)
				{
					fprintf(stderr, "Benchmark: Something went wrong :(\n");
					exit(1);
				}
				*/


				//Compute stats here
				//calculate total duration
				double totalDuration = cpuSecond() - startTime;

				//find the average time taken for each iteration
				double averageIterationTime = totalDuration / iterations;

				//TODO: *************** ADD THREAD NUM HERE!!!! ***************
				//Calculate the average samples processed per iteration in Mhz
				double averageHz = (h_numberOfSamples * h_batchSize * iterations) / totalDuration;
				double averageMhz =  averageHz / 1000000.0;



				//Print the results
				printf("Signal: (%llu, %llu, %llu)\nIterations: %llu\nTotal time: %fs\nAverage time: %fs\nAverage Mhz: %f\n\n",
						h_valuesPerSample, h_numberOfSamples, h_batchSize, iterations, totalDuration, averageIterationTime, averageMhz);


				cudaFree(d_signal);
				cudaFree(d_filteredSignal);
				RFIMMemoryStructDestroy(RFIMStruct);
			}




		}
	}



	printf("Benchmark complete!!\n");

}
