
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
	/*
	//Benchmark
	uint64_t iterations = 1;

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

		printf("i: %llu\n", i);

		//For each batchSize
		for(uint64_t j = 0; j < 4; ++j)
		{

			h_batchSize = 1 << j;

			printf("j: %llu\n", j);


			//Start the timer
			double startTime = cpuSecond();

			RFIMMemoryStruct* RFIMStruct = RFIMMemoryStructCreate(h_valuesPerSample, h_numberOfSamples, 2, h_batchSize, 0);

			float** d_signal = Device_GenerateWhiteNoiseSignal(&rngGen, h_valuesPerSample, h_numberOfSamples, h_batchSize, &RFIMStruct->cudaStream);
			float** d_filteredSignal = CudaUtility_BatchAllocateDeviceArrays(h_batchSize, sizeof(float) * h_valuesPerSample * h_numberOfSamples, &RFIMStruct->cudaStream);


			for(uint64_t currentIteration = 0; currentIteration < iterations; ++currentIteration)
			{
				RFIMRoutine(RFIMStruct, d_signal, d_filteredSignal);
				//printf("iterations done\n", j);
			}





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
			printf("Signal: (%llu, %llu, %llu, %llu)\nIterations: %llu\nTotal time: %fs\nAverage time: %fs\nAverage Mhz: %f\n\n",
					h_valuesPerSample, h_numberOfSamples, h_batchSize, h_threadNum, iterations, totalDuration, averageIterationTime, averageMhz);

			CudaUtility_BatchDeallocateDeviceArrays(d_signal, h_batchSize, &RFIMStruct->cudaStream);
			CudaUtility_BatchDeallocateDeviceArrays(d_filteredSignal, h_batchSize, &RFIMStruct->cudaStream);
			RFIMMemoryStructDestroy(RFIMStruct);



		}
	}



	printf("Benchmark complete!!\n");

	*/

}
