
#include "../Header/Benchmark.h"
#include "../Header/RFIM.h"
#include "../Header/RFIMHelperFunctions.h"
#include "../Header/CudaUtilityFunctions.h"

#include <stdio.h>
#include <sys/time.h>
#include <curand.h>

double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}


void Benchmark()
{
	//Benchmark
	uint64_t iterations = 100;

	//Signal
	uint64_t h_valuesPerSample = 26;
	uint64_t h_numberOfSamples;
	uint64_t h_batchSize;


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
	for(uint64_t i = 19; i < 20; ++i)
	{
		h_numberOfSamples = 1 << i;

		//For each batchSize
		for(uint64_t j = 5; j < 7; ++j)
		{

			h_batchSize = 1 << j;


			//1. Generate signal and allocate room for the filtered signal
			float** d_signalMatrices = Device_GenerateWhiteNoiseSignal(&rngGen, h_valuesPerSample, h_numberOfSamples, h_batchSize);
			float** d_filteredSignalMatrices = CudaUtility_BatchAllocateDeviceArrays(h_batchSize, sizeof(float) * h_valuesPerSample * h_numberOfSamples);

			//2. Create the RFIMStruct
			RFIMMemoryStruct* RFIMStruct = RFIMMemoryStructCreate(h_valuesPerSample, h_numberOfSamples, 2, h_batchSize);

			//3. Run the benchmark
			double startTime = cpuSecond();

			//Run RFIMRoutine iteration times
			for(uint64_t currentIteration = 0; currentIteration < iterations; ++currentIteration)
			{
				//printf("Iteration: %u\n", currentIteration);

				RFIMRoutine(RFIMStruct, d_signalMatrices, d_filteredSignalMatrices);
			}

			//calculate total duration
			double totalDuration = cpuSecond() - startTime;

			//find the average time taken for each iteration
			double averageIterationTime = totalDuration / iterations;

			//Calculate the average samples processed per iteration in Mhz
			double averageHz = (RFIMStruct->h_numberOfSamples * h_batchSize * iterations) / totalDuration;
			double averageMhz =  averageHz / 1000000.0;



			//Print the results
			printf("Signal: (%llu, %llu, %llu)\nIterations: %llu\nTotal time: %fs\nAverage time: %fs\nAverage Mhz: %f\n\n",
					RFIMStruct->h_valuesPerSample, RFIMStruct->h_numberOfSamples, RFIMStruct->h_batchSize, iterations, totalDuration, averageIterationTime, averageMhz);


			//4. Free everything
			CudaUtility_BatchDeallocateDeviceArrays(d_signalMatrices, h_batchSize);
			CudaUtility_BatchDeallocateDeviceArrays(d_filteredSignalMatrices, h_batchSize);
			RFIMMemoryStructDestroy(RFIMStruct);

		}
	}



	printf("Benchmark complete!!\n");

}
