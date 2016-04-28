
#include "../Header/Benchmark.h"
#include "../Header/RFIM.h"
#include "../Header/RFIMHelperFunctions.h"
#include "../Header/RFIMMemoryStructComplex.h"
#include "../Header/UtilityFunctions.h"
#include "../Header/CudaUtilityFunctions.h"

#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>
#include <curand.h>
#include <thread>
#include <vector>
#include <sstream>

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


void RFIMInstanceBatched(RFIMMemoryStructBatched* RFIM, float** d_signal, float** d_filteredSignal, uint64_t iterations)
{
	for(uint64_t i = 0; i < iterations; ++i)
	{
		RFIMRoutineBatched(RFIM, d_signal, d_filteredSignal);
	}
}



void RFIMInstanceComplex(RFIMMemoryStructComplex* RFIM, cuComplex* d_signal, cuComplex* d_filteredSignal, uint64_t iterations)
{
	for(uint64_t i = 0; i < iterations; ++i)
	{
		RFIMRoutineComplex(RFIM, d_signal, d_filteredSignal);
	}
}


void RFIMInstanceHost(RFIMMemoryStructCPU* RFIM, float* h_signal, float* h_filteredSignal, uint64_t iterations)
{
	for(uint64_t i = 0; i < iterations; ++i)
	{
		RFIMRoutineHost(RFIM, h_signal, h_filteredSignal);
	}
}


void Benchmark()
{

	//Benchmark
	uint64_t iterations = 5;

	//Signal
	uint64_t h_valuesPerSample = 13;
	uint64_t h_numberOfSamples;
	uint64_t h_dimensionsToReduce = 1;
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
	for(uint64_t i = 14; i < 15; ++i)
	{
		//h_numberOfSamples = 1 << i;
		h_numberOfSamples = 15625;

		//For each batchSize
		for(uint64_t j = 1; j < 2; ++j)
		{

			//h_batchSize = 1 << j;
			//h_numberOfCudaStreams = 1 << j;
			h_batchSize = 1024;
			h_numberOfCudaStreams = 1024;


			for(uint64_t p = 0; p < 1; ++p)
			{

				//h_numberOfThreads = 1 << p;
				h_numberOfThreads = 1;



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
				std::vector<std::thread*> threadVector;




				//Start the timer
				double startTime = cpuSecond();

				//Start the threads
				for(uint64_t currentThreadIndex = 0; currentThreadIndex < h_numberOfThreads; ++currentThreadIndex)
				{
					//Placement new, construct an object on already allocated memory
					threadVector.push_back( new std::thread(RFIMInstance,
							RFIMStructArray[currentThreadIndex],
							d_signal + (currentThreadIndex * signalThreadOffset),
							d_filteredSignal + (currentThreadIndex * signalThreadOffset), iterations));


				}


				//Join with each of the threads
				for(uint64_t currentThreadIndex = 0; currentThreadIndex < h_numberOfThreads; ++currentThreadIndex)
				{
					threadVector[currentThreadIndex]->join();
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
					std::thread* currentThread = threadVector[currentThreadIndex];
					delete currentThread;

				}



				cudaFreeHost(RFIMStructArray);


				cudaFree(d_signal);
				cudaFree(d_filteredSignal);


			}

		}
	}




	curandDestroyGenerator(rngGen);

	printf("Benchmark complete!!\n");

}



void BenchmarkBatched()
{
	//Benchmark
	uint64_t iterations = 1;

	//Signal
	uint64_t h_valuesPerSample = 13;
	uint64_t h_numberOfSamples;
	uint64_t h_dimensionsToReduce = 1;
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
	for(uint64_t i = 14; i < 15; ++i)
	{
		//h_numberOfSamples = 1 << i;
		h_numberOfSamples = 170;

		//For each batchSize
		for(uint64_t j = 1; j < 2; ++j)
		{

			//h_batchSize = 1 << j;
			//h_numberOfCudaStreams = 1 << j;
			h_batchSize = 5548;
			h_numberOfCudaStreams = h_batchSize + 1;


			for(uint64_t p = 1; p < 3; ++p)
			{

				//h_numberOfThreads = 1 << p;
				h_numberOfThreads = p;



				RFIMMemoryStructBatched** RFIMStructArray;
				cudaMallocHost(&RFIMStructArray, sizeof(RFIMMemoryStructBatched*) * h_numberOfThreads);

				//Allocate all the signal memory
				float* d_signal;
				float* d_filteredSignal;
				//uint64_t signalThreadOffset = h_valuesPerSample * h_numberOfSamples * h_batchSize;
				uint64_t signalByteSize = sizeof(float) * h_valuesPerSample * h_numberOfSamples * h_batchSize * h_numberOfThreads;


				cudaMalloc(&d_filteredSignal, signalByteSize);

				d_signal = Device_GenerateWhiteNoiseSignal(&rngGen, h_valuesPerSample, h_numberOfSamples, h_batchSize, h_numberOfThreads);

				uint64_t singleSignalLength = h_valuesPerSample * h_numberOfSamples;


				float** d_signalBatched = CudaUtility_createBatchedDevicePointers(d_signal, singleSignalLength, h_batchSize);
				float** d_filteredSignalBatched = CudaUtility_createBatchedDevicePointers(d_filteredSignal,
						singleSignalLength, h_batchSize);


				//Create a struct for each of the threads
				for(uint64_t currentThreadIndex = 0; currentThreadIndex < h_numberOfThreads; ++currentThreadIndex)
				{
					RFIMStructArray[currentThreadIndex] = RFIMMemoryStructBatchedCreate(h_valuesPerSample, h_numberOfSamples,
							h_dimensionsToReduce, h_batchSize, h_numberOfCudaStreams);

				}



				//Start a thread for each RFIMStruct
				std::vector<std::thread*> threadVector;




				//Start the timer
				double startTime = cpuSecond();

				//Start the threads
				for(uint64_t currentThreadIndex = 0; currentThreadIndex < h_numberOfThreads; ++currentThreadIndex)
				{
					//Placement new, construct an object on already allocated memory
					threadVector.push_back( new std::thread(RFIMInstanceBatched,
							RFIMStructArray[currentThreadIndex],
							d_signalBatched,
							d_filteredSignalBatched, iterations));


				}


				//Join with each of the threads
				for(uint64_t currentThreadIndex = 0; currentThreadIndex < h_numberOfThreads; ++currentThreadIndex)
				{
					threadVector[currentThreadIndex]->join();
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
					std::thread* currentThread = threadVector[currentThreadIndex];
					delete currentThread;

				}



				cudaFreeHost(RFIMStructArray);


				cudaFree(d_signal);
				cudaFree(d_signalBatched);
				cudaFree(d_filteredSignal);
				cudaFree(d_filteredSignalBatched);


			}

		}
	}




	curandDestroyGenerator(rngGen);

	printf("Benchmark complete!!\n");
}



void BenchmarkComplex()
{
	//Benchmark
	uint64_t iterations = 50;

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
	for(uint64_t i = 14; i < 24; ++i)
	{
		h_numberOfSamples = 1 << i;



		//For each batchSize
		for(uint64_t j = 0; j < 5; ++j)
		{

			h_batchSize = 1 << j;
			h_numberOfCudaStreams = 1 << j;


			for(uint64_t p = 0; p < 1; ++p)
			{

				h_numberOfThreads = 1 << p;




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
				std::vector<std::thread*> threadVector;




				//Start the timer
				double startTime = cpuSecond();

				//Start the threads
				for(uint64_t currentThreadIndex = 0; currentThreadIndex < h_numberOfThreads; ++currentThreadIndex)
				{
					//Placement new, construct an object on already allocated memory
					threadVector.push_back( new std::thread(RFIMInstanceComplex,
							RFIMStructArray[currentThreadIndex],
							d_signal + (currentThreadIndex * signalThreadOffset),
							d_filteredSignal + (currentThreadIndex * signalThreadOffset), iterations));


				}


				//Join with each of the threads
				for(uint64_t currentThreadIndex = 0; currentThreadIndex < h_numberOfThreads; ++currentThreadIndex)
				{
					threadVector[currentThreadIndex]->join();
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
					RFIMMemoryStructComplexDestroy(RFIMStructArray[currentThreadIndex]);
					std::thread* currentThread = threadVector[currentThreadIndex];
					delete currentThread;
				}



				cudaFreeHost(RFIMStructArray);


				cudaFree(d_signal);
				cudaFree(d_filteredSignal);


			}



		}
	}



	curandDestroyGenerator(rngGen);

	printf("Benchmark complete!!\n");
}




void BenchmarkHost()
{
	//Benchmark
	uint64_t iterations = 5;

	//Signal
	uint64_t h_valuesPerSample = 13;
	uint64_t h_numberOfSamples;
	uint64_t h_dimensionsToReduce = 1;
	uint64_t h_batchSize;
	uint64_t h_numberOfThreads;




	//For each numberOfSamples value
	for(uint64_t i = 14; i < 15; ++i)
	{
		//h_numberOfSamples = 1 << i;
		h_numberOfSamples = 15625;

		//For each batchSize
		for(uint64_t j = 1; j < 2; ++j)
		{

			//h_batchSize = 1 << j;
			//h_numberOfCudaStreams = 1 << j;
			h_batchSize = 1024;


			for(uint64_t p = 0; p < 4; ++p)
			{

				if(p == 0)
				{
					h_numberOfThreads = 1;
				}
				else
				{
					h_numberOfThreads = 1 << p;
					//h_numberOfThreads = p;
				}




				RFIMMemoryStructCPU** RFIMStructArray = (RFIMMemoryStructCPU**)malloc(sizeof(RFIMMemoryStructCPU*) * h_numberOfThreads);


				//Allocate all the signal memory
				float* h_signal;
				float* h_filteredSignal;
				uint64_t signalThreadOffset = h_valuesPerSample * h_numberOfSamples * h_batchSize;
				uint64_t signalLength = h_valuesPerSample * h_numberOfSamples * h_batchSize * h_numberOfThreads;
				uint64_t signalByteSize = sizeof(float) * signalLength;


				h_signal = Utility_GenerateWhiteNoiseHostMalloc(signalLength, 0.0f, 1.0f);
				h_filteredSignal = (float*)malloc(signalByteSize);





				//Create a struct for each of the threads
				for(uint64_t currentThreadIndex = 0; currentThreadIndex < h_numberOfThreads; ++currentThreadIndex)
				{
					RFIMStructArray[currentThreadIndex] = RFIMMemoryStructCreateCPU(h_valuesPerSample, h_numberOfSamples,
							h_dimensionsToReduce, h_batchSize);

				}



				//Start a thread for each RFIMStruct
				std::vector<std::thread*> threadVector;




				//Start the timer
				double startTime = cpuSecond();

				//Start the threads
				for(uint64_t currentThreadIndex = 0; currentThreadIndex < h_numberOfThreads; ++currentThreadIndex)
				{
					//Placement new, construct an object on already allocated memory
					threadVector.push_back( new std::thread(RFIMInstanceHost,
							RFIMStructArray[currentThreadIndex],
							h_signal + (currentThreadIndex * signalThreadOffset),
							h_filteredSignal + (currentThreadIndex * signalThreadOffset), iterations));


				}


				//Join with each of the threads
				for(uint64_t currentThreadIndex = 0; currentThreadIndex < h_numberOfThreads; ++currentThreadIndex)
				{
					threadVector[currentThreadIndex]->join();
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
				printf("Signal: (%llu, %llu, %llu, %llu)\nIterations: %llu\nTotal time: %fs\nAverage time: %fs\nAverage Mhz: %f\n\n",
						h_valuesPerSample, h_numberOfSamples, h_batchSize, h_numberOfThreads, iterations, totalDuration, averageIterationTime, averageMhz);





				//Free each of the RFIMStructs
				for(uint64_t currentThreadIndex = 0; currentThreadIndex < h_numberOfThreads; ++currentThreadIndex)
				{
					RFIMMemoryStructDestroy(RFIMStructArray[currentThreadIndex]);
					std::thread* currentThread = threadVector[currentThreadIndex];
					delete currentThread;

				}



				free(RFIMStructArray);


				free(h_signal);
				free(h_filteredSignal);


			}

		}
	}


	printf("Benchmark complete!!\n");
}





//Add a sine wave with random amplitudes but equal phase to multiple beams
//1/4 beams, half beams, all beams
void BenchmarkRFIMConstantInterferor()
{
	//Signal
	uint64_t h_valuesPerSample = 13;
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


	float whiteNoiseMean = 0.0f;
	float whiteNoiseStdDev = 1.0f;

	float h_sineWaveFreq = 3;
	float h_sineWaveAmplitude = 5;
	uint64_t h_numberOfBeamsToAdd; //Start at three, add three each time through the loop


	//Each time through this loop, add the interferor to one more beam
	//0 beams to all beams
	for(h_numberOfBeamsToAdd = 0; h_numberOfBeamsToAdd < h_valuesPerSample; ++h_numberOfBeamsToAdd)
	{



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



		//Calculate the variance of the signal, we'll need this later
		float h_totalVarianceBefore = Utility_Variance(h_signal, h_valuesPerSample * h_numberOfSamples);
		float* h_subSignalVarianceBefore = Utility_SubSignalVariance(h_signal, h_valuesPerSample, h_numberOfSamples);


		const float pi = 3.14159265359f;



		float* h_sineWave;
		cudaMallocHost(&h_sineWave, signalByteSize);

		//Add a sine wave to the beams
		for(uint64_t i = 0; i < h_numberOfSamples; ++i)
		{

			h_sineWave[i] =  sinf( (2 * pi * h_sineWaveFreq * i) / h_numberOfSamples) * h_sineWaveAmplitude;

			//Add sineValue to the existing noisy, uncorrelated signal to each beam
			for(uint64_t j = 0; j < h_numberOfBeamsToAdd; ++j)
			{
				h_signal[ (i * h_valuesPerSample) + j] += h_sineWave[i];
			}

		}


		std::stringstream ss;
		ss << "RFIMBenchmark/BenchmarkRFIMConstantInterferor/before" << h_numberOfBeamsToAdd << "BeamsToAdd.txt";


		Utility_WriteSignalMatrixToFile( ss.str(), h_signal, h_numberOfSamples, h_valuesPerSample);



		//Copy this altered signal back to the device
		cudaMemcpy(d_signal, h_signal, signalByteSize, cudaMemcpyHostToDevice);

		//Allocate memory for the output
		float* d_filteredSignal;
		cudaMalloc(&d_filteredSignal, signalByteSize);


		//Carry out the RFIM Routine
		RFIMRoutine(RFIM, d_signal, d_filteredSignal);



		//Copy the result back to the host
		float* h_filteredSignal;
		cudaMallocHost(&h_filteredSignal, signalByteSize);
		cudaMemcpy(h_filteredSignal, d_filteredSignal, signalByteSize, cudaMemcpyDeviceToHost);


		ss.str("");
		ss << "RFIMBenchmark/BenchmarkRFIMConstantInterferor/after" << h_numberOfBeamsToAdd << "BeamsToAdd.txt";



		//Write the result to file
		Utility_WriteSignalMatrixToFile( ss.str(), h_filteredSignal, h_numberOfSamples, h_valuesPerSample);


		//Compute a bunch of important stats


		//After variances

		float h_totalSignalVarianceAfter = Utility_Variance(h_filteredSignal, h_valuesPerSample * h_numberOfSamples);
		float* h_subSignalVarianceAfter = Utility_SubSignalVariance(h_filteredSignal, h_valuesPerSample, h_numberOfSamples);


		//TODO: Does it make sense to compute the signal to noise of a sine wave? I guess it does to see how strong it is?
		//Signal to noise of the interferor waves
		float h_signalToNoise = Utility_SignalToNoiseRatio(h_sineWave, h_numberOfSamples, h_sineWaveAmplitude);


		//Copy the largest eigenvalue over
		float h_largestEigenvalue;
		cudaMemcpy(&h_largestEigenvalue, RFIM->d_S, sizeof(float), cudaMemcpyDeviceToHost);


		float* h_subSignalCorrelationCoefficents = Utility_CoefficentOfCrossCorrelation(h_filteredSignal, h_sineWave,
				h_valuesPerSample, h_numberOfSamples, h_valuesPerSample * h_numberOfSamples);



		ss.str("");
		ss << "RFIMBenchmark/BenchmarkRFIMConstantInterferor/statsFile" << h_numberOfBeamsToAdd << "BeamsToAdd.txt";


		std::string statsFilename = ss.str();

		//Write all this stuff to disk
		FILE* statsFile = std::fopen(statsFilename.c_str(), "w");

		if(statsFile == NULL)
		{
			fprintf(stderr, "BenchmarkRFIMConstantInterferor: Unable to open statsFile for writing\n");
			exit(1);
		}


		//Start writing data to the file
		fprintf(statsFile, "Signal Info\n");
		fprintf(statsFile, "Number of samples: %llu\n", h_numberOfSamples);
		fprintf(statsFile, "White noise mean: %f\n", whiteNoiseMean);
		fprintf(statsFile, "White noise standard deviation: %f\n\n", whiteNoiseStdDev);


		fprintf(statsFile, "Interference Info\n");
		fprintf(statsFile, "Frequency: %f\n", h_sineWaveFreq);
		fprintf(statsFile, "Amplitude: %f\n", h_sineWaveAmplitude);
		fprintf(statsFile, "Signal to Noise: %f\n", h_signalToNoise);
		fprintf(statsFile, "Added to %llu beams\n", h_numberOfBeamsToAdd);
		fprintf(statsFile, "Number of samples: %llu\n\n", h_numberOfSamples);


		fprintf(statsFile, "Eigenvectors Info\n");
		fprintf(statsFile, "Dimensions removed: %llu\n", h_dimensionsToReduce);
		fprintf(statsFile, "Largest eigenvalue: %f\n\n", h_largestEigenvalue);


		fprintf(statsFile, "Variance Info\n");
		fprintf(statsFile, "Total variance before: %f\n", h_totalVarianceBefore);
		for(uint64_t i = 0; i < h_valuesPerSample; ++i)
		{
			fprintf(statsFile, "Beam %llu variance before: %f\n", i, h_subSignalVarianceBefore[i]);
		}



		fprintf(statsFile, "\nTotal variance after: %f\n", h_totalSignalVarianceAfter);
		for(uint64_t i = 0; i < h_valuesPerSample; ++i)
		{
			fprintf(statsFile, "Beam %llu variance after: %f\n", i, h_subSignalVarianceAfter[i]);
		}


		fprintf(statsFile, "\n");

		fprintf(statsFile, "Correlation Coefficent Info\n");
		for(uint64_t i = 0; i < h_valuesPerSample; ++i)
		{
			fprintf(statsFile, "Beam %llu Correlation Coefficent: %f\n", i, h_subSignalCorrelationCoefficents[i]);
		}


		fprintf(statsFile, "\n");

		std::fclose(statsFile);



		//Free all memory
		RFIMMemoryStructDestroy(RFIM);

		cudaFree(d_signal);
		cudaFree(d_filteredSignal);

		cudaFreeHost(h_signal);
		cudaFreeHost(h_sineWave);
		cudaFreeHost(h_filteredSignal);

		cudaFreeHost(h_subSignalVarianceBefore);
		cudaFreeHost(h_subSignalVarianceAfter);
		cudaFreeHost(h_subSignalCorrelationCoefficents);


	}

	curandDestroyGenerator(rngGen);


}




void BenchmarkRFIMVariableInterferorVariableEigenvectorRemoval()
{
	//Signal
	uint64_t h_valuesPerSample = 13;
	uint64_t h_numberOfSamples  = 1 << 15;
	uint64_t h_dimensionsToReduce;
	uint64_t h_batchSize = 1;
	uint64_t h_numberOfCudaStreams = 1;
	uint64_t h_numberOfThreads = 1;
	uint64_t h_numberOfBeamsToAdd;

	float whiteNoiseMean = 0.0f;
	float whiteNoiseStdDev = 1.0f;

	float sineWaveMean = 3;
	float sineWaveStdDev = 6;

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





	//generate 13 beams
	//Generate x sine waves
	//Add x sine waves to the 13 beams
		//Remove y amount of eigenvectors
		//Do the RFIM
		//Write the results to a file



	//Generate i sine waves at different amplitudes
	for(uint64_t i = 1; i < h_valuesPerSample; ++i)
	{

		h_numberOfBeamsToAdd = i;

		//Remove j eigenvectors
		for(uint64_t j = 1; j < h_valuesPerSample; ++j)
		{

			//printf("1\n");
			h_dimensionsToReduce = j;

			//Generate h_valuesPerSample beams
			float* d_signal = Device_GenerateWhiteNoiseSignal(&rngGen, h_valuesPerSample, h_numberOfSamples, h_batchSize,
					h_numberOfThreads);

			//Allocate memory for the results of RFIM
			uint64_t signalLength = h_valuesPerSample * h_numberOfSamples * h_batchSize * h_numberOfThreads;
			uint64_t signalByteSize = sizeof(float) * signalLength;

			float* d_filteredSignal;
			cudaMalloc(&d_filteredSignal, signalByteSize);

			//Allocate memory on the host
			float* h_signal;
			float* h_filteredSignal;

			cudaMallocHost(&h_signal, signalByteSize);
			cudaMallocHost(&h_filteredSignal, signalByteSize);


			//Copy the white noise signal to the host
			cudaMemcpy(h_signal, d_signal, signalByteSize, cudaMemcpyDeviceToHost);


			//Compute the before variance
			//Calculate the variance of the signal, we'll need this later
			float h_totalVarianceBefore = Utility_Variance(h_signal, h_valuesPerSample * h_numberOfSamples);
			float* h_subSignalVarianceBefore = Utility_SubSignalVariance(h_signal, h_valuesPerSample, h_numberOfSamples);



			uint64_t sineWaveSignalLength = h_numberOfSamples;
			//uint64_t sineWaveByteSize = sizeof(float) * sineWaveSignalLength;

			float** sineWaves;
			cudaMallocHost(&sineWaves, sizeof(float*) * h_numberOfBeamsToAdd);

			float* h_sineWaveAmplitudes;
			cudaMallocHost(&h_sineWaveAmplitudes, sizeof(float) * h_numberOfBeamsToAdd);

			float* h_sineWaveFrequencies;
			cudaMallocHost(&h_sineWaveFrequencies, sizeof(float) * h_numberOfBeamsToAdd);


			//printf("2\n");

			//Generate the sine waves and add it to the appropriate beam
			for(uint64_t currentSineWaveIndex = 0; currentSineWaveIndex < h_numberOfBeamsToAdd; ++currentSineWaveIndex)
			{

				h_sineWaveAmplitudes[currentSineWaveIndex] = Utility_GenerateSingleWhiteNoiseValueHost(sineWaveMean, sineWaveStdDev);
				h_sineWaveFrequencies[currentSineWaveIndex] = Utility_GenerateSingleWhiteNoiseValueHost(sineWaveMean, sineWaveStdDev);

				//printf("2.1\n");

				//Allocate memory for the current sine wave
				cudaMallocHost(sineWaves + currentSineWaveIndex, signalByteSize);

				//Generate the actual sine wave values
				sineWaves[currentSineWaveIndex] = Utility_GenerateSineWaveHost(sineWaveSignalLength,
						h_sineWaveFrequencies[currentSineWaveIndex], h_sineWaveAmplitudes[currentSineWaveIndex]);


				//printf("2.2\n");

				//Add each sine wave to the appropriate beam
				for(uint64_t currentSineWaveValueIndex = 0; currentSineWaveValueIndex < h_numberOfSamples; ++currentSineWaveValueIndex)
				{
					h_signal[ (currentSineWaveValueIndex * h_valuesPerSample) + currentSineWaveIndex] +=
							sineWaves[currentSineWaveIndex][currentSineWaveValueIndex];

					//printf("2.3\n");
				}

				//printf("2.4\n");

			}

			//printf("3\n");







			//Write the whole before signal to file
			std::stringstream ss;
			ss << "RFIMBenchmark/BenchmarkRFIMVariableInterferorVariableEigenvectorRemoval/before" << h_numberOfBeamsToAdd << "BeamsToAdd"
					<< h_dimensionsToReduce << "DimensionsToReduce.txt";

			Utility_WriteSignalMatrixToFile( ss.str(), h_signal, h_numberOfSamples, h_valuesPerSample);

			//printf("4\n");

			//copy the signal back to the device
			cudaMemcpy(d_signal, h_signal, signalByteSize, cudaMemcpyHostToDevice);


			//Do the RFIM
			RFIMMemoryStruct* RFIM = RFIMMemoryStructCreate(h_valuesPerSample, h_numberOfSamples, h_dimensionsToReduce, h_batchSize, h_numberOfCudaStreams);
			RFIMRoutine(RFIM, d_signal, d_filteredSignal);




			//Copy the results to the host
			cudaMemcpy(h_filteredSignal, d_filteredSignal, signalByteSize, cudaMemcpyDeviceToHost);




			//Write the after signal to file
			ss.str("");
			ss << "RFIMBenchmark/BenchmarkRFIMVariableInterferorVariableEigenvectorRemoval/after" << h_numberOfBeamsToAdd << "BeamsToAdd"
					<< h_dimensionsToReduce << "DimensionsToReduce.txt";

			Utility_WriteSignalMatrixToFile( ss.str(), h_filteredSignal, h_numberOfSamples, h_valuesPerSample);


			//printf("5\n");

			//Compute important stats
			float h_totalVarianceAfter = Utility_Variance(h_filteredSignal, h_valuesPerSample * h_numberOfSamples);
			float* h_subSignalVarianceAfter = Utility_SubSignalVariance(h_filteredSignal, h_valuesPerSample, h_numberOfSamples);

			//TODO: Does it make sense to compute the signal to noise of a sine wave? I guess it does to see how strong it is?
			//Signal to noise of the interferor waves
			float* h_signalToNoises;
			cudaMallocHost(&h_signalToNoises, sizeof(float) * h_numberOfBeamsToAdd);

			float** h_correlationCoefficents;
			cudaMallocHost(&h_correlationCoefficents, sizeof(float*) * h_numberOfBeamsToAdd);

			//Compute signal to noise of each sine wave and the subSignalCorrelationCoefficents for each sine wave with every beam
			for(uint64_t currentIndex = 0; currentIndex < h_numberOfBeamsToAdd; ++currentIndex)
			{

				//Calculate the signal to noise of the current sine wave
				h_signalToNoises[currentIndex] = Utility_SignalToNoiseRatio(sineWaves[currentIndex],
						sineWaveSignalLength, h_sineWaveAmplitudes[currentIndex]);


				//Allocate space for each set of coefficents
				cudaMallocHost(h_correlationCoefficents + currentIndex, sizeof(float) * h_valuesPerSample);

				//Calculate the current set of correlation coefficents
				h_correlationCoefficents[currentIndex] = Utility_CoefficentOfCrossCorrelation(h_filteredSignal, sineWaves[currentIndex], h_valuesPerSample, h_numberOfSamples, sineWaveSignalLength);

			}


			//printf("6\n");

			//Copy the largest eigenvalue over
			float h_largestEigenvalue;
			cudaMemcpy(&h_largestEigenvalue, RFIM->d_S, sizeof(float), cudaMemcpyDeviceToHost);


			//Write the results to a file

			ss.str("");
			ss << "RFIMBenchmark/BenchmarkRFIMVariableInterferorVariableEigenvectorRemoval/statsFile" << h_numberOfBeamsToAdd << "BeamsToAdd"
					<< h_dimensionsToReduce << "DimensionsToReduce.txt";


			std::string statsFilename = ss.str();

			//Write all this stuff to disk
			FILE* statsFile = std::fopen(statsFilename.c_str(), "w");

			if(statsFile == NULL)
			{
				fprintf(stderr, "BenchmarkRFIMConstantInterferor: Unable to open statsFile for writing\n");
				exit(1);
			}

			//printf("7\n");

			//Start writing data to the file
			fprintf(statsFile, "Signal Info\n");
			fprintf(statsFile, "Number of samples: %llu\n", h_numberOfSamples);
			fprintf(statsFile, "White noise mean: %f\n", whiteNoiseMean);
			fprintf(statsFile, "White noise standard deviation: %f\n\n", whiteNoiseStdDev);


			fprintf(statsFile, "Interference Info\n");

			for(uint64_t currentIndex = 0; currentIndex < h_numberOfBeamsToAdd; ++currentIndex)
			{
				fprintf(statsFile, "\tInterference sine wave %llu info\n", currentIndex);
				fprintf(statsFile, "\t\tFrequency: %f\n", h_sineWaveFrequencies[currentIndex]);
				fprintf(statsFile, "\t\tAmplitude: %f\n", h_sineWaveAmplitudes[currentIndex]);
				fprintf(statsFile, "\t\tSignal to Noise: %f\n", h_signalToNoises[currentIndex]);
				fprintf(statsFile, "\t\tNumber of samples: %llu\n\n", h_numberOfSamples);
			}


			//printf("8\n");

			fprintf(statsFile, "Eigenvectors Info\n");
			fprintf(statsFile, "Dimensions removed: %llu\n", h_dimensionsToReduce);
			fprintf(statsFile, "Largest eigenvalue: %f\n\n", h_largestEigenvalue);

			//printf("8.1\n");


			fprintf(statsFile, "Variance Info\n");
			fprintf(statsFile, "Total variance before: %f\n", h_totalVarianceBefore);
			for(uint64_t i = 0; i < h_valuesPerSample; ++i)
			{
				fprintf(statsFile, "Beam %llu variance before: %f\n", i, h_subSignalVarianceBefore[i]);
			}


			//printf("8.2\n");

			fprintf(statsFile, "\nTotal variance after: %f\n", h_totalVarianceAfter);
			for(uint64_t i = 0; i < h_valuesPerSample; ++i)
			{
				fprintf(statsFile, "Beam %llu variance after: %f\n", i, h_subSignalVarianceAfter[i]);
			}

			//printf("8.3\n");

			fprintf(statsFile, "\n");

			fprintf(statsFile, "Correlation Coefficent Info\n");
			for(uint64_t currentIndex = 0; currentIndex < h_numberOfBeamsToAdd; ++currentIndex)
			{
				for(uint64_t currentCorrelationIndex = 0; currentCorrelationIndex < h_valuesPerSample; ++currentCorrelationIndex)
				{
					//printf("8.4\n");
					fprintf(statsFile, "\tSine wave %llu correlation with beam %llu: %f\n", currentIndex,
							currentCorrelationIndex, h_correlationCoefficents[currentIndex][currentCorrelationIndex]);
				}

				fprintf(statsFile, "\n");

			}


			fprintf(statsFile, "\n");

			std::fclose(statsFile);


			//printf("9\n");
			//Free everything


			cudaFree(d_signal);
			cudaFree(d_filteredSignal);

			cudaFreeHost(h_signal);
			cudaFreeHost(h_filteredSignal);

			//Free sine waves
			for(uint64_t currentIndex = 0; currentIndex < h_numberOfBeamsToAdd; ++currentIndex)
			{
				cudaFreeHost(sineWaves[currentIndex]);
			}
			cudaFreeHost(sineWaves);
			cudaFreeHost(h_sineWaveAmplitudes);
			cudaFreeHost(h_sineWaveFrequencies);

			cudaFreeHost(h_signalToNoises);

			cudaFreeHost(h_subSignalVarianceBefore);
			cudaFreeHost(h_subSignalVarianceAfter);

			//Free the correlation coefficents
			for(uint64_t currentIndex = 0; currentIndex < h_valuesPerSample; ++currentIndex)
			{
				cudaFreeHost(h_correlationCoefficents[currentIndex]);
			}
			cudaFreeHost(h_correlationCoefficents);


			RFIMMemoryStructDestroy(RFIM);

			//printf("10\n");

		}

	}


	curandDestroyGenerator(rngGen);



}




void BenchmarkRFIMDualInterferor()
{
	//Signal
	uint64_t h_valuesPerSample = 13;
	uint64_t h_numberOfSamples  = 1 << 15;
	uint64_t h_dimensionsToReduce;
	uint64_t h_batchSize = 1;
	uint64_t h_numberOfCudaStreams = 1;
	uint64_t h_numberOfThreads = 1;
	uint64_t h_numberOfBeamsToAdd = 13;

	float whiteNoiseMean = 0.0f;
	float whiteNoiseStdDev = 1.0f;

	float sineWaveMean = 3;
	float sineWaveStdDev = 6;



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


	printf("1\n");


	//Generate 13 beam signal and space for it's result
	uint64_t signalLength = h_valuesPerSample * h_numberOfSamples * h_batchSize * h_numberOfThreads;
	uint64_t signalByteSize = sizeof(float) * signalLength;

	float* d_signal;
	d_signal = Device_GenerateWhiteNoiseSignal(&rngGen, h_valuesPerSample, h_numberOfSamples, h_batchSize, h_numberOfThreads);



	//Generate the sine waves with random frequencies
	uint64_t sineWaveLength = h_numberOfSamples;
	uint64_t sineWaveByteSize = sizeof(float) * h_numberOfSamples;
	uint64_t totalSineWaveByteSize = sineWaveByteSize * h_valuesPerSample;

	//Get random frequency values
	float sineWave1Freq = Utility_GenerateSingleWhiteNoiseValueHost(sineWaveMean, sineWaveStdDev);
	float sineWave2Freq = Utility_GenerateSingleWhiteNoiseValueHost(sineWaveMean, sineWaveStdDev);

	float* h_sineWave1; //Base sine wave
	float* h_sineWave2; //Base sine wave

	//Generate the actual sine waves, with amplitude of one
	h_sineWave1 = Utility_GenerateSineWaveHost(sineWaveLength, sineWave1Freq, 1.0f);
	h_sineWave2 = Utility_GenerateSineWaveHost(sineWaveLength, sineWave2Freq, 1.0f);

	float* h_sineWave1Amplitudes;
	float* h_sineWave2Amplitudes;

	cudaMallocHost(&h_sineWave1Amplitudes, sizeof(float) * h_valuesPerSample);
	cudaMallocHost(&h_sineWave2Amplitudes, sizeof(float) * h_valuesPerSample);

	float* h_allSineWaves1; //All thirteen versions of sine wave 1 used
	float* h_allSineWaves2; //All thirteen versions of sine wave 2 used

	cudaMallocHost(&h_allSineWaves1, totalSineWaveByteSize);
	cudaMallocHost(&h_allSineWaves2, totalSineWaveByteSize);


	printf("2\n");

	//Generate the random amplitude sine waves to use, from the base sine waves
	for(uint64_t i = 0; i < h_valuesPerSample; ++i)
	{
		h_sineWave1Amplitudes[i] = Utility_GenerateSingleWhiteNoiseValueHost(sineWaveMean, sineWaveStdDev);
		h_sineWave2Amplitudes[i] = Utility_GenerateSingleWhiteNoiseValueHost(sineWaveMean, sineWaveStdDev);

		/*
		for(uint64_t j = 0; j < h_numberOfSamples; ++j)
		{
			//Base index + beam base offset + line up with the actual beams index
			h_allSineWaves1[ (i * sineWaveLength) + (j * h_valuesPerSample) + i ] =  h_sineWave1[j] * h_sineWave1Amplitudes[i];
			h_allSineWaves2[ (i * sineWaveLength) + (j * h_valuesPerSample) + i ] =  h_sineWave2[j] * h_sineWave2Amplitudes[i];
		}
		*/


		for(uint64_t j = 0; j < h_numberOfSamples; ++j)
		{
			//Add random amplitudes to each sine wave that we are going to use
			h_allSineWaves1[ (i * sineWaveLength) + j] = h_sineWave1[j] * h_sineWave1Amplitudes[i];
			h_allSineWaves2[ (i * sineWaveLength) + j] = h_sineWave2[j] * h_sineWave2Amplitudes[i];
		}



	}

	printf("3\n");


	//Add the signal to each beam, do this twice and remove 1, then 2 eigenvectors and see what happens

	for(uint64_t i = 0; i < 2; ++i)
	{

		h_dimensionsToReduce = i + 1;


		//Use the same white noise signal as a base each time, add stuff to d_currentSignal rather than d_signal
		float* d_currentSignal;
		cudaMalloc(&d_currentSignal, signalByteSize);

		//Copy the original signal to the host
		float* h_signal;
		cudaMallocHost(&h_signal, signalByteSize);
		cudaMemcpy(h_signal, d_signal, signalByteSize, cudaMemcpyDeviceToHost);



		//TODO: CALCULATE THE BEFORE VARIANCE
		float h_totalVarianceBefore = Utility_Variance(h_signal, h_valuesPerSample * h_numberOfSamples);
		float* h_subSignalVarianceBefore = Utility_SubSignalVariance(h_signal, h_valuesPerSample, h_numberOfSamples);

		printf("4\n");

		//Add both the sine waves to each beam, with their different amplitudes each time
		//For each beam
		for(uint64_t currentIndex = 0; currentIndex < h_valuesPerSample; ++currentIndex)
		{

			//For each sample in that beam
			for(uint64_t currentValueIndex = 0; currentValueIndex < h_numberOfSamples; ++currentValueIndex)
			{
				uint64_t currentBeamIndex = (currentValueIndex * h_valuesPerSample) + currentIndex;
				uint64_t currentSineWaveIndex = (currentIndex * sineWaveLength) + currentValueIndex;

				//Add both sine wave to this beam
				h_signal[ currentBeamIndex ] += h_allSineWaves1[ currentSineWaveIndex ];
				h_signal[ currentBeamIndex ] += h_allSineWaves2[ currentSineWaveIndex ];
			}

		}


		printf("5\n");


		//Write the whole before signal to a file
		std::stringstream ss;
		ss << "RFIMBenchmark/BenchmarkRFIMDualInterferor/before" << 13 << "BeamsToAdd"
				<< h_dimensionsToReduce << "DimensionsToReduce.txt";

		Utility_WriteSignalMatrixToFile( ss.str(), h_signal, h_numberOfSamples, h_valuesPerSample);


		//Copy this new signal back to the device
		cudaMemcpy(d_currentSignal, h_signal, signalByteSize, cudaMemcpyHostToDevice);


		float* d_filteredSignal;
		cudaMalloc(&d_filteredSignal, signalByteSize);

		//Run the routine once and remove one eigenvector
		RFIMMemoryStruct* RFIM = RFIMMemoryStructCreate(h_valuesPerSample, h_numberOfSamples, h_dimensionsToReduce, h_batchSize, h_numberOfCudaStreams);
		RFIMRoutine(RFIM, d_currentSignal, d_filteredSignal);


		float* h_filteredSignal;
		cudaMallocHost(&h_filteredSignal, signalByteSize);
		cudaMemcpy(h_filteredSignal, d_filteredSignal, signalByteSize, cudaMemcpyDeviceToHost);


		//Write the whole after signal to a file
		ss.str("");
		ss << "RFIMBenchmark/BenchmarkRFIMDualInterferor/after" << h_numberOfBeamsToAdd << "BeamsToAdd"
				<< h_dimensionsToReduce << "DimensionsToReduce.txt";

		Utility_WriteSignalMatrixToFile( ss.str(), h_filteredSignal, h_numberOfSamples, h_valuesPerSample);



		printf("6\n");


		//Compute a bunch of stats
		float h_totalVarianceAfter = Utility_Variance(h_filteredSignal, h_valuesPerSample * h_numberOfSamples);
		float* h_subSignalVarianceAfter = Utility_SubSignalVariance(h_filteredSignal, h_valuesPerSample, h_numberOfSamples);

		//TODO: Does it make sense to compute the signal to noise of a sine wave? I guess it does to see how strong it is?
		//Signal to noise of the interferor waves
		//float* h_signalToNoises;
		//cudaMallocHost(&h_signalToNoises, sizeof(float) * h_numberOfBeamsToAdd);

		float** h_sineWave1CorrelationCoefficents;
		float** h_sineWave2CorrelationCoefficents;
		cudaMallocHost(&h_sineWave1CorrelationCoefficents, sizeof(float*) * h_valuesPerSample);
		cudaMallocHost(&h_sineWave2CorrelationCoefficents, sizeof(float*) * h_valuesPerSample);


		printf("7\n");



		//Compute the subSignalCorrelationCoefficents for each sine wave with every beam
		for(uint64_t currentIndex = 0; currentIndex < h_valuesPerSample; ++currentIndex)
		{
			h_sineWave1CorrelationCoefficents[currentIndex] = Utility_CoefficentOfCrossCorrelation(h_filteredSignal,
					h_allSineWaves1 + (currentIndex * h_numberOfSamples), h_valuesPerSample, h_numberOfSamples,
					sineWaveLength);
			h_sineWave2CorrelationCoefficents[currentIndex] = Utility_CoefficentOfCrossCorrelation(h_filteredSignal,
					h_allSineWaves2 + (currentIndex * h_numberOfSamples), h_valuesPerSample, h_numberOfSamples,
					sineWaveLength);
		}


		printf("8\n");

		//Copy the largest eigenvalue over
		float h_largestEigenvalue;
		cudaMemcpy(&h_largestEigenvalue, RFIM->d_S, sizeof(float), cudaMemcpyDeviceToHost);


		//Write the stats to file
		ss.str("");
		ss << "RFIMBenchmark/BenchmarkRFIMDualInterferor/statsFile" << h_numberOfBeamsToAdd << "BeamsToAdd"
							<< h_dimensionsToReduce << "DimensionsToReduce.txt";


		std::string statsFilename = ss.str();

		//Write all this stuff to disk
		FILE* statsFile = std::fopen(statsFilename.c_str(), "w");

		if(statsFile == NULL)
		{
			fprintf(stderr, "BenchmarkRFIMDualInterferor: Unable to open statsFile for writing\n");
			exit(1);
		}


		fprintf(statsFile, "Signal Info\n");
		fprintf(statsFile, "Number of samples: %llu\n", h_numberOfSamples);
		fprintf(statsFile, "White noise mean: %f\n", whiteNoiseMean);
		fprintf(statsFile, "White noise standard deviation: %f\n\n", whiteNoiseStdDev);


		printf("9\n");


		fprintf(statsFile, "Interference Info\n");

		//Sine wave one
		fprintf(statsFile, "Sine wave 1\n");
		for(uint64_t currentIndex = 0; currentIndex < h_numberOfBeamsToAdd; ++currentIndex)
		{
			fprintf(statsFile, "\tInterference sine wave1, added to beam %llu info\n", currentIndex);
			fprintf(statsFile, "\t\tFrequency: %f\n", sineWave1Freq);
			fprintf(statsFile, "\t\tAmplitude: %f\n", h_sineWave1Amplitudes[currentIndex]);
			fprintf(statsFile, "\t\tNumber of samples: %llu\n\n", h_numberOfSamples);
		}

		printf("10\n");


		fprintf(statsFile, "Sine wave 2\n");
		for(uint64_t currentIndex = 0; currentIndex < h_numberOfBeamsToAdd; ++currentIndex)
		{
			fprintf(statsFile, "\tInterference sine wave2, added to beam %llu info\n", currentIndex);
			fprintf(statsFile, "\t\tFrequency: %f\n", sineWave2Freq);
			fprintf(statsFile, "\t\tAmplitude: %f\n", h_sineWave2Amplitudes[currentIndex]);
			fprintf(statsFile, "\t\tNumber of samples: %llu\n\n", h_numberOfSamples);
		}

		printf("11\n");


		fprintf(statsFile, "Eigenvectors Info\n");
		fprintf(statsFile, "Dimensions removed: %llu\n", h_dimensionsToReduce);
		fprintf(statsFile, "Largest eigenvalue: %f\n\n", h_largestEigenvalue);



		fprintf(statsFile, "Variance Info\n");
		fprintf(statsFile, "Total variance before: %f\n", h_totalVarianceBefore);
		for(uint64_t currentIndex = 0; currentIndex < h_valuesPerSample; ++currentIndex)
		{
			fprintf(statsFile, "Beam %llu variance before: %f\n", currentIndex, h_subSignalVarianceBefore[currentIndex]);
		}

		printf("12\n");

		fprintf(statsFile, "\n");
		fprintf(statsFile, "Total variance after: %f\n", h_totalVarianceAfter);
		for(uint64_t currentIndex = 0; currentIndex < h_valuesPerSample; ++currentIndex)
		{
			fprintf(statsFile, "Beam %llu variance after: %f\n", currentIndex, h_subSignalVarianceAfter[currentIndex]);
		}


		printf("13\n");

		//Correlation coefficents
		fprintf(statsFile, "\nSine Wave correlation coefficents\n");
		//For each sine wave
		for(uint64_t currentSineWaveIndex = 0; currentSineWaveIndex < h_valuesPerSample; ++currentSineWaveIndex)
		{
			//Print it's correlation with each beam
			for(uint64_t currentBeamindex = 0; currentBeamindex < h_valuesPerSample; ++currentBeamindex)
			{
				fprintf(statsFile, "Correlation of base sine wave [1, %llu] with beam %llu: %f\n",
						currentSineWaveIndex, currentBeamindex, h_sineWave1CorrelationCoefficents[currentSineWaveIndex][currentBeamindex]);

				fprintf(statsFile, "Correlation of base sine wave [2, %llu] with beam %llu: %f\n",
								currentSineWaveIndex, currentBeamindex, h_sineWave2CorrelationCoefficents[currentSineWaveIndex][currentBeamindex]);
			}
		}


		printf("14\n");

		std::fclose(statsFile);

		//Free everything from this iteration
		cudaFree(d_currentSignal);
		cudaFree(d_filteredSignal);


		cudaFreeHost(h_signal);
		cudaFreeHost(h_filteredSignal);
		cudaFreeHost(h_subSignalVarianceBefore);
		cudaFreeHost(h_subSignalVarianceAfter);

		for(uint64_t currentIndex = 0; currentIndex < h_valuesPerSample; ++currentIndex)
		{
			cudaFreeHost(h_sineWave1CorrelationCoefficents[currentIndex]);
			cudaFreeHost(h_sineWave2CorrelationCoefficents[currentIndex]);
		}

		cudaFreeHost(h_sineWave1CorrelationCoefficents);
		cudaFreeHost(h_sineWave2CorrelationCoefficents);

		RFIMMemoryStructDestroy(RFIM);

		printf("15\n");

	}



	//Free everything
	cudaFree(d_signal);

	cudaFreeHost(h_sineWave1);
	cudaFreeHost(h_sineWave2);

	cudaFreeHost(h_sineWave1Amplitudes);
	cudaFreeHost(h_sineWave2Amplitudes);

	printf("16\n");

}




