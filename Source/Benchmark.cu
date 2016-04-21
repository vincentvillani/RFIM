
#include "../Header/Benchmark.h"
#include "../Header/RFIM.h"
#include "../Header/RFIMHelperFunctions.h"
#include "../Header/RFIMMemoryStructComplex.h"
#include "../Header/UtilityFunctions.h"

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



void RFIMInstanceComplex(RFIMMemoryStructComplex* RFIM, cuComplex* d_signal, cuComplex* d_filteredSignal, uint64_t iterations)
{
	for(uint64_t i = 0; i < iterations; ++i)
	{
		RFIMRoutineComplex(RFIM, d_signal, d_filteredSignal);
	}
}



void Benchmark()
{

	//Benchmark
	uint64_t iterations = 50;

	//Signal
	uint64_t h_valuesPerSample = 13;
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



	for(uint64_t f = 0; f < 11; ++f)
	{

		//For each numberOfSamples value
		for(uint64_t i = 14; i < 15; ++i)
		{
			h_numberOfSamples = 1 << i;



			//For each batchSize
			for(uint64_t j = 0; j < 4; ++j)
			{

				h_batchSize = 1 << j;
				h_numberOfCudaStreams = 1 << j;



				for(uint64_t p = 0; p < 1; ++p)
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

		h_valuesPerSample *= 2;
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
	for(h_numberOfBeamsToAdd = 0; h_numberOfBeamsToAdd < h_valuesPerSample + 1; ++h_numberOfBeamsToAdd)
	{

		//Create an RFIMStruct
		RFIMMemoryStruct* RFIM = RFIMMemoryStructCreate(h_valuesPerSample, h_numberOfSamples,
				h_dimensionsToReduce, h_batchSize, h_numberOfCudaStreams);


		//Generate a signal
		uint64_t signalByteSize = sizeof(float) * h_valuesPerSample * h_numberOfSamples * h_batchSize * h_numberOfThreads;
		float* d_signal = Device_GenerateWhiteNoiseSignal(&rngGen, h_valuesPerSample, h_numberOfSamples,
				h_batchSize,  h_numberOfThreads, whiteNoiseMean, whiteNoiseStdDev);

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
		ss << "RFIMBenchmark/BenchmarkRFIMConstantInterferor/before" << h_numberOfBeamsToAdd << "StreamsToRemove.txt";


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
		ss << "RFIMBenchmark/BenchmarkRFIMConstantInterferor/after" << h_numberOfBeamsToAdd << "StreamsToRemove.txt";


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
		ss << "RFIMBenchmark/BenchmarkRFIMConstantInterferor/statsFile" << h_numberOfBeamsToAdd << "StreamsToRemove.txt";


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


}








