
#include "../Header/Benchmark.h"
#include "../Header/RFIM.h"

#include <stdio.h>
#include <sys/time.h>


double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}


void Benchmark(RFIMMemoryStruct* RFIM, float* d_signal, float* d_filteredSignal, uint32_t calculationNum, uint32_t iterations)
{
	//Start the timer
	double startTime = cpuSecond();

	//Run the benchmark iterations times. For a total of calculationNum * iterations executions of RFIMRoutine
	for(uint32_t iterationCounter = 0; iterationCounter < iterations; ++iterationCounter)
	{
		for(uint32_t calculationCounter = 0; calculationCounter < calculationNum; ++calculationCounter)
		{
			RFIMRoutine(RFIM, d_signal, d_filteredSignal);
		}
	}

	//end the timer
	double duration = cpuSecond() - startTime;

	//find the average time taken
	duration /= iterations;

	printf("Signal: (%u, %u)\nIterations: (%u, %u)\nAverage time: %f\n",
			RFIM->h_valuesPerSample, RFIM->h_numberOfSamples, calculationNum, iterations, duration);

}
