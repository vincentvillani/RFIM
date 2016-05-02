/*
 * UtilityFunctions.cu
 *
 *  Created on: 17/03/2016
 *      Author: vincentvillani
 */

#include "../Header/UtilityFunctions.h"
#include "../Header/RFIMHelperFunctions.h"

#include <cublas.h>
#include <random>
#include <math.h>
#include <string.h>
#include <chrono>


float* Utility_GenerateWhiteNoiseHost(uint64_t length, float mean, float stdDev)
{
	float* result;
	cudaMallocHost(&result, sizeof(float) * length);


	//Setup RNG generator
	std::default_random_engine generator;
	std::normal_distribution<float> distribution(mean, stdDev);


	//Generate the random numbers
	for(uint64_t i = 0; i < length; ++i)
	{
		result[i] = distribution(generator);
	}


	return result;

}


float* Utility_GenerateWhiteNoiseHostMalloc(uint64_t length, float mean, float stdDev)
{
	float* result = (float*)malloc(sizeof(float) * length);


	//Setup RNG generator
	std::default_random_engine generator;
	std::normal_distribution<float> distribution(mean, stdDev);


	//Generate the random numbers
	for(uint64_t i = 0; i < length; ++i)
	{
		result[i] = distribution(generator);
	}


	return result;

}


float Utility_GenerateSingleWhiteNoiseValueHost(float mean, float stdDev)
{
	std::default_random_engine generator;
	std::normal_distribution<float> distribution(mean, stdDev);

	//Get a seed value from the clock
	std::chrono::high_resolution_clock::time_point beginning = std::chrono::high_resolution_clock::now();

	std::chrono::high_resolution_clock::duration timeDifference = std::chrono::high_resolution_clock::now() - beginning;
	uint64_t seedValue = timeDifference.count();

	generator.seed(seedValue);

	return distribution(generator);
}



float* Utility_GenerateSineWaveHost(uint64_t length, float frequency, float amplitude)
{
	float* result;
	cudaMallocHost(&result, sizeof(float) * length);

	const float pi = 3.14159265359f;

	for(uint64_t i = 0; i < length; ++i)
	{
		result[i] = sinf( (2 * pi * frequency * i) / length) * amplitude;
	}

	return result;
}




float Utility_Mean(float* h_signal, uint64_t signalLength)
{
	float result = 0;

	for(uint64_t i = 0; i < signalLength; ++i)
	{
		result += h_signal[i];
	}

	return result / signalLength;
}



float* Utility_SubSignalMean(float* h_signal, uint64_t h_valuesPerSample, uint64_t h_numberOfSamples)
{
	float* result;
	uint64_t resultByteSize = sizeof(float) * h_valuesPerSample;
	cudaMallocHost(&result, resultByteSize);
	memset(result, 0, resultByteSize);

	//Find the mean for each sub-signal
	for(uint64_t i = 0; i < h_numberOfSamples; ++i)
	{
		for(uint64_t j = 0; j < h_valuesPerSample; ++j)
		{
			result[j] = h_signal[ (i * h_valuesPerSample) + j];
		}
	}


	//Divide all results by h_numberOfSamples
	for(uint64_t i = 0; i < h_valuesPerSample; ++i)
	{
		result[i] /= h_numberOfSamples;
	}

	return result;
}





float Utility_SignalToNoiseRatio(float* h_signal, uint64_t signalLength, float signalAmplitude)
{
	//SNR = signalAmplitude / RootMeanSquare (RMS)
	float RMS = 0;

	for(uint64_t i = 0; i < signalLength; ++i)
	{
		RMS += h_signal[i] * h_signal[i];
	}

	RMS = RMS / signalLength;
	RMS = sqrtf(RMS);

	return signalAmplitude / RMS;
}


float* Utility_SubSignalSignalToNoiseRatio(float* h_signal, uint64_t h_valuesPerSample, uint64_t h_numberOfSamples, float signalAmplitude)
{
	float* result;
	uint64_t resultByteSize = sizeof(float) * h_valuesPerSample;
	cudaMallocHost(&result, resultByteSize);
	memset(result, 0, resultByteSize);

	//SNR = signalAmplitude / RootMeanSquare (RMS)
	float* RMS;
	uint64_t RMSByteSize = sizeof(float) * h_valuesPerSample;
	cudaMallocHost(&RMS, RMSByteSize);
	memset(RMS, 0, RMSByteSize);

	//For each sample
	for(uint64_t i = 0; i < h_numberOfSamples; ++i)
	{
		//For each beam
		for(uint64_t j = 0; j < h_valuesPerSample; ++j)
		{
			//Calculate the RMS for each beam
			RMS[j] += h_signal[ (i * h_valuesPerSample) + j] * h_signal[ (i * h_valuesPerSample) + j];
		}
	}


	//Divide by 1/n and sqrt
	for(uint64_t i = 0; i < h_valuesPerSample; ++i)
	{
		RMS[i] /= h_numberOfSamples;
		RMS[i] = sqrtf(RMS[i]);
		result[i] = signalAmplitude / RMS[i];
	}

	cudaFreeHost(RMS);


	return result;

}


float* Utility_CoefficentOfCrossCorrelation(float* h_multiplexedSignal, float* h_secondSignal,
		uint64_t h_valuesPerSample, uint64_t h_numberOfSamples)
{

	float* result;
	int64_t resultByteLength = sizeof(float) * h_valuesPerSample;
	cudaMallocHost(&result, resultByteLength);
	//memset(result, 0, resultByteLength);

	//Calculate the means
	float h_secondSignalMean = Utility_Mean(h_secondSignal, h_numberOfSamples);
	float* h_subSignalMeans = Utility_SubSignalMean(h_multiplexedSignal, h_valuesPerSample, h_numberOfSamples);


	float* numerators;
	float* denominators;

	cudaMallocHost(&numerators, resultByteLength);
	cudaMallocHost(&denominators, resultByteLength);

	memset(numerators, 0, resultByteLength);
	memset(denominators, 0, resultByteLength);


	//For the current sub-beam/signal
	for(uint64_t currentBeamIndex = 0; currentBeamIndex < h_valuesPerSample; ++currentBeamIndex)
	{
		//Correlate samples from the second signal with the current sub-beam/signal
		for(uint64_t currentSampleIndex = 0; currentSampleIndex < h_numberOfSamples; ++currentSampleIndex)
		{

			float currentYValue = h_secondSignal[currentSampleIndex] - h_secondSignalMean;
			float currentXValue = (h_multiplexedSignal[currentSampleIndex * h_valuesPerSample + currentBeamIndex]) -
					h_subSignalMeans[currentBeamIndex];

			//Calculate the numerator and the denominator
			numerators[currentBeamIndex] += currentXValue * currentYValue;
			denominators[currentBeamIndex] += sqrtf(currentXValue * currentXValue) *
					sqrtf(currentYValue * currentYValue);

		}
	}

	/*
	//Calculate the coefficent of  cross correlation for each beam
	for(uint64_t i = 0; i < h_numberOfSamples; ++i)
	{

		float currentYValue = h_secondSignal[i] - h_secondSignalMean;

		//Add the current sample from each beam to calculate the numerator and denominator
		for(uint64_t j = 0; j < h_valuesPerSample; ++j)
		{
			float currentXValue = (h_multiplexedSignal[ (i * h_valuesPerSample) + j]) - h_subSignalMeans[j];

			//Calculate the numerator
			numerators[j] += (currentXValue - currentYValue);

			//Calculate the denominator
			denominators[j] += sqrtf(currentXValue * currentXValue) * sqrtf(currentYValue * currentYValue);
		}

	}
	*/


	//Do the final calculations to get the correlation coefficents
	for(uint64_t i = 0; i < h_valuesPerSample; ++i)
	{
		result[i] = numerators[i] / denominators[i];
	}


	//Free things we don't need anymore
	cudaFreeHost(h_subSignalMeans);
	cudaFreeHost(numerators);
	cudaFreeHost(denominators);


	return result;

}




float Utility_Variance(float* h_signal, uint64_t signalLength)
{
	float variance = 0;

	for(uint64_t i = 0; i < signalLength; ++i)
	{
		variance += h_signal[i] * h_signal[i];
	}

	return variance / signalLength;
}




float* Utility_SubSignalVariance(float* h_signal, uint64_t h_valuesPerSample, uint64_t h_numberOfSamples)
{
	float* result;
	uint64_t resultByteSize = sizeof(float) * h_valuesPerSample;
	cudaMallocHost(&result, resultByteSize);
	memset(result, 0, resultByteSize);

	//For each sample
	for(uint64_t i = 0; i < h_numberOfSamples; ++i)
	{
		//For each beam
		for(uint64_t j = 0; j < h_valuesPerSample; ++j)
		{
			//Calculate the variance for each beam
			result[j] += h_signal[ (i * h_valuesPerSample) + j] * h_signal[ (i * h_valuesPerSample) + j];
		}
	}


	//Divide by the signal length
	for(uint64_t i = 0; i < h_valuesPerSample; ++i)
	{
		result[i] /= h_numberOfSamples;
	}

	return result;
}






//Write a host signal matrix to a file
void Utility_WriteSignalMatrixToFile(const std::string filename, float* h_rowMajorSignalMatrix, uint64_t rows, uint64_t columns)
{


	FILE* signalFile = fopen(filename.c_str(), "w");

	if(signalFile == NULL)
	{
		fprintf(stderr, "WriteSignalMatrixToFile: failed to open %s file\n", filename.c_str());
		//exit(1);
	}


	for(uint32_t currentRow = 0; currentRow < rows; ++currentRow)
	{
		for(uint32_t currentCol = 0; currentCol < columns; ++currentCol)
		{
			//If last item in the column, write it without the " "
			if(currentCol == columns - 1)
				fprintf(signalFile, "%f", h_rowMajorSignalMatrix[currentRow * columns + currentCol] );
			else
				fprintf(signalFile, "%f ", h_rowMajorSignalMatrix[currentRow * columns + currentCol] );
		}

		//Print a newline for each row except the last one
		if(currentRow != currentRow - 1)
			fprintf(signalFile, "\n");
	}


	fclose(signalFile);
}


void Utility_DeviceWriteSignalMatrixToFile(const std::string filename, float* d_rowMajorSignalMatrix, uint64_t rows, uint64_t columns, bool transpose)
{
	/*
	uint32_t matrixByteSize = sizeof(float) * rows * columns;

	//Copy the matrix to the device
	float* h_rowMajorSignalMatrix = (float*)malloc(matrixByteSize);
	float* d_transposedMatrix = d_rowMajorSignalMatrix;
	*/
	/*
	cublasHandle_t cublasHandle;


	if(transpose)
	{
		cublasCreate_v2(&cublasHandle);
		cudaMalloc(&d_transposedMatrix, matrixByteSize);

		//Transpose the matrix
		Device_MatrixTranspose(&cublasHandle, d_rowMajorSignalMatrix, d_transposedMatrix, rows, columns);
	}
	*/
	/*
	CudaUtility_CopySignalToHost(d_transposedMatrix, &h_rowMajorSignalMatrix, sizeof(float) * rows * columns);

	//Call the host version of this function
	Utility_WriteSignalMatrixToFile(filename, h_rowMajorSignalMatrix, rows, columns);

	free(h_rowMajorSignalMatrix);
	*/

	/*
	if(transpose)
	{
		cublasDestroy(cublasHandle);
		cudaFree(d_transposedMatrix);
	}
	*/
}


void Utility_PrintFilterbankMetadata(Filterbank* filterbank)
{

	printf("Number of Samples: %lu\n", filterbank->get_nsamps());
	printf("Number of Channels: %f\n", filterbank->get_nchans());
	printf("Bits per Sample: %f\n", filterbank->get_nbits());
	printf("Sample time: %f\n", filterbank->get_tsamp());
	printf("First channel Frequency: %f\n", filterbank->get_fch1());
	printf("Per channel bandwidth: %f\n", filterbank->get_foff());

}


