
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand.h>
#include <cublas.h>

#include <stdint.h>

#include "../Header/Kernels.h"
#include "../Header/UnitTests.h"
#include "../Header/CudaMacros.h"



int main(int argc, char **argv)
{
	//Run all the unit tests
	RunAllUnitTests();


	//1. Generate a signal on the device
	//----------------------------------
	uint64_t h_valuesPerSample = 26;
	uint64_t h_numberOfSamples = 1024;

	float* d_whiteNoiseSignalMatrix = Device_GenerateWhiteNoiseSignal(h_valuesPerSample, h_numberOfSamples);

	//----------------------------------

	//2.Calculate the covariance matrix of this signal
	//----------------------------------
	float* d_covarianceMatrix = Device_CalculateCovarianceMatrix(d_whiteNoiseSignalMatrix, h_valuesPerSample, h_numberOfSamples);

	//----------------------------------

	//3. Graph the covariance matrix
	//----------------------------------
	//http://gnuplot.sourceforge.net/demo/heatmaps.html

	/*
	 *
				void SignalWriteToTextFile(const std::string filename, const Signal* signal)
				{
					FILE* signalFile = fopen(filename.c_str(), "w");

					if(signalFile == NULL)
						return;

					uint32_t i = 0;

					for(; i < signal->sampleLength - 1; ++i)
					{
						fprintf(signalFile, "%u %f\n", i, signal->samples[i]);
					}

					//print last line to the text file without the newline
					fprintf(signalFile, "%u %f", i, signal->samples[i]);


					fclose(signalFile);
				}


				void SignalGraph(const Signal* signal)
				{
					char filenameBuffer[50];
					sprintf(filenameBuffer, "TempSignal%u.txt", tempGraphNumber);
					tempGraphNumber++;


					SignalWriteToTextFile(filenameBuffer, signal);

					FILE* gnuplot;
					//gnuplot = popen("gnuplot -persist", "w"); //Linux
					gnuplot = popen("/usr/local/bin/gnuplot -persist", "w"); //OSX

					if (gnuplot == NULL)
						return;

					fprintf(gnuplot, "set xrange[0 : %u]\n", signal->sampleLength);
					fprintf(gnuplot, "set offset graph 0.01, 0.01, 0.01, 0.01\n");
					fprintf(gnuplot, "set samples %u\n", signal->sampleLength);
					fprintf(gnuplot, "plot \"%s\" with points pointtype 5  notitle\n", filenameBuffer);
					//fprintf(gnuplot, "plot \"%s\" with impulses lw 1 notitle\n", "TempGraphFile.txt");

					//Deletes the temp file
					//remove(filenameBuffer);

				}
	 *
	 *
	 */

	//----------------------------------


	//Free all memory
	//----------------------------------

	cudaFree(d_whiteNoiseSignalMatrix);
	cudaFree(d_covarianceMatrix);

	//----------------------------------

	return 0;
}
