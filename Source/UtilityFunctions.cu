/*
 * UtilityFunctions.cu
 *
 *  Created on: 17/03/2016
 *      Author: vincentvillani
 */

#include "../Header/UtilityFunctions.h"


//Write a device signal matrix to a file and graph it
void GraphData(const float* d_signalMatrix, uint64_t rows, uint64_t columns)
{

}

//Write a host signal matrix to a file
void WriteSignalMatrixToFile(const std::string filename, const float* h_signalMatrix, uint64_t rows, uint64_t columns)
{
	FILE* signalFile = fopen(filename.c_str(), "w");

	if(signalFile == NULL)
		return;

	uint32_t i = 0;

	//Stored as column-major, but write row by row
	//Write rows first


	/*
	for(; i < signal->sampleLength - 1; ++i)
	{
		fprintf(signalFile, "%u %f\n", i, signal->samples[i]);
	}

	//print last line to the text file without the newline
	fprintf(signalFile, "%u %f", i, signal->samples[i]);
	*/

	fclose(signalFile);
}



/*
 * void SignalWriteToTextFile(const std::string filename, const Signal* signal)
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
 */
