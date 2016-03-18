/*
 * UtilityFunctions.cu
 *
 *  Created on: 17/03/2016
 *      Author: vincentvillani
 */

#include "../Header/UtilityFunctions.h"
#include "../Header/RFIMHelperFunctions.h"
#include "../Header/CudaUtilityFunctions.h"

/*
//Write a device signal matrix to a file and graph it
void Utility_GraphData(float* d_signalMatrix, uint64_t rows, uint64_t columns, bool transpose)
{

	uint64_t h_rows;
	uint64_t h_columns;
	float* h_rowMajorMatrixToGraph;
	float* d_matrixToGraph = d_signalMatrix;

	//Transpose if neccessary
	if(transpose)
	{
		d_matrixToGraph = Device_MatrixTranspose(d_signalMatrix, rows, columns);
		h_rows = columns;
		h_columns = rows;
	}
	else
	{
		h_rows = rows;
		h_columns = columns;
	}


	h_rowMajorMatrixToGraph = CudaUtility_CopySignalToHost(d_matrixToGraph, rows * columns * sizeof(float));


	//Write the signal to a file
	Utility_WriteSignalMatrixToFile(std::string("Graph.txt"), h_rowMajorMatrixToGraph, h_rows, h_columns);

	//Open up GNUPlot and graph a heatmap


	if(transpose)
		cudaFree(d_matrixToGraph);

	free(h_rowMajorMatrixToGraph);
}
*/

//Write a host signal matrix to a file
void Utility_WriteSignalMatrixToFile(const std::string filename, const float* h_rowMajorSignalMatrix, uint64_t rows, uint64_t columns)
{


	FILE* signalFile = fopen(filename.c_str(), "w");

	if(signalFile == NULL)
	{
		fprintf(stderr, "WriteSignalMatrixToFile: failed to open %s file\n", filename.c_str());
		exit(1);
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
