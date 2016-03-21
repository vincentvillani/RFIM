/*
 * UtilityFunctions.cu
 *
 *  Created on: 17/03/2016
 *      Author: vincentvillani
 */

#include "../Header/UtilityFunctions.h"
#include "../Header/RFIMHelperFunctions.h"
#include "../Header/CudaUtilityFunctions.h"


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
