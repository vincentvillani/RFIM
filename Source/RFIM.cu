/*
 * RFIM.cu
 *
 *  Created on: 17/03/2016
 *      Author: vincentvillani
 */

#include "../Header/RFIM.h"

#include <stdio.h>

float* RFIM(RFIMMemoryStruct* RFIMStruct, float* d_columnMajorSignalMatrix, uint32_t h_valuesPerSample, uint32_t h_numberOfSamples)
{
	//The filtered signal to return
	float* d_filteredSignal;

	//if these are mismatched the program will probably crash
	if(RFIMStruct->h_valuesPerSample != h_valuesPerSample || RFIMStruct->h_numberOfSamples != h_numberOfSamples)
	{
		fprintf(stderr, "RFIM: RFIMStruct->h_valuesPerSample != h_valuesPerSample OR RFIMStruct->h_numberOfSamples != h_numberOfSamples");
		//exit(1);
	}




	return d_filteredSignal;
}

