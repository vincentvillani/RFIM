/*
 * RFIM.cu
 *
 *  Created on: 17/03/2016
 *      Author: vincentvillani
 */

#include "../Header/RFIM.h"

#include "../Header/CudaUtilityFunctions.h"
#include "../Header/UtilityFunctions.h"
#include "../Header/RFIMHelperFunctions.h"

#include <stdio.h>

float* RFIMRoutine(RFIMMemoryStruct* RFIMStruct, float* d_columnMajorSignalMatrix)
{

	//The filtered signal to return
	float* d_filteredSignal = NULL;


	//If we reduce everything, we will have nothing left...
	if(RFIMStruct->h_eigenVectorDimensionsToReduce >= RFIMStruct->h_valuesPerSample)
	{
		fprintf(stderr, "RFIMStruct->h_eigenVectorDimensionsToReduce >= RFIMStruct->h_valuesPerSample\n");
		exit(1);
	}


	//Calculate covariance matrix for this signal
	Device_CalculateCovarianceMatrix(RFIMStruct, d_columnMajorSignalMatrix);


	//Calculate the eigenvectors/values
	Device_EigenvalueSolver(RFIMStruct);

	//Device_EigenReductionAndFiltering(RFIMStruct, d_columnMajorSignalMatrix);



	return d_filteredSignal;
}

