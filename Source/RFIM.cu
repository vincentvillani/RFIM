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

void RFIMRoutine(RFIMMemoryStruct* RFIMStruct, float* d_columnMajorSignalMatrix, float* d_columnMajorFilteredSignalMatrix)
{

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

	//Project the signal against the reduced eigenvector matrix and back again to the original dimensions
	Device_EigenReductionAndFiltering(RFIMStruct, d_columnMajorSignalMatrix, d_columnMajorFilteredSignalMatrix);

}

