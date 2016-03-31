/*
 * UnitTests.cu
 *
 *  Created on: 10/03/2016
 *      Author: vincentvillani
 */


#include "../Header/UnitTests.h"

#include "../Header/CudaMacros.h"
#include "../Header/CudaUtilityFunctions.h"
#include "../Header/Kernels.h"
#include "../Header/RFIMHelperFunctions.h"
#include "../Header/UtilityFunctions.h"
#include "../Header/RFIMMemoryStruct.h"

#include <cublas.h>

#include <assert.h>
#include <cmath>
#include <string>


//Production tests
void MeanCublasProduction();
void CovarianceCublasProduction();
void EigendecompProduction();
void FilteringProduction();
//void TransposeProduction();
//void GraphProduction();





//-------------------------------------

//Production
//-------------------------------------

void MeanCublasProduction()
{

	uint32_t valuesPerSample = 3;
	uint32_t sampleNum = 2;

	RFIMMemoryStruct* RFIMStruct = RFIMMemoryStructCreate(valuesPerSample, sampleNum, 2);


	float* h_signal = (float*)malloc(sizeof(float) * valuesPerSample * sampleNum);
	float* d_signal;
	cudaMalloc(&d_signal, sizeof(float) * valuesPerSample * sampleNum);

	//Set the host signal
	for(uint32_t i = 0; i < valuesPerSample * sampleNum; ++i)
	{
		h_signal[i] = i + 1;
	}

	CudaUtility_CopySignalToDevice(h_signal, &d_signal,  sizeof(float) * valuesPerSample * sampleNum);

	//Calculate the mean matrix
	Device_CalculateMeanMatrix(RFIMStruct, d_signal);


	//Copy it back to the host
	//At this point d_upperTriangularCovarianceMatrix is the mean matrix
	float* h_meanMatrix = (float*)malloc(sizeof(float) * valuesPerSample * valuesPerSample);
	CudaUtility_CopySignalToHost(RFIMStruct->d_upperTriangularCovarianceMatrix, &h_meanMatrix, valuesPerSample * valuesPerSample * sizeof(float));

	/*
	//Print out the result
	for(uint32_t i = 0; i < valuesPerSample * valuesPerSample; ++i)
	{
		printf("final: %u: %f\n", i, h_meanMatrix[i]);
	}
	*/



	bool failed = false;

	if(fabsf(h_meanMatrix[0] - 6.25f) > 0.000001f)
		failed = true;
	if(fabsf(h_meanMatrix[1]) > 0.000001f)
		failed = true;
	if(fabsf(h_meanMatrix[2]) > 0.000001f)
		failed = true;
	if(fabsf(h_meanMatrix[3] - 8.75f) > 0.000001f)
		failed = true;
	if(fabsf(h_meanMatrix[4] - 12.25f) > 0.000001f)
		failed = true;
	if(fabsf(h_meanMatrix[5]) > 0.000001f)
		failed = true;
	if(fabsf(h_meanMatrix[6] - 11.25f) > 0.000001f)
		failed = true;
	if(fabsf(h_meanMatrix[7] - 15.75f) > 0.000001f)
		failed = true;
	if(fabsf(h_meanMatrix[8] - 20.25f) > 0.000001f)
		failed = true;


	if(failed)
	{
		fprintf(stderr, "MeanCublasProduction failed!\n");
		exit(1);
	}

	RFIMMemoryStructDestroy(RFIMStruct);

}




void CovarianceCublasProduction()
{
	uint64_t valuesPerSample = 3;
	uint64_t sampleNum = 2;

	RFIMMemoryStruct* RFIMStruct = RFIMMemoryStructCreate(valuesPerSample, sampleNum, 2);

	float* h_signal; //Column first signal (3, 2), 3 == valuesPerSample, 2 == sampleNum

	h_signal = (float*)malloc( sizeof(float) * valuesPerSample * sampleNum);


	//Set the host signal
	for(uint32_t i = 0; i < valuesPerSample * sampleNum; ++i)
	{
		h_signal[i] = i + 1;
	}

	float* d_signal;
	cudaMalloc(&d_signal, sizeof(float) * valuesPerSample * sampleNum);

	CudaUtility_CopySignalToDevice(h_signal, &d_signal, sizeof(float) * valuesPerSample * sampleNum);
	free(h_signal);

	//Calculate the covariance matrix
	Device_CalculateCovarianceMatrix(RFIMStruct, d_signal);

	//Copy the data back to the device and print it


	//Test the upper tricovariance matrix
	float* h_upperTriCovarMatrix = (float*)malloc(sizeof(float) * valuesPerSample * valuesPerSample);

	CudaUtility_CopySignalToHost(RFIMStruct->d_upperTriangularCovarianceMatrix, &h_upperTriCovarMatrix, valuesPerSample * valuesPerSample * sizeof(float));

	/*
	for(int i = 0; i < valuesPerSample * valuesPerSample; ++i)
	{
		printf("Covar %d: %f\n", i, h_upperTriCovarMatrix[i]);
	}
	*/


	bool failed = false;


	if(fabsf(h_upperTriCovarMatrix[0] - 2.25f) > 0.000001f)
	{
		failed = true;
	}

	if(fabsf(h_upperTriCovarMatrix[1] - 0.0f) > 0.000001f)
	{
		failed = true;
	}

	if(fabsf(h_upperTriCovarMatrix[2] - 0.0f) > 0.000001f)
	{
		failed = true;
	}

	if(fabsf(h_upperTriCovarMatrix[3] - 2.25f) > 0.000001f)
	{
		failed = true;
	}

	if(fabsf(h_upperTriCovarMatrix[4] - 2.25f) > 0.000001f)
	{
		failed = true;
	}

	if(fabsf(h_upperTriCovarMatrix[5] - 0.0f) > 0.000001f)
	{
		failed = true;
	}

	if(fabsf(h_upperTriCovarMatrix[6] - 2.25f) > 0.000001f)
	{
		failed = true;
	}

	if(fabsf(h_upperTriCovarMatrix[7] - 2.25f) > 0.000001f)
	{
		failed = true;
	}

	if(fabsf(h_upperTriCovarMatrix[8] - 2.25f) > 0.000001f)
	{
		failed = true;
	}

	if(failed)
	{
		fprintf(stderr, "CovarianceCublasProduction Unit test failed!\n");
		exit(1);
	}



	//Test the full matrix
	float* h_fullCovarianceMatrix = (float*)malloc(sizeof(float) * valuesPerSample * valuesPerSample);
	CudaUtility_CopySignalToHost(RFIMStruct->d_fullSymmetricCovarianceMatrix, &h_fullCovarianceMatrix, sizeof(float) * valuesPerSample * valuesPerSample);


	for(int i = 0; i < valuesPerSample * valuesPerSample; ++i)
	{
		if(h_fullCovarianceMatrix[i] - 2.25f > 0.000001f)
			failed = true;


		//printf("Full covar %d: %f\n", i, h_fullCovarianceMatrix[i]);
	}


	if(failed)
	{
		fprintf(stderr, "CovarianceCublasProduction Unit test at generating the correct full covariance matrix!\n");
		exit(1);
	}


	RFIMMemoryStructDestroy(RFIMStruct);

	free(h_upperTriCovarMatrix);
	free(h_fullCovarianceMatrix);
	cudaFree(d_signal);
}



void EigendecompProduction()
{
	int valuesPerSample = 2;
	int covarianceMatrixByteSize = sizeof(float) * valuesPerSample * valuesPerSample;



	RFIMMemoryStruct* RFIM = RFIMMemoryStructCreate(valuesPerSample, valuesPerSample, 2);


	//Create small full covariance matrix
	float* h_fullSymmCovarianceMatrix = (float*)malloc( covarianceMatrixByteSize );

	h_fullSymmCovarianceMatrix[0] = 5.0f;
	h_fullSymmCovarianceMatrix[1] = 2.0f;
	h_fullSymmCovarianceMatrix[2] = 2.0f;
	h_fullSymmCovarianceMatrix[3] = 5.0f;



	CudaUtility_CopySignalToDevice(h_fullSymmCovarianceMatrix, &RFIM->d_fullSymmetricCovarianceMatrix,  covarianceMatrixByteSize);

	//Compute the eigenvectors/values
	Device_EigenvalueSolver(RFIM);


	//Check to see that everything is correct
	float* h_eigenvalues = (float*)malloc(sizeof(float) * valuesPerSample);
	float* h_eigenvectorMatrix = (float*)malloc(sizeof(float) * valuesPerSample * valuesPerSample);

	CudaUtility_CopySignalToHost(RFIM->d_S, &h_eigenvalues, sizeof(float) * valuesPerSample);
	CudaUtility_CopySignalToHost(RFIM->d_U, &h_eigenvectorMatrix, sizeof(float) * valuesPerSample * valuesPerSample);

	/*
	for(int i = 0; i < valuesPerSample; ++i)
	{
		printf("Eigenvalue %d: %f\n", i, h_eigenvalues[i]);
	}

	printf("\n");

	for(int i = 0; i < valuesPerSample * valuesPerSample; ++i)
	{
		printf("Eigenvec %d: %f\n", i, h_eigenvectorMatrix[i]);
	}
	*/


	bool failed = false;

	if(h_eigenvalues[0] - 7.0f > 0.0000001f)
		failed = true;
	if(h_eigenvalues[1] - 3.0f > 0.0000001f)
		failed = true;

	if(failed)
	{
		fprintf(stderr, "EigendecompProduction Unit test: failed to correctly calculate eigenvalues\n");
		exit(1);
	}

	if(fabs(h_eigenvectorMatrix[0] + 0.707107) > 0.000001f)
		failed = true;
	if(fabs(h_eigenvectorMatrix[1] + 0.707107) > 0.000001f)
		failed = true;
	if(fabs(h_eigenvectorMatrix[2] + 0.707107) > 0.000001f)
		failed = true;
	if(fabs(h_eigenvectorMatrix[3] - 0.707107) > 0.000001f)
		failed = true;

	if(failed)
	{
		fprintf(stderr, "EigendecompProduction Unit test: failed to correctly calculate eigenvectors\n");
		exit(1);
	}

	free(h_eigenvalues);
	free(h_eigenvectorMatrix);

	RFIMMemoryStructDestroy(RFIM);

}



//Doesn't actually prove that the filter itself works, just that the math functions are working as you would expected
//By removing 0 dimensions we should get the same signal back
void FilteringProduction()
{
	int valuesPerSample = 2;
	int signalByteSize = sizeof(float) * valuesPerSample * valuesPerSample;

	//REDUCE NOTHING! This should give us back the same signal
	RFIMMemoryStruct* RFIM = RFIMMemoryStructCreate(valuesPerSample, valuesPerSample, 0);


	//Create small full covariance matrix
	float* h_signal = (float*)malloc( signalByteSize );

	h_signal[0] = 1.0f;
	h_signal[1] = 2.0f;
	h_signal[2] = 7.0f;
	h_signal[3] = -8.0f;

	//Copy signal to the device
	float* d_signal;
	cudaMalloc(&d_signal, signalByteSize);
	CudaUtility_CopySignalToDevice(h_signal, &d_signal, signalByteSize);

	//Calculate the covariance matrix
	Device_CalculateCovarianceMatrix(RFIM, d_signal);

	//Calculate the eigenvectors
	Device_EigenvalueSolver(RFIM);

	//Setup the signal output
	float* d_filteredSignal;
	cudaMalloc(&d_filteredSignal, signalByteSize);


	//Do the projection
	Device_EigenReductionAndFiltering(RFIM, d_signal, d_filteredSignal);


	//Copy the signal back to the host
	float* h_filteredSignal = (float*)malloc(signalByteSize);
	CudaUtility_CopySignalToHost(d_filteredSignal, &h_filteredSignal, signalByteSize);

	bool failed = false;

	//Make sure we got the same signal back
	for(uint32_t i = 0; i < valuesPerSample * valuesPerSample; ++i)
	{
		//print the signal
		//printf("Orig %d: %f, filt %d: %f\n", i, h_signal[i], i, h_filteredSignal[i]);

		if(fabs(h_signal[i]) - fabs(h_filteredSignal[i]) > 0.0000001f)
		{
			failed = true;
		}
	}



	if(failed)
	{
		fprintf(stderr, "FilteringProduction: Unit test failed!\n");
		exit(1);
	}


	RFIMMemoryStructDestroy(RFIM);
	free(h_signal);
	free(h_filteredSignal);
	cudaFree(d_signal);
	cudaFree(d_filteredSignal);
}






void RunAllUnitTests()
{
	MeanCublasProduction();
	CovarianceCublasProduction();
	EigendecompProduction();
	FilteringProduction();

	printf("All tests passed!\n");

}

