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
#include <string>


//Production tests
void MeanCublasProduction();
/*
void CovarianceCublasProduction();
void TransposeProduction();
void GraphProduction();
void EigendecompProduction();
*/



//-------------------------------------

//Production
//-------------------------------------

void MeanCublasProduction()
{

	uint32_t valuesPerSample = 3;
	uint32_t sampleNum = 2;

	RFIMMemoryStruct* RFIMStruct = RFIMMemoryStructCreate(valuesPerSample, sampleNum);

	float* h_signal;


	h_signal = (float*)malloc(sizeof(float) * 6);


	float* d_signal;

	//Set the host signal
	for(uint32_t i = 0; i < 6; ++i)
	{
		h_signal[i] = i + 1;
	}

	d_signal = CudaUtility_CopySignalToDevice(h_signal, sizeof(float) * 6);

	//Calculate the mean matrix
	DEBUG_CALCULATE_MEAN_MATRIX(RFIMStruct, d_signal);


	//Copy it back to the host
	//At this point d_upperTriangularCovarianceMatrix is the mean matrix
	float* h_meanMatrix = CudaUtility_CopySignalToHost(RFIMStruct->d_upperTriangularCovarianceMatrix, valuesPerSample * valuesPerSample * sizeof(float));


	//Print out the result

	/*
	for(uint32_t i = 0; i < valuesPerSample * valuesPerSample; ++i)
	{
		printf("final: %u: %f\n", i, h_meanMatrix[i]);
	}
	*/



	bool failed = false;

	if(h_meanMatrix[0] - 6.25f > 0.000001f)
		failed = true;
	else if(h_meanMatrix[1] - 0.0f > 0.000001f)
		failed = true;
	else if(h_meanMatrix[2] - 0.0f > 0.000001f)
		failed = true;
	else if(h_meanMatrix[3] - 8.75f > 0.000001f)
		failed = true;
	else if(h_meanMatrix[4] - 12.25f > 0.000001f)
		failed = true;
	else if(h_meanMatrix[5] - 0.0f > 0.000001f)
		failed = true;
	else if(h_meanMatrix[6] - 11.25f > 0.000001f)
		failed = true;
	else if(h_meanMatrix[7] - 15.75f > 0.000001f)
		failed = true;
	else if(h_meanMatrix[8] - 20.25f > 0.000001f)
		failed = true;


	if(failed)
	{
		fprintf(stderr, "MeanCublasProduction failed!\n");
		exit(1);
	}

	RFIMMemoryStructDestroy(RFIMStruct);

}



/*
void CovarianceCublasProduction()
{
	uint64_t valuesPerSample = 3;
	uint64_t sampleNum = 2;

	float* h_signal; //Column first signal (3, 2), 3 == valuesPerSample, 2 == sampleNum

	h_signal = (float*)malloc( sizeof(float) * 6);

	float* d_signal;
	float* d_covarianceMatrix;

	//Set the host signal
	for(uint32_t i = 0; i < 6; ++i)
	{
		h_signal[i] = i + 1;
	}

	d_signal = CudaUtility_CopySignalToDevice(h_signal, sizeof(float) * 6);

	cublasHandle_t cublasHandle;
	cublasCreate_v2(&cublasHandle);

	d_covarianceMatrix = Device_CalculateCovarianceMatrix(&cublasHandle, d_signal, valuesPerSample, sampleNum);

	cublasDestroy_v2(cublasHandle);

	//Copy the data back to the device and print it
	free(h_signal);

	h_signal = CudaUtility_CopySignalToHost(d_covarianceMatrix, valuesPerSample * valuesPerSample * sizeof(float));

	/*
	for(int i = 0; i < valuesPerSample * valuesPerSample; ++i)
	{
		printf("%d: %f\n", i, h_signal[i]);
	}




	bool failed = false;

	if(h_signal[0] - 10.75f > 0.000001f)
		failed = true;
	else if(h_signal[1] - 0.0f > 0.000001f)
		failed = true;
	else if(h_signal[2] - 0.0f > 0.000001f)
		failed = true;
	else if(h_signal[3] - 13.25f > 0.000001f)
		failed = true;
	else if(h_signal[4] - 16.75f > 0.000001f)
		failed = true;
	else if(h_signal[5] - 0.0f > 0.000001f)
		failed = true;
	else if(h_signal[6] - 15.75f > 0.000001f)
		failed = true;
	else if(h_signal[7] - 20.25f > 0.000001f)
		failed = true;
	else if(h_signal[8] - 24.75f > 0.000001f)
		failed = true;

	if(failed)
	{
		fprintf(stderr, "CovarianceCublasProduction Unit test failed!\n");
		exit(1);
	}



	free(h_signal);
	cudaFree(d_signal);
	cudaFree(d_covarianceMatrix);
}




void TransposeProduction()
{
	uint64_t valuesPerSample = 3;
	uint64_t sampleNum = 2;

	float* h_signal; //Column first signal (3, 2), 3 == valuesPerSample, 2 == sampleNum

	h_signal = (float*)malloc( sizeof(float) * 6);

	float* d_signal;
	float* d_transposedSignal;

	//Set the host signal
	for(uint32_t i = 0; i < 6; ++i)
	{
		h_signal[i] = i + 1;
	}

	d_signal = CudaUtility_CopySignalToDevice(h_signal, sizeof(float) * 6);

	//Transpose the matrix
	d_transposedSignal = Device_MatrixTranspose(d_signal, valuesPerSample, sampleNum);


	free(h_signal);
	h_signal = CudaUtility_CopySignalToHost(d_transposedSignal, 6 * sizeof(float));

	/*
	for(int i = 0; i < 6; ++i)
	{
		printf("%d: %f\n", i, h_signal[i]);
	}


	bool failed = false;

	if(h_signal[0] - 1.0f > 0.000001f)
		failed = true;
	else if(h_signal[1] - 4.0f > 0.000001f)
		failed = true;
	else if(h_signal[2] - 2.0f > 0.000001f)
		failed = true;
	else if(h_signal[3] - 5.0f > 0.000001f)
		failed = true;
	else if(h_signal[4] - 3.0f > 0.000001f)
		failed = true;
	else if(h_signal[5] - 6.0f > 0.000001f)
		failed = true;

	if(failed)
	{
		fprintf(stderr, "TransposeProduction unit test failed!\n");
		exit(1);
	}



	free(h_signal);

	cudaFree(d_signal);
	cudaFree(d_transposedSignal);
}




void GraphProduction()
{
	uint64_t valuesPerSample = 3;
	uint64_t sampleNum = 2;

	float* h_signal; //Column first signal (3, 2), 3 == valuesPerSample, 2 == sampleNum

	h_signal = (float*)malloc( sizeof(float) * 6);

	//Set the host signal
	for(uint32_t i = 0; i < 6; ++i)
	{
		h_signal[i] = i + 1;
	}

	//Write it to file
	Utility_WriteSignalMatrixToFile(std::string( "signal.txt"), h_signal, valuesPerSample, sampleNum);

}



void EigendecompProduction()
{
	int valuesPerSample = 2;
	int covarianceMatrixByteSize = sizeof(float) * valuesPerSample * valuesPerSample;


	cusolverStatus_t cusolverStatus;
	cusolverDnHandle_t handle;
	cusolverStatus = cusolverDnCreate(&handle);

	if(cusolverStatus != CUSOLVER_STATUS_SUCCESS)
	{
		fprintf(stderr, "EigendecompProduction: Error creating a cusolver handle\n");
		exit(1);
	}



	float* h_covarianceMatrix = (float*)malloc( covarianceMatrixByteSize );

	h_covarianceMatrix[0] = 5.0f;
	h_covarianceMatrix[1] = 0.0f;
	h_covarianceMatrix[2] = 2.0f;
	h_covarianceMatrix[3] = 5.0f;

	float* d_covarianceMatrix = CudaUtility_CopySignalToDevice(h_covarianceMatrix, covarianceMatrixByteSize);


	//Add it to the covariance matrix
	cublasHandle_t cublasHandle;
	cublasCreate_v2(&cublasHandle);


	//Transpose the matrix
	float* d_fullySymmetricCovarianceMatrix = Device_FullSymmetricMatrix(&cublasHandle, d_covarianceMatrix,
			valuesPerSample);


	float* d_S;
	float* d_U;
	float* d_VT;
	float* d_Lworkspace;
	float* d_Rworkspace;
	int* d_devInfo;
	int workspaceLength = 0;

	cudaMalloc(&d_S, sizeof(float) * valuesPerSample);
	cudaMalloc(&d_U, sizeof(float) * valuesPerSample * valuesPerSample);
	cudaMalloc(&d_VT, sizeof(float) * valuesPerSample * valuesPerSample);
	cusolverStatus = cusolverDnSgesvd_bufferSize(handle, valuesPerSample, valuesPerSample, &workspaceLength);

	if(cusolverStatus != CUSOLVER_STATUS_SUCCESS)
	{
		fprintf(stderr, "EigendecompProduction: Error calculating buffer size\n");
		exit(1);
	}

	cudaMalloc(&d_Lworkspace, workspaceLength);
	cudaMalloc(&d_Rworkspace, workspaceLength);
	cudaMalloc(&d_devInfo, sizeof(int));



	Device_EigenvalueSolver(&cublasHandle, &handle, d_fullySymmetricCovarianceMatrix,
			d_U, d_S, d_VT, d_Lworkspace, NULL, workspaceLength, d_devInfo, valuesPerSample);

	float* h_eigenvalues = CudaUtility_CopySignalToHost(d_S,  sizeof(float) * valuesPerSample);
	float* h_eigenvectorMatrix = CudaUtility_CopySignalToHost(d_U, sizeof(float) * valuesPerSample * valuesPerSample);

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

	if(h_eigenvectorMatrix[0] + 0.707107 > 0.000001f)
		failed = true;
	if(h_eigenvectorMatrix[1] + 0.707107 > 0.000001f)
		failed = true;
	if(h_eigenvectorMatrix[2] + 0.707107 > 0.000001f)
		failed = true;
	if(h_eigenvectorMatrix[3] - 0.707107 > 0.000001f)
		failed = true;

	if(failed)
	{
		fprintf(stderr, "EigendecompProduction Unit test: failed to correctly calculate eigenvectors\n");
		exit(1);
	}

	free(h_covarianceMatrix);
	free(h_eigenvalues);
	free(h_eigenvectorMatrix);
	cudaFree(d_covarianceMatrix);
	cudaFree(d_fullySymmetricCovarianceMatrix);
	cudaFree(d_S);
	cudaFree(d_U);
	cudaFree(d_VT);
	cudaFree(d_Lworkspace);
	cudaFree(d_Rworkspace);
	cudaFree(d_devInfo);

	cusolverDnDestroy(handle);
	cublasDestroy_v2(cublasHandle);

}

*/







void RunAllUnitTests()
{
	MeanCublasProduction();
	/*
	CovarianceCublasProduction();
	TransposeProduction();
	//GraphProduction();
	EigendecompProduction();

	ParallelMeanUnitTest();
	ParallelMeanCublas();

	CovarianceMatrixUsingMyCodeUnitTest();
	CovarianceMatrixCUBLAS();
	CovarianceMatrixCUBLASSsyrk_v2();


	printf("All tests passed!\n");
	*/
}

