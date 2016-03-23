/*
 * RFIMHelperFunctions.cu
 *
 *  Created on: 17/03/2016
 *      Author: vincentvillani
 */

#include "../Header/RFIMHelperFunctions.h"

#include <stdio.h>



#include "../Header/CudaUtilityFunctions.h"
#include "../Header/Kernels.h"
#include "../Header/CudaMacros.h"


//Private helper functions
//--------------------------

void CalculateMeanMatrix(RFIMMemoryStruct* RFIMStruct, const float* d_signalMatrix);



//Private functions implementation
//----------------------------------

void CalculateMeanMatrix(RFIMMemoryStruct* RFIMStruct, const float* d_signalMatrix)
{

	//Calculate d_meanVec
	//d_meanVec = d_oneMatrix (1 x h_numberOfSamples) * d_signal (transposed) (h_numberOfSamples x h_valuesPerSample ) matrix = 1 * h_valuesPerSample matrix
	//This each of the beams added up. It adds up the columns of transposed d_signal
	//---------------------------
	cublasStatus_t cublasError;


	float alpha = 1.0f / RFIMStruct->h_numberOfSamples;
	float beta = 0;

	cublasError = cublasSgemm_v2(*RFIMStruct->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 1, RFIMStruct->h_valuesPerSample, RFIMStruct->h_numberOfSamples,
			&alpha, RFIMStruct->d_oneVec, 1, d_signalMatrix, RFIMStruct->h_valuesPerSample, &beta, RFIMStruct->d_meanVec, 1);

	if(cublasError != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "CalculateMeanMatrix: An error occured while computing d_meanVec\n");
		//exit(1);
	}

	//--------------------------------------


	//Calculate mean matrix
	//mean matrix = outer product of the transposed d_meanVec with itself
	//d_meanMatrix = d_meanVec_Transposed (h_valuesPerSample x 1) * d_meanVec (1 x h_valuesPerSample)
	//--------------------------------------

	alpha = 1.0f;

	cublasError = cublasSsyrk_v2(*RFIMStruct->cublasHandle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, RFIMStruct->h_valuesPerSample, 1,
			&alpha, RFIMStruct->d_meanVec, 1, &beta, RFIMStruct->d_upperTriangularCovarianceMatrix, RFIMStruct->h_valuesPerSample);

	if(cublasError != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "CalculateMeanMatrix: An error occured while computing d_meanMatrix\n");
		//exit(1);
	}

}


//--------------------------


float* Device_GenerateWhiteNoiseSignal(curandGenerator_t* rngGen, uint64_t h_valuesPerSample, uint64_t h_numberOfSamples)
{

	uint64_t totalSignalLength = h_valuesPerSample * h_numberOfSamples;
	uint64_t totalSignalByteSize = totalSignalLength * sizeof(float);

	float* d_signal;

	cudaError_t error;

	//Allocate the memory required to store the signal
	error =  cudaMalloc(&d_signal, totalSignalByteSize);

	//Check that it was allocated successfully
	if(error != cudaSuccess)
	{
		fprintf(stderr, "Device_GenerateWhiteNoiseSignal: Unable to allocate %llu bytes of memory on the device\n", totalSignalByteSize);
		//exit(1);
	}


	//Generate the signal!
	//Generate random numbers on the device
	//Generate random numbers using a normal distribution
	//Normal distribution should emulate white noise hopefully?
	//Generate signal
	if(curandGenerateNormal(*rngGen, d_signal, totalSignalLength, 0.0f, 1.0f) != CURAND_STATUS_SUCCESS)
	{
		fprintf(stderr, "Device_GenerateWhiteNoiseSignal: Error when generating the signal\n");
		//exit(1);
	}


	//Return the generated signal that resides in DEVICE memory
	return d_signal;

}





void Device_CalculateCovarianceMatrix(RFIMMemoryStruct* RFIMStruct, const float* d_signalMatrix)
{
	//d_signalMatrix should be column-major as CUBLAS is column-major library (indexes start at 1 also)
	//Remember to take that into account!


	//Calculate the meanMatrix of the signal
	//--------------------------------

	CalculateMeanMatrix(RFIMStruct, d_signalMatrix);


	//TODO: DEBUGGGGG
	float* h_meanMatrix = (float*)malloc(sizeof(float) * RFIMStruct->h_valuesPerSample *  RFIMStruct->h_valuesPerSample);


			CudaUtility_CopySignalToHost(RFIMStruct->d_upperTriangularCovarianceMatrix, &h_meanMatrix,
			sizeof(float) * RFIMStruct->h_valuesPerSample * RFIMStruct->h_valuesPerSample);


	//TODO: DEBUGGGGG
	for(int i = 0; i < RFIMStruct->h_valuesPerSample * RFIMStruct->h_valuesPerSample; ++i)
	{
		printf("Intermediate mean %d: %f\n", i, h_meanMatrix[i]);
	}

	//--------------------------------


	//Calculate the covariance matrix
	//-------------------------------
	//1. Calculate the outer product of the signal (sampleElements x sampleNumber) * ( sampleNumber x sampleElements)
	//	AKA. signal * (signal)T, where T = transpose, which will give you a (sampleNumber x sampleNumber) matrix as a result

	//Take the outer product of the signal with itself
	float alpha = 1.0f / RFIMStruct->h_numberOfSamples;
	float beta = -1.0f;

	cublasStatus_t cublasError;



	//At this point RFIMStruct->d_upperTriangularCovarianceMatrix is actually the upper triangular mean matrix,
	//this is done to get better performance out of the cublas API
	cublasError = cublasSsyrk_v2(*RFIMStruct->cublasHandle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, RFIMStruct->h_valuesPerSample,
			RFIMStruct->h_numberOfSamples,
			&alpha, d_signalMatrix, RFIMStruct->h_valuesPerSample,
			&beta, RFIMStruct->d_upperTriangularCovarianceMatrix, RFIMStruct->h_valuesPerSample);

	if(cublasError != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "Device_CalculateCovarianceMatrix: error calculating the covariance matrix\n");
		//exit(1);
	}

	/*
	//TODO: DEBUGGGGG
	float* h_covarMatrix = CudaUtility_CopySignalToHost(RFIMStruct->d_upperTriangularCovarianceMatrix,
			sizeof(float) * RFIMStruct->h_valuesPerSample * RFIMStruct->h_valuesPerSample);

	//TODO: DEBUGGGGG
	for(int i = 0; i < RFIMStruct->h_valuesPerSample * RFIMStruct->h_valuesPerSample; ++i)
	{
		printf("Intermediate covar %d: %f\n", i, h_covarMatrix[i]);
	}

	*/



	/*
	//Calculate the full symmetric covariance matrix
	//1. Transpose the covariance matrix
	Device_MatrixTranspose(&RFIMStruct->cublasHandle, RFIMStruct->d_upperTriangularCovarianceMatrix, RFIMStruct->d_upperTriangularTransposedMatrix,
			RFIMStruct->h_valuesPerSample, RFIMStruct->h_valuesPerSample);

	//2. Set the transposed covariance matrix diagonal to zero
	dim3 blockDim(32);
	dim3 gridDim(1, ceilf(RFIMStruct->h_valuesPerSample / (float)32));
	setDiagonalToZero<<<gridDim, blockDim>>> (RFIMStruct->d_upperTriangularTransposedMatrix, RFIMStruct->h_valuesPerSample);


	//3. Add the two matrices together

	//TODO: Look into whether or not I need to do this. This memory is reused each time around
	cudaMemset(RFIMStruct->d_fullSymmetricCovarianceMatrix, 0, sizeof(float) * RFIMStruct->h_valuesPerSample * RFIMStruct->h_valuesPerSample);

	alpha = 1.0f;
	beta = 1.0f;

	cublasError = cublasSgeam(RFIMStruct->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, RFIMStruct->h_valuesPerSample, RFIMStruct->h_valuesPerSample,
				&alpha, RFIMStruct->d_upperTriangularCovarianceMatrix, RFIMStruct->h_valuesPerSample,
				&beta, RFIMStruct->d_upperTriangularTransposedMatrix, RFIMStruct->h_valuesPerSample,
				RFIMStruct->d_fullSymmetricCovarianceMatrix, RFIMStruct->h_valuesPerSample);


	if(cublasError != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "Device_CalculateCovarianceMatrix: cublasSgeam call failed\n");
		exit(1);
	}

	*/
}




void Device_MatrixTranspose(cublasHandle_t* cublasHandle, const float* d_matrix, float* d_matrixTransposed, uint64_t rowNum, uint64_t colNum)
{

	cublasStatus_t cublasStatus;

	float alpha = 1;
	float beta = 0;


	cublasStatus = cublasSgeam(*cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, colNum, rowNum,
			&alpha, d_matrix, rowNum,
			&beta, d_matrix, rowNum,
			d_matrixTransposed, colNum);


	if(cublasStatus != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "Device_InplaceMatrixTranspose: Transposition of the matrix failed!\n");
		//exit(1);
	}

}



/*
float* Device_FullSymmetricMatrix(cublasHandle_t* cublasHandle, const float* d_triangularMatrix, uint64_t rowAndColNum)
{
	float* d_fullMatrix;

	//Transpose the d_triangularMatrix
	//Transpose the matrix
	float* d_triangularMatrixTransposed = Device_MatrixTranspose(d_triangularMatrix, rowAndColNum, rowAndColNum);

	//Set the transposes diagonal to zero
	dim3 blockDim(32);
	dim3 gridDim(1, ceilf(rowAndColNum / (float)32));
	setDiagonalToZero<<<gridDim, blockDim>>>(d_triangularMatrixTransposed, rowAndColNum);

	//TODO: Debug, remove this. It will affect performance
	CudaCheckError();


	//Add the triangular matrices together
	float alpha = 1.0f;
	float beta = 1.0f;

	//Allocate memory for the full matrix
	cudaMalloc(&d_fullMatrix, sizeof(float) * rowAndColNum * rowAndColNum);

	cublasStatus_t cublasStatus = cublasSgeam(*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, rowAndColNum, rowAndColNum,
			&alpha, d_triangularMatrix, rowAndColNum, &beta, d_triangularMatrixTransposed, rowAndColNum, d_fullMatrix, rowAndColNum);

	if(cublasStatus != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "Device_FullSymmetricMatrix: cublasSgeam call failed\n");
		exit(1);
	}

	//Free the transposed matrix
	cudaFree(d_triangularMatrixTransposed);

	//return the result
	return d_fullMatrix;
}
*/



void Device_EigenvalueSolver(cublasHandle_t* cublasHandle, cusolverDnHandle_t* cusolverHandle, float* d_fullCovarianceMatrix, float* d_U, float* d_S, float* d_VT,
		float* d_Lworkspace, float* d_Rworkspace, int workspaceLength, int* d_devInfo, int h_valuesPerSample)
{


	cusolverStatus_t cusolverStatus;



	cusolverStatus = cusolverDnSgesvd(*cusolverHandle, 'A', 'A', h_valuesPerSample, h_valuesPerSample,
			d_fullCovarianceMatrix, h_valuesPerSample, d_S, d_U, h_valuesPerSample, d_VT, h_valuesPerSample,
			d_Lworkspace, workspaceLength, d_Rworkspace, d_devInfo);


	int* h_devInfo = (int*)malloc(sizeof(int));
	cudaMemcpy(h_devInfo, d_devInfo, sizeof(int), cudaMemcpyDeviceToHost);

	if(*h_devInfo != 0)
	{
		fprintf(stderr, "Device_EigenvalueSolver: Error with the %dth parameter\n", *h_devInfo);
		//exit(1);
	}

	if(cusolverStatus != CUSOLVER_STATUS_SUCCESS)
	{
		/*
		if(cusolverStatus == CUSOLVER_STATUS_NOT_INITIALIZED)
			printf("1\n");
		if(cusolverStatus == CUSOLVER_STATUS_INVALID_VALUE)
			printf("2\n");
		if(cusolverStatus == CUSOLVER_STATUS_ARCH_MISMATCH)
			printf("3\n");
		if(cusolverStatus == CUSOLVER_STATUS_INTERNAL_ERROR)
			printf("4\n");
		*/


		fprintf(stderr, "Device_EigenvalueSolver: Error solving eigenvalues\n");
		exit(1);
	}


	free(h_devInfo);

}




void DEBUG_CALCULATE_MEAN_MATRIX(RFIMMemoryStruct* RFIMStruct, float* d_signalMatrix)
{
	CalculateMeanMatrix(RFIMStruct, d_signalMatrix);
}


