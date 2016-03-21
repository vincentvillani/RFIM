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

#include <cublas.h>

#include <assert.h>
#include <string>

//Production tests
void MeanCublasProduction();
void CovarianceCublasProduction();
void TransposeProduction();
void GraphProduction();
void EigendecompProduction();

//Private dec's
void ParallelMeanUnitTest();
void ParallelMeanCublas();


void CovarianceMatrixUsingMyCodeUnitTest();
void CovarianceMatrixCUBLAS();
void CovarianceMatrixCUBLASSsyrk_v2();




//-------------------------------------

//Production
//-------------------------------------

void MeanCublasProduction()
{
	uint64_t valuesPerSample = 3;
	uint64_t sampleNum = 2;

	float* h_signal; //Column first signal (3, 2), 3 == valuesPerSample, 2 == sampleNum
	float* h_meanMatrix;

	h_signal = (float*)malloc(sizeof(float) * 6);


	float* d_signal;
	float* d_meanMatrix;

	//Set the host signal
	for(uint32_t i = 0; i < 6; ++i)
	{
		h_signal[i] = i + 1;
	}

	d_signal = CudaUtility_CopySignalToDevice(h_signal, sizeof(float) * 6);


	//Calculate the mean matrix
	d_meanMatrix= DEBUG_CALCULATE_MEAN_MATRIX(d_signal, valuesPerSample, sampleNum);


	//Copy it back to the host
	h_meanMatrix = CudaUtility_CopySignalToHost(d_meanMatrix, valuesPerSample * valuesPerSample * sizeof(float));


	//Print out the result
	/*
	printf("\n\n");

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

	free(h_signal);
	free(h_meanMatrix);

	cudaFree(d_signal);
	cudaFree(d_meanMatrix);
}




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
	*/



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
	*/

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



	//Transpose the matrix
	float* d_covarianceMatrixTransposed = Device_MatrixTranspose(d_covarianceMatrix, valuesPerSample, valuesPerSample);

	//Set the transposes diagonal to zero
	dim3 blockDim(32);
	dim3 gridDim(1, ceil(valuesPerSample / (float)32));
	setDiagonalToZero<<<gridDim, blockDim>>>(d_covarianceMatrixTransposed, valuesPerSample);

	//Add it to the covariance matrix
	cublasHandle_t cublasHandle;
	cublasCreate_v2(&cublasHandle);

	cublasStatus_t cublasStatus;

	float alpha = 1.0f;
	float beta = 1.0f;

	float* d_fullCovarianceMatrix;
	cudaMalloc(&d_fullCovarianceMatrix, sizeof(float) * valuesPerSample * valuesPerSample);

	cublasStatus = cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, valuesPerSample, valuesPerSample, &alpha, d_covarianceMatrix, valuesPerSample,
			&beta, d_covarianceMatrixTransposed, valuesPerSample, d_fullCovarianceMatrix, valuesPerSample);

	if(cublasStatus_t != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "EigendecompProduction cublasSgeam call failed\n");
		exit(1);
	}


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



	Device_EigenvalueSolver(&cublasHandle, &handle, d_fullCovarianceMatrix, d_U, d_S, d_VT, d_Lworkspace, NULL, workspaceLength, d_devInfo, valuesPerSample);

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
	cudaFree(d_covarianceMatrixTransposed);
	cudaFree(d_fullCovarianceMatrix);
	cudaFree(d_S);
	cudaFree(d_U);
	cudaFree(d_VT);
	cudaFree(d_Lworkspace);
	cudaFree(d_Rworkspace);
	cudaFree(d_devInfo);

	cusolverDnDestroy(handle);
	cublasDestroy_v2(cublasHandle);

}



//-------------------------------------



void ParallelMeanUnitTest()
{
	float* d_knownSignal;
	float* d_mean;

	float* h_knownSignal;
	float* h_mean;

	float expectedMean = 0;


	uint64_t n = 8;

	//Allocate memory
	cudaMalloc(&d_knownSignal, sizeof(float) * n);
	cudaMalloc(&d_mean, sizeof(float));
	CudaCheckError();

	h_knownSignal = (float*)calloc(n, sizeof(float));
	h_mean = (float*)calloc(1, sizeof(float));

	//Create a signal
	for(uint32_t i = 0; i < n; ++i)
	{
		h_knownSignal[i] = i;

		expectedMean += i;
	}

	expectedMean /= (n - 1);

	cudaMemcpy(d_knownSignal, h_knownSignal, sizeof(float) * n, cudaMemcpyHostToDevice);
	CudaCheckError();

	//Run the kernel
	parallelMeanUnroll2 <<<2, 2>>> (d_knownSignal, n, d_mean);
	CudaCheckError();

	//copy the result back to the host
	cudaMemcpy(h_mean, d_mean, sizeof(float), cudaMemcpyDeviceToHost);
	CudaCheckError();

	if(*h_mean - expectedMean > 0.000001)
	{
		fprintf(stderr, "ParallelMeanUnitTest() failed. Expected: %f, Actual: %f\n", expectedMean, *h_mean);
		exit(1);
	}

	//Free all memory
	free(h_knownSignal);
	free(h_mean);

	cudaFree(d_knownSignal);
	cudaFree(d_mean);

}



void ParallelMeanCublas()
{
	float* d_meanCoefficentMatrix; //A 1x3 matrix containing just 1's (COL MAJOR)
	float* d_matrix; //A 3x3 matrix containing data that we want to sum down the columns (COL MAJOR)
	float* d_resultMatrix; //A 1x3 matrix containing the mean for each column of the matrix (COL MAJOR)

	float* h_meanCoefficentMatrix; //A 1x3 matrix containing just 1's (COL MAJOR)
	float* h_matrix; //A 3x3 matrix containing data that we want to sum down the columns (COL MAJOR)
	float* h_resultMatrix; //A 1x3 matrix containing the mean for each column of the matrix (COL MAJOR)

	//Allocate data for the host and the device
	cudaMalloc(&d_meanCoefficentMatrix, sizeof(float) * 3);
	cudaMalloc(&d_matrix, sizeof(float) * 9);
	cudaMalloc(&d_resultMatrix, sizeof(float) * 3);
	cudaMemset(d_resultMatrix, 0, sizeof(float) * 3); //Set the inital values to zero

	h_meanCoefficentMatrix = (float*)malloc(sizeof(float) * 3);
	h_matrix = (float*)malloc(sizeof(float) * 9);
	h_resultMatrix = (float*)malloc(sizeof(float) * 3);

	//Setup the data on the host
	for(uint32_t i = 0; i < 3; ++i)
	{
		h_meanCoefficentMatrix[i] = 1;
	}

	//Setup the matrix whose columns we want to sum
	for(uint32_t i = 0; i < 9; ++i)
	{
		h_matrix[i] = i + 1;
	}

	//Copy the data over to the device
	cudaMemcpy(d_meanCoefficentMatrix, h_meanCoefficentMatrix, sizeof(float) * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrix, h_matrix, sizeof(float) * 9, cudaMemcpyHostToDevice);

	//Setup cublas
	cublasHandle_t cublasHandle;
	cublasCreate_v2(&cublasHandle); //Create the handle

	//Do the matrix * matrix multiplication
	float alpha = 1.0f / 3.0f; //To calculate the mean after the matrix multiplication takes place
	float beta = 1.0f;
	cublasSgemm_v2(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 3, 3,
			&alpha, d_meanCoefficentMatrix, 1, d_matrix, 3, &beta, d_resultMatrix, 1);

	//copy the data back across to the host
	cudaMemcpy(h_resultMatrix, d_resultMatrix, sizeof(float) * 3, cudaMemcpyDeviceToHost);

	/*
	//print out the results
	for(uint64_t i = 0; i < 3; ++i)
	{
		printf("%llu: %f\n", i, h_resultMatrix[i]);
	}
	*/

	bool failed = false;

	if(h_resultMatrix[0] - 2.0f > 0.000001)
		failed = true;
	else if(h_resultMatrix[1] - 5.0f > 0.000001)
		failed = true;
	else if(h_resultMatrix[2] - 8.0f > 0.000001)
		failed = true;

	if(failed)
	{
		fprintf(stderr, "ParallelMeanCublas() failed.\nExpected: %f, %f, %f\nActual: %f, %f, %f\n", 2.0f, 5.0f, 8.0f,
				h_resultMatrix[0], h_resultMatrix[1], h_resultMatrix[2]);
		exit(1);
	}

	//Free all memory
	free(h_matrix);
	free(h_meanCoefficentMatrix);
	free(h_resultMatrix);

	cudaFree(d_matrix);
	cudaFree(d_meanCoefficentMatrix);
	cudaFree(d_resultMatrix);

}





void CovarianceMatrixUsingMyCodeUnitTest()
{
	float* d_vec;
	float* d_resultMatrix;

	float* h_vec;
	float* h_resultMatrix;

	uint64_t n = 2;
	uint64_t resultMatrixLength = upperTriangularLength(n);

	dim3 block = dim3(2, 2);
	dim3 grid = dim3(1, 1);


	//Allocate memory
	cudaMalloc(&d_vec, sizeof(float) * n);
	cudaMalloc(&d_resultMatrix, sizeof(float) * resultMatrixLength);
	CudaCheckError();

	h_vec = (float*)calloc(n, sizeof(float));
	h_resultMatrix = (float*)calloc(resultMatrixLength, sizeof(float));


	//Generate the vector/signal
	for(uint64_t i = 0; i < n; ++i)
	{
		h_vec[i] = 1 + i;
	}

	cudaMemcpy(d_vec, h_vec, n * sizeof(float), cudaMemcpyHostToDevice);
	CudaCheckError();

	/*
	printf("Launching kernel with parameters\nGrid(%d, %d), Block(%d, %d)\n",
			grid.x, grid.y, block.x, block.y);
	*/

	//Run the kernel
	outerProductSmartBruteForceLessThreads <<<grid, block>>> (d_resultMatrix, d_vec, n);
	CudaCheckError();

	//check the results
	cudaMemcpy(h_resultMatrix, d_resultMatrix, resultMatrixLength * sizeof(float), cudaMemcpyDeviceToHost);
	CudaCheckError();

	/*
	for(uint64_t i = 0; i < resultMatrixLength; ++i)
	{
		printf("%llu: %f\n", i, h_resultMatrix[i]);
	}
	*/

	bool failed = false;

	if(h_resultMatrix[0] - 1.0f > 0.000001)
		failed = true;
	else if(h_resultMatrix[1] - 2.0f > 0.000001)
		failed = true;
	else if(h_resultMatrix[2] - 4.0f > 0.000001)
		failed = true;

	if(failed)
	{
		fprintf(stderr, "CovarianceMatrixUsingMyCodeUnitTest() failed.\nExpected: %f, %f, %f\nActual: %f, %f, %f\n", 1.0f, 2.0f, 4.0f,
				h_resultMatrix[0], h_resultMatrix[1], h_resultMatrix[2]);
		exit(1);
	}

	cudaFree(d_vec);
	cudaFree(d_resultMatrix);
	CudaCheckError();

	free(h_vec);
	free(h_resultMatrix);
}




void CovarianceMatrixCUBLAS()
{
	float* d_vec;
	float* d_resultMatrix;

	float* h_vec;
	float* h_resultMatrix;

	uint64_t vectorLength = 3;
	uint64_t resultMatrixLength = upperTriangularLength(vectorLength);

	//Start the cublas context
	cublasStatus_t cuStat;
	cublasHandle_t cublasHandle;
	cuStat = cublasCreate_v2(&cublasHandle); //Create the handle

	if(cuStat != CUBLAS_STATUS_SUCCESS)
	{
		printf("ERROR!\n");
		exit(1);
	}

	//Allocate storage for the arrays on the host and device
	cudaMalloc(&d_vec, sizeof(float) * vectorLength);
	h_vec = (float*)malloc(sizeof(float) * vectorLength);

	cudaMalloc(&d_resultMatrix, sizeof(float) * resultMatrixLength);
	cudaMemset(d_resultMatrix, 0, sizeof(float) * resultMatrixLength); //Set the resultMatrix to zero
	h_resultMatrix = (float*)malloc(sizeof(float) * resultMatrixLength);


	//Set the vector for on the host
	for(uint64_t i = 0; i < vectorLength; ++i)
	{
		h_vec[i] = 1 + i;
	}

	//Copy the data over
	//Column major, cublas uses 1-based indexing
	//cuStat = cublasSetMatrix(vectorLength, 1, sizeof(float), h_vec, vectorLength, d_vec, vectorLength);
	cudaMemcpy(d_vec, h_vec, sizeof(float) * vectorLength, cudaMemcpyHostToDevice);

	//Perform the outer product
	float alpha = 1.0f;
	cuStat = cublasSspr_v2(cublasHandle, CUBLAS_FILL_MODE_LOWER, vectorLength, &alpha, d_vec, 1, d_resultMatrix);


	if(cuStat != CUBLAS_STATUS_SUCCESS)
	{
		printf("ERROR!\n");
		exit(1);
	}

	//Copy the data back to the host
	//cublasGetMatrix(vectorLength, vectorLength, sizeof(float), d_resultMatrix, vectorLength, h_resultMatrix, vectorLength);
	cudaMemcpy(h_resultMatrix, d_resultMatrix, sizeof(float) * resultMatrixLength, cudaMemcpyDeviceToHost);

	bool failed = false;

	if(h_resultMatrix[0] - 1.0f > 0.000001f)
		failed = true;
	else if(h_resultMatrix[1] - 2.0f > 0.000001f)
		failed = true;
	else if(h_resultMatrix[2] - 3.0f > 0.000001f)
		failed = true;
	else if(h_resultMatrix[3] - 4.0f > 0.000001f)
		failed = true;
	else if(h_resultMatrix[4] - 6.0f > 0.000001f)
		failed = true;
	else if(h_resultMatrix[5] - 9.0f > 0.000001f)
		failed = true;


	if(failed)
	{
		fprintf(stderr, "CovarianceMatrixCUBLAS() failed.\nExpected: %f, %f, %f, %f, %f, %f\nActual: %f, %f, %f, %f, %f, %f\n", 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 9.0f,
				h_resultMatrix[0], h_resultMatrix[1], h_resultMatrix[2], h_resultMatrix[3], h_resultMatrix[4], h_resultMatrix[5]);
		exit(1);
	}

	/*
	for(uint64_t i = 0; i < resultMatrixLength; ++i)
	{
		printf("%llu: %f\n", i, h_resultMatrix[i]);
	}
	 */
	//Free all memory
	free(h_vec);
	free(h_resultMatrix);

	cudaFree(d_vec);
	cudaFree(d_resultMatrix);

	cublasDestroy_v2(cublasHandle);


}



void CovarianceMatrixCUBLASSsyrk_v2()
{
	float* h_signal;
	float* h_covarianceMatrix;

	float* d_signal;
	float* d_covarianceMatrix;

	uint64_t sampleElements = 4;
	uint64_t sampleNumber = 2;
	uint64_t totalElements = sampleElements * sampleNumber;
	uint64_t covarianceMatrixElements = sampleNumber * sampleNumber;



	//Allocate data and set data
	cudaMalloc(&d_signal, sizeof(float) * totalElements);
	cudaMalloc(&d_covarianceMatrix, sizeof(float) * covarianceMatrixElements);
	cudaMemset(d_covarianceMatrix, 0, sizeof(float) * covarianceMatrixElements);

	CudaCheckError();

	h_signal = (float*)malloc(sizeof(float) * totalElements);
	h_covarianceMatrix = (float*)malloc(sizeof(float) * covarianceMatrixElements);

	for(uint32_t i = 0; i < totalElements; ++i)
	{
		h_signal[i] = i + 1;
		//printf("%f, ", h_signal[i]);
	}

	//Copy the signal to the device
	cudaMemcpy(d_signal, h_signal, sizeof(float) * totalElements, cudaMemcpyHostToDevice);

	CudaCheckError();

	//Startup cublas
	cublasHandle_t cublasHandle;
	cublasCreate_v2(&cublasHandle);

	cublasStatus_t status;

	float alpha = 1.0f;
	float beta = 1.0f;
			//Handle, fill mode, transpose, n, k, alpha, Matrix A, lda, beta, Matrix C, ldc
	status = cublasSsyrk_v2(cublasHandle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, 2, 4, &alpha, d_signal, 2, &beta, d_covarianceMatrix, 2);

	CudaCheckError();

	if(status != CUBLAS_STATUS_SUCCESS)
	{
		printf("ERROR!\n");
	}

	//Copy the data back to the host
	cudaMemcpy(h_covarianceMatrix, d_covarianceMatrix, sizeof(float) * covarianceMatrixElements, cudaMemcpyDeviceToHost);

	CudaCheckError();

	/*
	//print the result
	for(uint64_t i = 0; i < covarianceMatrixElements; ++i)
	{
		printf("%llu: %f\n", i, h_covarianceMatrix[i]);
	}
	*/

	free(h_signal);
	free(h_covarianceMatrix);

	cudaFree(d_signal);
	cudaFree(d_covarianceMatrix);

	cublasDestroy_v2(cublasHandle);
}











void RunAllUnitTests()
{
	MeanCublasProduction();
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
}

