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





//Private functions implementation
//----------------------------------




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
		exit(1);
	}


	//Generate the signal!
	//Generate random numbers on the device
	//Generate random numbers using a normal distribution
	//Normal distribution should emulate white noise hopefully?
	//Generate signal
	if(curandGenerateNormal(*rngGen, d_signal, totalSignalLength, 0.0f, 1.0f) != CURAND_STATUS_SUCCESS)
	{
		fprintf(stderr, "Device_GenerateWhiteNoiseSignal: Error when generating the signal\n");
		exit(1);
	}


	//Return the generated signal that resides in DEVICE memory
	return d_signal;

}



void Device_CalculateMeanMatrices(RFIMMemoryStruct* RFIMStruct, float** d_signalMatrices)
{

	//Calculate d_meanVec
	//d_meanVec = d_oneMatrix (1 x h_numberOfSamples) * d_signal (transposed) (h_numberOfSamples x h_valuesPerSample ) matrix = 1 * h_valuesPerSample matrix
	//This each of the beams added up. It adds up the columns of transposed d_signal
	//---------------------------
	cublasStatus_t cublasError;


	float alpha = 1.0f / RFIMStruct->h_numberOfSamples;
	float beta = 0;

	cublasError = cublasSgemmBatched(*RFIMStruct->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
			1, RFIMStruct->h_valuesPerSample, RFIMStruct->h_numberOfSamples,
			&alpha, (const float**)RFIMStruct->d_oneVec, 1,
			(const float**)d_signalMatrices, RFIMStruct->h_valuesPerSample, &beta,
			RFIMStruct->d_meanVec, 1,
			RFIMStruct->h_batchSize);


	if(cublasError != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "Device_CalculateMeanMatrices: An error occured while computing d_meanVec\n");
		exit(1);
	}

	//--------------------------------------


	//Calculate mean matrix
	//mean matrix = outer product of the transposed d_meanVec with itself
	//d_meanMatrix = d_meanVec_Transposed (h_valuesPerSample x 1) * d_meanVec (1 x h_valuesPerSample)
	//--------------------------------------

	alpha = 1.0f;

	cublasError = cublasSgemmBatched(*RFIMStruct->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
			RFIMStruct->h_valuesPerSample, RFIMStruct->h_valuesPerSample, 1,
			&alpha, (const float**)RFIMStruct->d_meanVec, RFIMStruct->h_valuesPerSample,
			(const float**)RFIMStruct->d_meanVec, RFIMStruct->h_valuesPerSample, &beta,
			RFIMStruct->d_covarianceMatrix, RFIMStruct->h_valuesPerSample,
			RFIMStruct->h_batchSize);

	if(cublasError != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "Device_CalculateMeanMatrices: An error occurred while computing d_meanMatrix\n");
		exit(1);
	}

}




void Device_CalculateCovarianceMatrix(RFIMMemoryStruct* RFIMStruct, float** d_signalMatrices)
{
	//d_signalMatrix should be column-major as CUBLAS is column-major library (indexes start at 1 also)
	//Remember to take that into account!


	//Calculate the meanMatrix of the signal
	//--------------------------------
	Device_CalculateMeanMatrices(RFIMStruct, d_signalMatrices);

	//--------------------------------



	//Calculate the covariance matrix
	//-------------------------------
	//1. Calculate the outer product of the signal (sampleElements x sampleNumber) * ( sampleNumber x sampleElements)
	//	AKA. signal * (signal)T, where T = transpose, which will give you a (sampleNumber x sampleNumber) matrix as a result

	//Take the outer product of the signal with itself
	float alpha = 1.0f / RFIMStruct->h_numberOfSamples;
	float beta = -1;

	cublasStatus_t cublasError;


	cublasError = cublasSgemmBatched(*RFIMStruct->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
			RFIMStruct->h_valuesPerSample, RFIMStruct->h_valuesPerSample, RFIMStruct->h_numberOfSamples,
			&alpha, (const float**)d_signalMatrices, RFIMStruct->h_valuesPerSample,
			(const float**)d_signalMatrices, RFIMStruct->h_valuesPerSample, &beta,
			RFIMStruct->d_covarianceMatrix, RFIMStruct->h_valuesPerSample,
			RFIMStruct->h_batchSize);


	if(cublasError != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "Device_CalculateCovarianceMatrix: error calculating the covariance matrix\n");
		exit(1);
	}


}









void Device_EigenvalueSolver(RFIMMemoryStruct* RFIMStruct)
{


	cusolverStatus_t cusolverStatus;

	//Calculate the eigenvectors/values for each batch
	for(uint32_t i = 0; i < RFIMStruct->h_batchSize; ++i)
	{

		cusolverStatus = cusolverDnSgesvd(*RFIMStruct->cusolverHandle, 'A', 'A', RFIMStruct->h_valuesPerSample, RFIMStruct->h_valuesPerSample,
				RFIMStruct->h_covarianceMatrixDevicePointers[i], RFIMStruct->h_valuesPerSample,
				RFIMStruct->h_SDevicePointers[i],  RFIMStruct->h_UDevicePointers[i], RFIMStruct->h_valuesPerSample,
				RFIMStruct->h_VTDevicePointers[i], RFIMStruct->h_valuesPerSample,
				RFIMStruct->h_eigWorkingSpaceDevicePointers[i], RFIMStruct->h_eigWorkingSpaceLength, NULL, RFIMStruct->d_devInfo + i);


		//Check for errors
		if(cusolverStatus != CUSOLVER_STATUS_SUCCESS)
		{

			if(cusolverStatus == CUSOLVER_STATUS_NOT_INITIALIZED)
				printf("1\n");
			if(cusolverStatus == CUSOLVER_STATUS_INVALID_VALUE)
				printf("2\n");
			if(cusolverStatus == CUSOLVER_STATUS_ARCH_MISMATCH)
				printf("3\n");
			if(cusolverStatus == CUSOLVER_STATUS_INTERNAL_ERROR)
				printf("4\n");


			fprintf(stderr, "Device_EigenvalueSolver: Error solving eigenvalues\n");
			exit(1);

		}

	}



	//Check all dev info's, copy them all at once
	cudaMemcpy(RFIMStruct->h_devInfoValues, RFIMStruct->d_devInfo, sizeof(int) * RFIMStruct->h_batchSize, cudaMemcpyDeviceToHost);


	for(uint32_t i = 0; i < RFIMStruct->h_batchSize; ++i)
	{
		if(RFIMStruct->h_devInfoValues[i] != 0)
		{
			fprintf(stderr, "Device_EigenvalueSolver: Error with the %dth parameter on the %dth batch\n", RFIMStruct->h_devInfoValues[i], i);
			//exit(1);
		}
	}


}

/*

//Eigenvector reduction and signal projection/filtering
//All matrices are column-major

//h_eigenVectorDimensionsToReduce is the number of eigenvectors to remove from the eigenvector matrix, for now it's 2

//Os = Original signal matrix
//A column-major matrix containing the signal
//It has dimensionality: h_valuesPerSample * h_numberOfSamples, which will probably be 26 x 1024?

//Er = Reduced Eigenvector matrix.
//The eigenvectors of the fully symmetrical covariance matrix, with some of the eigenvectors removed.
//It has dimensions: h_valuesPerSample x (h_valuesPerSample - h_eigenVectorDimensionsToReduce), probably 26 x 24?

//Ps = Projected signal matrix.
//The original data projected along the reduced eigenvector axis's
//This matrix will have dimensions: (h_valuesPerSample - h_eigenVectorDimensionsToReduce) x h_numberOfSamples, probably 24 x 1024?

//Fs = Final signal matrix
//This is the original data projected into the lower reduced eigenvector dimensionality, then back into the original dimensionality. This has the effect of flattening data along the removed dimensions. It may add correlations were there was previously none?
//But should also hopefully remove some RFI
//It will have dimensions: h_valuesPerSample * h_numberOfSamples, probably 26 x 1024?


//Equations!
// Ps = (Er Transposed) * Os
// Fs = Er * Ps      Note that the inverse of Er should just be its transpose, even if you remove some of the eigenvectors. This is because all the eigenvectors are orthogonal unit vectors (or should be anyway...)


//Steps
//1. Remove RFIMStruct->h_eigenVectorDimensionsToReduce dimensions from the eigenvector matrix (this is done via pointer arithmetic rather than actually removing the data) THIS WON'T WORK IF THE COLUMNS TO REMOVE ARE NOT ALL NEXT TO EACH OTHER!
//2. Compute the matrix Ps
//3. Compute the matrix Fs (final signal matrix)
//4. Pass on Fs, down the line? Keep it on the GPU? Copy it to the host? Write it to a file in the file system? Dunno.



void Device_EigenReductionAndFiltering(RFIMMemoryStruct* RFIMStruct, float* d_originalSignalMatrix, float* d_filteredSignal)
{

	cublasStatus_t cublasStatus;

	//Projected signal matrix
	//Ps = (Er Transposed) * Os
	float alpha = 1;
	float beta = 0;

	uint32_t reducedDimension = RFIMStruct->h_valuesPerSample - RFIMStruct->h_eigenVectorDimensionsToReduce;
	uint32_t eigenvectorPointerOffset = RFIMStruct->h_valuesPerSample * RFIMStruct->h_eigenVectorDimensionsToReduce;

	cublasStatus = cublasSgemm_v2(*RFIMStruct->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
			reducedDimension, RFIMStruct->h_numberOfSamples, RFIMStruct->h_valuesPerSample,
			&alpha,  RFIMStruct->d_U + eigenvectorPointerOffset, RFIMStruct->h_valuesPerSample,
			d_originalSignalMatrix, RFIMStruct->h_valuesPerSample, &beta,
			RFIMStruct->d_projectedSignalMatrix, reducedDimension);

	if(cublasStatus != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "Device_EigenReductionAndFiltering: error calculating the projected signal\n");
		exit(1);
	}


	//final signal matrix
	// Fs = Er * Ps

	cublasStatus = cublasSgemm_v2(*RFIMStruct->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			RFIMStruct->h_valuesPerSample, RFIMStruct->h_numberOfSamples, reducedDimension,
			&alpha, RFIMStruct->d_U + eigenvectorPointerOffset, RFIMStruct->h_valuesPerSample,
			RFIMStruct->d_projectedSignalMatrix, reducedDimension, &beta,
			d_filteredSignal, RFIMStruct->h_valuesPerSample);


	if(cublasStatus != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "Device_EigenReductionAndFiltering: error calculating the filtered signal\n");
		exit(1);
	}

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


*/


