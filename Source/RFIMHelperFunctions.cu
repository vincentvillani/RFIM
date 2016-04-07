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


float* Device_GenerateWhiteNoiseSignal(curandGenerator_t* rngGen, uint64_t h_valuesPerSample, uint64_t h_numberOfSamples, uint64_t h_batchSize)
{

	uint64_t totalSignalLength = h_valuesPerSample * h_numberOfSamples * h_batchSize;
	uint64_t totalSignalByteSize = sizeof(float) * totalSignalLength;


	float* d_signalMatrix;

	cudaMalloc(&d_signalMatrix, totalSignalByteSize);

	//Generate the signal!
	//Generate random numbers on the device
	//Generate random numbers using a normal distribution
	//Normal distribution should emulate white noise hopefully?
	//Generate signal
	if(curandGenerateNormal(*rngGen, d_signalMatrix, totalSignalLength, 0.0f, 1.0f) != CURAND_STATUS_SUCCESS)
	{
		fprintf(stderr, "Device_GenerateWhiteNoiseSignal: Error when generating the signal\n");
		exit(1);
	}


	return d_signalMatrix;

}



void Device_CalculateMeanMatrices(RFIMMemoryStruct* RFIMStruct, float* d_signalMatrices)
{


	//Calculate d_meanVec
	//d_meanVec = d_oneMatrix (1 x h_numberOfSamples) * d_signal (transposed) (h_numberOfSamples x h_valuesPerSample ) matrix = 1 * h_valuesPerSample matrix
	//This each of the beams added up. It adds up the columns of transposed d_signal
	//---------------------------
	cublasStatus_t cublasError;


	float alpha = 1.0f / RFIMStruct->h_numberOfSamples;
	float beta = 0;

	uint64_t signalMatrixOffset = RFIMStruct->h_valuesPerSample * RFIMStruct->h_numberOfSamples;
	uint64_t meanVecOffset = RFIMStruct->h_valuesPerSample;

	uint64_t streamIndex = 0;

	for(uint64_t i = 0; i < RFIMStruct->h_batchSize; ++i)
	{

		//Set the cuda stream
		cublasSetStream_v2(*RFIMStruct->cublasHandle, RFIMStruct->h_cudaStreams[streamIndex]);

		//Compute the mean vector
		//We use the same d_onevec each time
		cublasError = cublasSgemm_v2(*RFIMStruct->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
									1, RFIMStruct->h_valuesPerSample, RFIMStruct->h_numberOfSamples,
									&alpha, RFIMStruct->d_oneVec, 1,
									d_signalMatrices + (i * signalMatrixOffset), RFIMStruct->h_valuesPerSample, &beta,
									RFIMStruct->d_meanVec + (i * meanVecOffset), 1);


		//Check for errors
		if(cublasError != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "Device_CalculateMeanMatrices: An error occured while computing d_meanVec\n");
			exit(1);
		}

		//Iterate stream index
		streamIndex += 1;
		if(streamIndex >= RFIMStruct->h_cudaStreamsLength)
		{
			streamIndex = 0;
		}
	}





	//Calculate mean matrix
	//mean matrix = outer product of the transposed d_meanVec with itself
	//d_meanMatrix = d_meanVec_Transposed (h_valuesPerSample x 1) * d_meanVec (1 x h_valuesPerSample)
	//--------------------------------------

	alpha = 1.0f;
	streamIndex = 0;

	uint64_t covarianceMatrixOffset = RFIMStruct->h_valuesPerSample * RFIMStruct->h_valuesPerSample;

	for(uint64_t i = 0; i < RFIMStruct->h_batchSize; ++i)
	{

		//Set the cuda stream
		cublasSetStream_v2(*RFIMStruct->cublasHandle, RFIMStruct->h_cudaStreams[streamIndex]);

		//Compute the mean outer product
		cublasError = cublasSgemm_v2(*RFIMStruct->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
				RFIMStruct->h_valuesPerSample, RFIMStruct->h_valuesPerSample, 1,
				&alpha, RFIMStruct->d_meanVec + (i * meanVecOffset), 1,
				RFIMStruct->d_meanVec + (i * meanVecOffset), 1, &beta,
				RFIMStruct->d_covarianceMatrix + (i * covarianceMatrixOffset), RFIMStruct->h_valuesPerSample);



		//Check for errors
		if(cublasError != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "Device_CalculateMeanMatrices: An error occured while computing d_meanVec\n");
			exit(1);
		}

		//Iterate stream index
		streamIndex += 1;
		if(streamIndex >= RFIMStruct->h_cudaStreamsLength)
		{
			streamIndex = 0;
		}
	}


}




void Device_CalculateCovarianceMatrix(RFIMMemoryStruct* RFIMStruct, float** d_signalMatrices)
{
	/*
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

	*/
}









void Device_EigenvalueSolver(RFIMMemoryStruct* RFIMStruct)
{

	/*
	cusolverStatus_t cusolverStatus;

	//Calculate the eigenvectors/values for each batch
	for(uint64_t i = 0; i < RFIMStruct->h_batchSize; ++i)
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

	cudaDeviceSynchronize();


	//Check all dev info's, copy them all at once
	cudaMemcpyAsync(RFIMStruct->h_devInfoValues, RFIMStruct->d_devInfo, sizeof(int) * RFIMStruct->h_batchSize, cudaMemcpyDeviceToHost, RFIMStruct->cudaStream);

	cudaDeviceSynchronize();

	for(uint64_t i = 0; i < RFIMStruct->h_batchSize; ++i)
	{
		if(RFIMStruct->h_devInfoValues[i] != 0)
		{
			fprintf(stderr, "Device_EigenvalueSolver: Error with the %dth parameter on the %lluth batch\n", RFIMStruct->h_devInfoValues[i], i);
			//exit(1);
		}
	}

*/

}



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



void Device_EigenReductionAndFiltering(RFIMMemoryStruct* RFIMStruct, float** d_originalSignalMatrices, float** d_filteredSignals)
{
	/*

	cublasStatus_t cublasStatus;

	//Projected signal matrix
	//Ps = (Er Transposed) * Os
	float alpha = 1;
	float beta = 0;

	//uint32_t reducedDimension = RFIMStruct->h_valuesPerSample - RFIMStruct->h_eigenVectorDimensionsToReduce;
	uint64_t eigenvectorPointerOffset = RFIMStruct->h_valuesPerSample * RFIMStruct->h_eigenVectorDimensionsToReduce;


	//Set the appropriate number of columns to zero
	for(uint64_t i = 0; i < RFIMStruct->h_batchSize; ++i)
	{
		cudaMemsetAsync(RFIMStruct->h_UDevicePointers[i], 0, sizeof(float) * eigenvectorPointerOffset, 0);
	}


	//Wait for the memsets to complete
	cudaStreamSynchronize(RFIMStruct->cudaStream);



	cublasStatus = cublasSgemmBatched(*RFIMStruct->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
			RFIMStruct->h_valuesPerSample, RFIMStruct->h_numberOfSamples, RFIMStruct->h_valuesPerSample,
			&alpha,  (const float**)(RFIMStruct->d_U), RFIMStruct->h_valuesPerSample,
			(const float**)d_originalSignalMatrices, RFIMStruct->h_valuesPerSample, &beta,
			RFIMStruct->d_projectedSignalMatrix, RFIMStruct->h_valuesPerSample,
			RFIMStruct->h_batchSize);



	if(cublasStatus != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "Device_EigenReductionAndFiltering: error calculating the projected signal\n");
		exit(1);
	}


	//final signal matrix
	// Fs = Er * Ps


	cublasStatus = cublasSgemmBatched(*RFIMStruct->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			RFIMStruct->h_valuesPerSample, RFIMStruct->h_numberOfSamples, RFIMStruct->h_valuesPerSample,
			&alpha, (const float**)(RFIMStruct->d_U), RFIMStruct->h_valuesPerSample,
			(const float**)RFIMStruct->d_projectedSignalMatrix, RFIMStruct->h_valuesPerSample, &beta,
			d_filteredSignals, RFIMStruct->h_valuesPerSample,
			RFIMStruct->h_batchSize);



	if(cublasStatus != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "Device_EigenReductionAndFiltering: error calculating the filtered signal\n");
		exit(1);
	}
	*/

}

/*
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


