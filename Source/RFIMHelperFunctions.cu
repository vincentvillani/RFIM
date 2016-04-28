/*
 * RFIMHelperFunctions.cu
 *
 *  Created on: 17/03/2016
 *      Author: vincentvillani
 */

#include "../Header/RFIMHelperFunctions.h"

#include <stdio.h>
#include <string.h>
#include <mkl.h>
#include <mkl_lapacke.h>



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


	cudaDeviceSynchronize();

	return d_signalMatrix;

}


float* Device_GenerateWhiteNoiseSignal(curandGenerator_t* rngGen, uint64_t h_valuesPerSample, uint64_t h_numberOfSamples,
		uint64_t h_batchSize, uint64_t h_threadNum)
{

	/*
	CURAND_STATUS_NOT_INITIALIZED if the generator was never created \n
	 * - CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
	 *    a previous kernel launch \n
	 * - CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \n
	 * - CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
	 *    not a multiple of the quasirandom dimension, or is not a multiple
	 *    of two for pseudorandom generators \n
	 * - CURAND_STATUS_SUCCESS if the results were generated successfully \n
	 * */


	uint64_t totalSignalLength = h_valuesPerSample * h_numberOfSamples * h_batchSize * h_threadNum;
	uint64_t totalSignalByteSize = sizeof(float) * totalSignalLength;


	float* d_signalMatrix;

	cudaMalloc(&d_signalMatrix, totalSignalByteSize);

	//Generate the signal!
	//Generate random numbers on the device
	//Generate random numbers using a normal distribution
	//Normal distribution should emulate white noise hopefully?
	//Generate signal
	curandStatus_t curandStatus = curandGenerateNormal(*rngGen, d_signalMatrix, totalSignalLength, 0.0f, 1.0f);

	if(curandStatus != CURAND_STATUS_SUCCESS)
	{
		fprintf(stderr, "Device_GenerateWhiteNoiseSignal: Error when generating the signal\n");

		if(curandStatus == CURAND_STATUS_NOT_INITIALIZED)
			fprintf(stderr, "CURAND_STATUS_NOT_INITIALIZED\n");
		else if(curandStatus == CURAND_STATUS_PREEXISTING_FAILURE)
			fprintf(stderr, "CURAND_STATUS_PREEXISTING_FAILURE\n");
		else if(curandStatus == CURAND_STATUS_LAUNCH_FAILURE)
			fprintf(stderr, "CURAND_STATUS_LAUNCH_FAILURE\n");
		else if(curandStatus == CURAND_STATUS_LENGTH_NOT_MULTIPLE)
			fprintf(stderr, "CURAND_STATUS_LENGTH_NOT_MULTIPLE\n");
		else
			fprintf(stderr, "Unknown cuRand error\n");


		exit(1);
	}


	cudaDeviceSynchronize();

	return d_signalMatrix;

}


/*
float* Device_GenerateWhiteNoiseSignal(curandGenerator_t* rngGen, uint64_t h_valuesPerSample, uint64_t h_numberOfSamples,
		uint64_t h_batchSize, uint64_t h_threadNum, float mean, float stdDev)
{
	uint64_t totalSignalLength = h_valuesPerSample * h_numberOfSamples * h_batchSize * h_threadNum;
	uint64_t totalSignalByteSize = sizeof(float) * totalSignalLength;


	float* d_signalMatrix;

	cudaMalloc(&d_signalMatrix, totalSignalByteSize);

	//Generate the signal!
	//Generate random numbers on the device
	//Generate random numbers using a normal distribution
	//Normal distribution should emulate white noise hopefully?
	//Generate signal
	if(curandGenerateNormal(*rngGen, d_signalMatrix, totalSignalLength, mean, stdDev) != CURAND_STATUS_SUCCESS)
	{
		fprintf(stderr, "Device_GenerateWhiteNoiseSignal: Error when generating the signal\n");
		exit(1);
	}


	cudaDeviceSynchronize();

	return d_signalMatrix;
}
*/



cuComplex* Device_GenerateWhiteNoiseSignalComplex(curandGenerator_t* rngGen, uint64_t h_valuesPerSample, uint64_t h_numberOfSamples,
		uint64_t h_batchSize, uint64_t h_threadNum)
{
	uint64_t totalSignalLength = h_valuesPerSample * h_numberOfSamples * h_batchSize * h_threadNum;
	uint64_t totalSignalByteSize = sizeof(cuComplex) * totalSignalLength;


	cuComplex* d_signalMatrix;

	cudaMalloc(&d_signalMatrix, totalSignalByteSize);

	//Generate the signal!
	//Generate random numbers on the device
	//Generate random numbers using a normal distribution
	//Normal distribution should emulate white noise hopefully?
	//Generate signal
	if(curandGenerateNormal(*rngGen, (float*)d_signalMatrix, totalSignalLength * 2, 0.0f, 1.0f) != CURAND_STATUS_SUCCESS) //* 2 for complex numbers
	{
		fprintf(stderr, "Device_GenerateWhiteNoiseSignalComplex: Error when generating the signal\n");
		exit(1);
	}


	cudaDeviceSynchronize();

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


		/*
		//TODO: DEBUG REMOVE
		cudaError_t cudaError = cudaDeviceSynchronize();
		cublasError = cublasGetError();

		if(cudaError != cudaSuccess || cublasError != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "CalculateMeanMatrix 1 error\n");
		}
		*/

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

		/*
		//TODO: DEBUG REMOVE
		cudaError_t cudaError = cudaDeviceSynchronize();
		cublasError = cublasGetError();

		if(cudaError != cudaSuccess || cublasError != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "CalculateMeanMatrix 2 error\n");
		}
		*/
	}


}




void Device_CalculateMeanMatricesBatched(RFIMMemoryStructBatched* RFIMStruct, float** d_signalMatrices)
{
	//Calculate d_meanVec
	//d_meanVec = d_oneMatrix (1 x h_numberOfSamples) * d_signal (transposed) (h_numberOfSamples x h_valuesPerSample ) matrix = 1 * h_valuesPerSample matrix
	//This each of the beams added up. It adds up the columns of transposed d_signal
	//---------------------------
	cublasStatus_t cublasError;


	float alpha = 1.0f / RFIMStruct->h_numberOfSamples;
	float beta = 0;

	//uint64_t signalMatrixOffset = RFIMStruct->h_valuesPerSample * RFIMStruct->h_numberOfSamples;
	//uint64_t meanVecOffset = RFIMStruct->h_valuesPerSample;

	//uint64_t streamIndex = 0;



	//Set the cuda stream
	//cublasSetStream_v2(*RFIMStruct->cublasHandle, RFIMStruct->h_cudaStreams[streamIndex]);

	//Compute the mean vector
	//We use the same d_onevec each time
	cublasError = cublasSgemmBatched(*RFIMStruct->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
								1, RFIMStruct->h_valuesPerSample, RFIMStruct->h_numberOfSamples,
								&alpha, (const float**)RFIMStruct->d_oneVecBatched, 1,
								(const float**)d_signalMatrices, RFIMStruct->h_valuesPerSample, &beta,
								RFIMStruct->d_meanVecBatched, 1,
								RFIMStruct->h_batchSize);


	//Check for errors
	if(cublasError != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "Device_CalculateMeanMatrices: An error occured while computing d_meanVec\n");
		exit(1);
	}


	/*
	//TODO: DEBUG REMOVE
	cudaError_t cudaError = cudaDeviceSynchronize();
	cublasError = cublasGetError();

	if(cudaError != cudaSuccess || cublasError != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "CalculateMeanMatrix 1 error\n");
	}
	*/



	//Calculate mean matrix
	//mean matrix = outer product of the transposed d_meanVec with itself
	//d_meanMatrix = d_meanVec_Transposed (h_valuesPerSample x 1) * d_meanVec (1 x h_valuesPerSample)
	//--------------------------------------

	alpha = 1.0f;

	//Compute the mean outer product
	cublasError = cublasSgemmBatched(*RFIMStruct->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
			RFIMStruct->h_valuesPerSample, RFIMStruct->h_valuesPerSample, 1,
			&alpha, (const float**)RFIMStruct->d_meanVecBatched, 1,
			(const float**)RFIMStruct->d_meanVecBatched, 1, &beta,
			RFIMStruct->d_covarianceMatrixBatched, RFIMStruct->h_valuesPerSample,
			RFIMStruct->h_batchSize);



	//Check for errors
	if(cublasError != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "Device_CalculateMeanMatrices: An error occured while computing d_meanVec\n");
		exit(1);
	}


}






void Device_CalculateMeanMatricesComplex(RFIMMemoryStructComplex* RFIMStruct, cuComplex* d_signalMatrices)
{
	//Calculate d_meanVec
	//d_meanVec = d_oneMatrix (1 x h_numberOfSamples) * d_signal (transposed) (h_numberOfSamples x h_valuesPerSample ) matrix = 1 * h_valuesPerSample matrix
	//This each of the beams added up. It adds up the columns of transposed d_signal
	//---------------------------
	cublasStatus_t cublasError;


	cuComplex alpha = make_cuComplex(1.0f / RFIMStruct->h_numberOfSamples, 0) ;
	cuComplex beta = make_cuComplex(0, 0);

	uint64_t signalMatrixOffset = RFIMStruct->h_valuesPerSample * RFIMStruct->h_numberOfSamples;
	uint64_t meanVecOffset = RFIMStruct->h_valuesPerSample;

	uint64_t streamIndex = 0;

	for(uint64_t i = 0; i < RFIMStruct->h_batchSize; ++i)
	{

		//Set the cuda stream
		cublasSetStream_v2(*RFIMStruct->cublasHandle, RFIMStruct->h_cudaStreams[streamIndex]);

		//Compute the mean vector
		//We use the same d_onevec each time
		cublasError = cublasCgemm_v2(*RFIMStruct->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_C,
									1, RFIMStruct->h_valuesPerSample, RFIMStruct->h_numberOfSamples,
									&alpha, RFIMStruct->d_oneVec, 1,
									d_signalMatrices + (i * signalMatrixOffset), RFIMStruct->h_valuesPerSample, &beta,
									RFIMStruct->d_meanVec + (i * meanVecOffset), 1);


		//Check for errors
		if(cublasError != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "Device_CalculateMeanMatricesComplex: An error occured while computing d_meanVec\n");
			exit(1);
		}

		//Iterate stream index
		streamIndex += 1;
		if(streamIndex >= RFIMStruct->h_cudaStreamsLength)
		{
			streamIndex = 0;
		}


		/*
		//TODO: DEBUG REMOVE
		cudaError_t cudaError = cudaDeviceSynchronize();
		cublasError = cublasGetError();

		if(cudaError != cudaSuccess || cublasError != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "CalculateMeanMatrix 1 error\n");
		}
		*/

	}





	//Calculate mean matrix
	//mean matrix = outer product of the transposed d_meanVec with itself
	//d_meanMatrix = d_meanVec_Transposed (h_valuesPerSample x 1) * d_meanVec (1 x h_valuesPerSample)
	//--------------------------------------

	alpha = make_cuComplex(1.0f, 0);
	streamIndex = 0;

	uint64_t covarianceMatrixOffset = RFIMStruct->h_valuesPerSample * RFIMStruct->h_valuesPerSample;

	for(uint64_t i = 0; i < RFIMStruct->h_batchSize; ++i)
	{

		//Set the cuda stream
		cublasSetStream_v2(*RFIMStruct->cublasHandle, RFIMStruct->h_cudaStreams[streamIndex]);

		//Compute the mean outer product
		cublasError = cublasCgemm_v2(*RFIMStruct->cublasHandle, CUBLAS_OP_C, CUBLAS_OP_N,
				RFIMStruct->h_valuesPerSample, RFIMStruct->h_valuesPerSample, 1,
				&alpha, RFIMStruct->d_meanVec + (i * meanVecOffset), 1,
				RFIMStruct->d_meanVec + (i * meanVecOffset), 1, &beta,
				RFIMStruct->d_covarianceMatrix + (i * covarianceMatrixOffset), RFIMStruct->h_valuesPerSample);



		//Check for errors
		if(cublasError != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "Device_CalculateMeanMatricesComplex: An error occured while computing d_meanVec\n");
			exit(1);
		}

		//Iterate stream index
		streamIndex += 1;
		if(streamIndex >= RFIMStruct->h_cudaStreamsLength)
		{
			streamIndex = 0;
		}

		/*
		//TODO: DEBUG REMOVE
		cudaError_t cudaError = cudaDeviceSynchronize();
		cublasError = cublasGetError();

		if(cudaError != cudaSuccess || cublasError != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "CalculateMeanMatrix 2 error\n");
		}
		*/
	}
}



void Host_CalculateMeanMatrices(RFIMMemoryStructCPU* RFIMStruct, float* h_signalMatrices)
{
	float alphaOne = 1.0f / RFIMStruct->h_numberOfSamples;
	float alphaTwo = 1;
	float beta = 0;

	uint64_t signalMatrixOffset = RFIMStruct->h_valuesPerSample * RFIMStruct->h_numberOfSamples;
	uint64_t meanVecOffset = RFIMStruct->h_valuesPerSample;
	uint64_t covarianceMatrixOffset = RFIMStruct->h_valuesPerSample * RFIMStruct->h_valuesPerSample;


	//Compute the mean vector
	//We use the same d_onevec each time
	for(uint64_t i = 0; i < RFIMStruct->h_batchSize; ++i)
	{

		//Calculate meanVec
		cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
				1, RFIMStruct->h_valuesPerSample, RFIMStruct->h_numberOfSamples,
				alphaOne, RFIMStruct->h_oneVec, 1,
				h_signalMatrices + (i * signalMatrixOffset), RFIMStruct->h_valuesPerSample, beta,
				RFIMStruct->h_meanVec + (i * meanVecOffset), 1);


		//Calculate mean outer product matrix
		cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
				RFIMStruct->h_valuesPerSample, RFIMStruct->h_valuesPerSample, 1,
				alphaTwo, RFIMStruct->h_meanVec + (i * meanVecOffset), 1,
				RFIMStruct->h_meanVec + (i * meanVecOffset), 1, beta,
				RFIMStruct->h_covarianceMatrix + (i * covarianceMatrixOffset), RFIMStruct->h_valuesPerSample);

	}



}






void Device_CalculateCovarianceMatrix(RFIMMemoryStruct* RFIMStruct, float* d_signalMatrices)
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

	uint64_t signalOffset = RFIMStruct->h_valuesPerSample * RFIMStruct->h_numberOfSamples;
	uint64_t covarianceMatrixOffset = RFIMStruct->h_valuesPerSample * RFIMStruct->h_valuesPerSample;

	uint64_t cudaStreamIterator = 0;

	cublasStatus_t cublasError;

	for(uint64_t i = 0; i < RFIMStruct->h_batchSize; ++i)
	{
		//Set the stream for the library
		cublasSetStream_v2(*RFIMStruct->cublasHandle, RFIMStruct->h_cudaStreams[cudaStreamIterator]);

		cublasError = cublasSgemm_v2(*RFIMStruct->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
				RFIMStruct->h_valuesPerSample, RFIMStruct->h_valuesPerSample, RFIMStruct->h_numberOfSamples,
				&alpha, d_signalMatrices + (i * signalOffset), RFIMStruct->h_valuesPerSample,
				d_signalMatrices + (i * signalOffset), RFIMStruct->h_valuesPerSample, &beta,
				RFIMStruct->d_covarianceMatrix + (i * covarianceMatrixOffset), RFIMStruct->h_valuesPerSample);


		if(cublasError != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "Device_CalculateCovarianceMatrix: error calculating the covariance matrix\n");
			exit(1);
		}


		cudaStreamIterator += 1;
		if(cudaStreamIterator >= RFIMStruct->h_cudaStreamsLength)
		{
			cudaStreamIterator = 0;
		}

		/*
		//TODO: DEBUG REMOVE
		cudaError_t cudaError = cudaDeviceSynchronize();
		cublasError = cublasGetError();

		if(cudaError != cudaSuccess || cublasError != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "Device_CalculateCovarianceMatrix 1 error\n");
		}
		*/

	}


}




void Device_CalculateCovarianceMatrixBatched(RFIMMemoryStructBatched* RFIMStruct, float** d_signalMatrices)
{
	//d_signalMatrix should be column-major as CUBLAS is column-major library (indexes start at 1 also)
	//Remember to take that into account!


	//Calculate the meanMatrix of the signal
	//--------------------------------

	//Set the stream to stream zero, so it's not on the default stream
	cublasSetStream_v2(*RFIMStruct->cublasHandle, RFIMStruct->h_cudaStreams[0]);

	Device_CalculateMeanMatricesBatched(RFIMStruct, d_signalMatrices);

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
			RFIMStruct->d_covarianceMatrixBatched, RFIMStruct->h_valuesPerSample,
			RFIMStruct->h_batchSize);


	if(cublasError != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "Device_CalculateCovarianceMatrix: error calculating the covariance matrix\n");
		exit(1);
	}



	/*
	//TODO: DEBUG REMOVE
	cudaError_t cudaError = cudaDeviceSynchronize();
	cublasError = cublasGetError();

	if(cudaError != cudaSuccess || cublasError != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "Device_CalculateCovarianceMatrix 1 error\n");
	}
	*/


}







void Device_CalculateCovarianceMatrixComplex(RFIMMemoryStructComplex* RFIMStruct, cuComplex* d_signalMatrices)
{
	//d_signalMatrix should be column-major as CUBLAS is column-major library (indexes start at 1 also)
	//Remember to take that into account!


	//Calculate the meanMatrix of the signal
	//--------------------------------

	Device_CalculateMeanMatricesComplex(RFIMStruct, d_signalMatrices);

	//--------------------------------



	//Calculate the covariance matrix
	//-------------------------------
	//1. Calculate the outer product of the signal (sampleElements x sampleNumber) * ( sampleNumber x sampleElements)
	//	AKA. signal * (signal)T, where T = transpose, which will give you a (sampleNumber x sampleNumber) matrix as a result

	//Take the outer product of the signal with itself
	cuComplex alpha = make_cuComplex(1.0f / RFIMStruct->h_numberOfSamples, 0);
	cuComplex beta = make_cuComplex(-1, 0);

	uint64_t signalOffset = RFIMStruct->h_valuesPerSample * RFIMStruct->h_numberOfSamples;
	uint64_t covarianceMatrixOffset = RFIMStruct->h_valuesPerSample * RFIMStruct->h_valuesPerSample;

	uint64_t cudaStreamIterator = 0;

	cublasStatus_t cublasError;

	for(uint64_t i = 0; i < RFIMStruct->h_batchSize; ++i)
	{
		//Set the stream for the library
		cublasSetStream_v2(*RFIMStruct->cublasHandle, RFIMStruct->h_cudaStreams[cudaStreamIterator]);

		cublasError = cublasCgemm_v2(*RFIMStruct->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_C,
				RFIMStruct->h_valuesPerSample, RFIMStruct->h_valuesPerSample, RFIMStruct->h_numberOfSamples,
				&alpha, d_signalMatrices + (i * signalOffset), RFIMStruct->h_valuesPerSample,
				d_signalMatrices + (i * signalOffset), RFIMStruct->h_valuesPerSample, &beta,
				RFIMStruct->d_covarianceMatrix + (i * covarianceMatrixOffset), RFIMStruct->h_valuesPerSample);


		if(cublasError != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "Device_CalculateCovarianceMatrixComplex: error calculating the covariance matrix\n");
			exit(1);
		}


		cudaStreamIterator += 1;
		if(cudaStreamIterator >= RFIMStruct->h_cudaStreamsLength)
		{
			cudaStreamIterator = 0;
		}

		/*
		//TODO: DEBUG REMOVE
		cudaError_t cudaError = cudaDeviceSynchronize();
		cublasError = cublasGetError();

		if(cudaError != cudaSuccess || cublasError != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "Device_CalculateCovarianceMatrix 1 error\n");
		}
		*/

	}

}




void Host_CalculateCovarianceMatrix(RFIMMemoryStructCPU* RFIMStruct, float* signalMatrices)
{
	//Calculate the mean matrices
	Host_CalculateMeanMatrices(RFIMStruct, signalMatrices);


	//Calculate the covariance matrices
	//Take the outer product of the signal with itself
	float alpha = 1.0f / RFIMStruct->h_numberOfSamples;
	float beta = -1;

	uint64_t signalOffset = RFIMStruct->h_valuesPerSample * RFIMStruct->h_numberOfSamples;
	uint64_t covarianceMatrixOffset = RFIMStruct->h_valuesPerSample * RFIMStruct->h_valuesPerSample;




	for(uint64_t i = 0; i < RFIMStruct->h_batchSize; ++i)
	{

		cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
				RFIMStruct->h_valuesPerSample, RFIMStruct->h_valuesPerSample, RFIMStruct->h_numberOfSamples,
				alpha, signalMatrices + (i * signalOffset), RFIMStruct->h_valuesPerSample,
				signalMatrices + (i * signalOffset), RFIMStruct->h_valuesPerSample, beta,
				RFIMStruct->h_covarianceMatrix + (i * covarianceMatrixOffset), RFIMStruct->h_valuesPerSample);

	}

}





void Device_EigenvalueSolver(RFIMMemoryStruct* RFIMStruct)
{

	cusolverStatus_t cusolverStatus;

	uint64_t cudaStreamIterator = 0;


	for(uint64_t i = 0; i < RFIMStruct->h_batchSize; ++i)
	{

		//Set the stream
		cusolverDnSetStream(*RFIMStruct->cusolverHandle, RFIMStruct->h_cudaStreams[cudaStreamIterator]);

		//Tell the device to solve the eigenvectors
		cusolverStatus = cusolverDnSgesvd(*RFIMStruct->cusolverHandle, 'A', 'A',
				RFIMStruct->h_valuesPerSample, RFIMStruct->h_valuesPerSample,
				RFIMStruct->d_covarianceMatrix + (i * RFIMStruct->h_covarianceMatrixBatchOffset), RFIMStruct->h_valuesPerSample,
				RFIMStruct->d_S + (i * RFIMStruct->h_SBatchOffset),
				RFIMStruct->d_U + (i * RFIMStruct->h_UBatchOffset), RFIMStruct->h_valuesPerSample,
				RFIMStruct->d_VT + (i * RFIMStruct->h_VTBatchOffset), RFIMStruct->h_valuesPerSample,
				RFIMStruct->d_eigenWorkingSpace + (i * RFIMStruct->h_eigenWorkingSpaceBatchOffset),
				RFIMStruct->h_singleEigWorkingSpaceByteSize,
				NULL,
				RFIMStruct->d_devInfo + (i * RFIMStruct->h_devInfoBatchOffset));


		//Check for startup errors
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


		//Put in a request to copy the devInfo back to the host so you can check it later
		cudaMemcpyAsync(RFIMStruct->h_devInfo + (i * RFIMStruct->h_devInfoBatchOffset),
				RFIMStruct->d_devInfo + (i * RFIMStruct->h_devInfoBatchOffset), sizeof(int),
				cudaMemcpyDeviceToHost, RFIMStruct->h_cudaStreams[cudaStreamIterator]);



		//Iterate to the next stream
		cudaStreamIterator += 1;
		if(cudaStreamIterator >= RFIMStruct->h_cudaStreamsLength)
		{
			cudaStreamIterator = 0;
		}

		/*
		//TODO: DEBUG REMOVE
		cudaError_t cudaError = cudaDeviceSynchronize();
		cublasStatus_t cublasError = cublasGetError();

		if(cudaError != cudaSuccess )
		{
			fprintf(stderr, "Device_EigenvalueSolver 1 error\n");
		}
		*/

	}



	/*
	//TODO: ****************** EXPERIMENT WITH PUTTING THIS AT THE END OF THE RFIM ROUTINE ******************
	//Wait for everything to complete
	for(uint64_t i = 0; i < RFIMStruct->h_cudaStreamsLength; ++i)
	{
		cudaStreamSynchronize(RFIMStruct->h_cudaStreams[i]);
	}


	//Check each devInfo value
	for(uint64_t i = 0; i < RFIMStruct->h_batchSize; ++i)
	{
		if(RFIMStruct->h_devInfo[i] != 0)
		{
			fprintf(stderr, "Device_EigenvalueSolver: Error with the %dth parameter on the %lluth batch\n", RFIMStruct->h_devInfo[i], i);
			exit(1);
		}
	}

*/

	//********************************************************************************************************

	/*
	//TODO: DEBUG REMOVE
	cudaError_t cudaError = cudaDeviceSynchronize();


	if(cudaError != cudaSuccess)
	{
		fprintf(stderr, "Device_EigenvalueSolver 2 error\n");
	}
	*/

}




void Device_EigenvalueSolverBatched(RFIMMemoryStructBatched* RFIMStruct)
{


	//Have to wait for all batched calls to finish
	cudaStreamSynchronize(RFIMStruct->h_cudaStreams[0]);


	cusolverStatus_t cusolverStatus;

	//use streams 1 or greater for memcopies and eigenvalue solving
	uint64_t cudaStreamIterator = 1;

	for(uint64_t i = 0; i < RFIMStruct->h_batchSize; ++i)
	{

		//Set the stream
		cusolverDnSetStream(*RFIMStruct->cusolverHandle, RFIMStruct->h_cudaStreams[cudaStreamIterator]);

		//Tell the device to solve the eigenvectors
		cusolverStatus = cusolverDnSgesvd(*RFIMStruct->cusolverHandle, 'A', 'A',
				RFIMStruct->h_valuesPerSample, RFIMStruct->h_valuesPerSample,
				RFIMStruct->d_covarianceMatrix + (i * RFIMStruct->h_covarianceMatrixBatchOffset), RFIMStruct->h_valuesPerSample,
				RFIMStruct->d_S + (i * RFIMStruct->h_SBatchOffset),
				RFIMStruct->d_U + (i * RFIMStruct->h_UBatchOffset), RFIMStruct->h_valuesPerSample,
				RFIMStruct->d_VT + (i * RFIMStruct->h_VTBatchOffset), RFIMStruct->h_valuesPerSample,
				RFIMStruct->d_eigenWorkingSpace + (i * RFIMStruct->h_eigenWorkingSpaceBatchOffset),
				RFIMStruct->h_singleEigWorkingSpaceByteSize,
				NULL,
				RFIMStruct->d_devInfo + (i * RFIMStruct->h_devInfoBatchOffset));


		//Check for startup errors
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


		//Put in a request to copy the devInfo back to the host so you can check it later
		cudaMemcpyAsync(RFIMStruct->h_devInfo + (i * RFIMStruct->h_devInfoBatchOffset),
				RFIMStruct->d_devInfo + (i * RFIMStruct->h_devInfoBatchOffset), sizeof(int),
				cudaMemcpyDeviceToHost, RFIMStruct->h_cudaStreams[cudaStreamIterator]);



		//Iterate to the next stream
		cudaStreamIterator += 1;
		if(cudaStreamIterator >= RFIMStruct->h_cudaStreamsLength)
		{
			//use streams 1 or greater for memcopies and eigenvalue solving
			cudaStreamIterator = 1;
		}

		/*
		//TODO: DEBUG REMOVE
		cudaError_t cudaError = cudaDeviceSynchronize();
		cublasStatus_t cublasError = cublasGetError();

		if(cudaError != cudaSuccess )
		{
			fprintf(stderr, "Device_EigenvalueSolver 1 error\n");
		}
		*/

	}



	/*
	//TODO: ****************** EXPERIMENT WITH PUTTING THIS AT THE END OF THE RFIM ROUTINE ******************
	//Wait for everything to complete
	for(uint64_t i = 0; i < RFIMStruct->h_cudaStreamsLength; ++i)
	{
		cudaStreamSynchronize(RFIMStruct->h_cudaStreams[i]);
	}


	//Check each devInfo value
	for(uint64_t i = 0; i < RFIMStruct->h_batchSize; ++i)
	{
		if(RFIMStruct->h_devInfo[i] != 0)
		{
			fprintf(stderr, "Device_EigenvalueSolver: Error with the %dth parameter on the %lluth batch\n", RFIMStruct->h_devInfo[i], i);
			exit(1);
		}
	}

*/

	//********************************************************************************************************

	/*
	//TODO: DEBUG REMOVE
	cudaError_t cudaError = cudaDeviceSynchronize();


	if(cudaError != cudaSuccess)
	{
		fprintf(stderr, "Device_EigenvalueSolver 2 error\n");
	}
	*/
}







void Device_EigenvalueSolverComplex(RFIMMemoryStructComplex* RFIMStruct)
{
	cusolverStatus_t cusolverStatus;

	uint64_t cudaStreamIterator = 0;

	for(uint64_t i = 0; i < RFIMStruct->h_batchSize; ++i)
	{

		//Set the stream
		cusolverDnSetStream(*RFIMStruct->cusolverHandle, RFIMStruct->h_cudaStreams[cudaStreamIterator]);

		//Tell the device to solve the eigenvectors
		cusolverStatus = cusolverDnCgesvd(*RFIMStruct->cusolverHandle, 'A', 'A',
				RFIMStruct->h_valuesPerSample, RFIMStruct->h_valuesPerSample,
				RFIMStruct->d_covarianceMatrix + (i * RFIMStruct->h_covarianceMatrixBatchOffset), RFIMStruct->h_valuesPerSample,
				RFIMStruct->d_S + (i * RFIMStruct->h_SBatchOffset),
				RFIMStruct->d_U + (i * RFIMStruct->h_UBatchOffset), RFIMStruct->h_valuesPerSample,
				RFIMStruct->d_VT + (i * RFIMStruct->h_VTBatchOffset), RFIMStruct->h_valuesPerSample,
				RFIMStruct->d_eigenWorkingSpace + (i * RFIMStruct->h_eigenWorkingSpaceBatchOffset),
				RFIMStruct->h_singleEigWorkingSpaceByteSize,
				RFIMStruct->h_rWork + (i * RFIMStruct->h_rWorkBatchOffset),
				RFIMStruct->d_devInfo + (i * RFIMStruct->h_devInfoBatchOffset));


		//Check for startup errors
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


		//Put in a request to copy the devInfo back to the host so you can check it later
		cudaMemcpyAsync(RFIMStruct->h_devInfo + (i * RFIMStruct->h_devInfoBatchOffset),
				RFIMStruct->d_devInfo + (i * RFIMStruct->h_devInfoBatchOffset), sizeof(int),
				cudaMemcpyDeviceToHost, RFIMStruct->h_cudaStreams[cudaStreamIterator]);



		//Iterate to the next stream
		cudaStreamIterator += 1;
		if(cudaStreamIterator >= RFIMStruct->h_cudaStreamsLength)
		{
			cudaStreamIterator = 0;
		}

		/*
		//TODO: DEBUG REMOVE
		cudaError_t cudaError = cudaDeviceSynchronize();
		cublasStatus_t cublasError = cublasGetError();

		if(cudaError != cudaSuccess )
		{
			fprintf(stderr, "Device_EigenvalueSolver 1 error\n");
		}
		*/

	}



	//TODO: ****************** EXPERIMENT WITH PUTTING THIS AT THE END OF THE RFIM ROUTINE ******************
	//Wait for everything to complete
	for(uint64_t i = 0; i < RFIMStruct->h_cudaStreamsLength; ++i)
	{
		cudaStreamSynchronize(RFIMStruct->h_cudaStreams[i]);
	}


	//Check each devInfo value
	for(uint64_t i = 0; i < RFIMStruct->h_batchSize; ++i)
	{
		if(RFIMStruct->h_devInfo[i] != 0)
		{
			fprintf(stderr, "Device_EigenvalueSolver: Error with the %dth parameter on the %lluth batch\n", RFIMStruct->h_devInfo[i], i);
			exit(1);
		}
	}


	//********************************************************************************************************

	/*
	//TODO: DEBUG REMOVE
	cudaError_t cudaError = cudaDeviceSynchronize();


	if(cudaError != cudaSuccess)
	{
		fprintf(stderr, "Device_EigenvalueSolver 2 error\n");
	}
	*/
}




void Host_EigenvalueSolver(RFIMMemoryStructCPU* RFIMStruct)
{
	int info;

	//Compute the SVD for each covariance matrix
	for(uint64_t i = 0; i < RFIMStruct->h_batchSize; ++i)
	{

		info =  LAPACKE_sgesdd(LAPACK_COL_MAJOR, 'A',
				RFIMStruct->h_valuesPerSample, RFIMStruct->h_valuesPerSample,
				RFIMStruct->h_covarianceMatrix + (i * RFIMStruct->h_covarianceMatrixBatchOffset), RFIMStruct->h_valuesPerSample,
				RFIMStruct->h_S + (i * RFIMStruct->h_SBatchOffset),
				RFIMStruct->h_U + (i * RFIMStruct->h_UBatchOffset), RFIMStruct->h_valuesPerSample,
				RFIMStruct->h_VT + (i * RFIMStruct->h_VTBatchOffset), RFIMStruct->h_valuesPerSample);


		//Check to see if everything went ok
		if(info != 0)
		{
			//If info = -i, the i-th parameter had an illegal value
			//If info = i, then sgesdd did not converge, updataing process failed
			fprintf(stderr, "Host_EigenvalueSolver: SVD computation didn't converge. Info: %d\n", info);
			exit(1);
		}

		/*
		//Tell the device to solve the eigenvectors
		cusolverStatus = cusolverDnSgesvd(*RFIMStruct->cusolverHandle, 'A', 'A',
				RFIMStruct->h_valuesPerSample, RFIMStruct->h_valuesPerSample,
				RFIMStruct->d_covarianceMatrix + (i * RFIMStruct->h_covarianceMatrixBatchOffset), RFIMStruct->h_valuesPerSample,
				RFIMStruct->d_S + (i * RFIMStruct->h_SBatchOffset),
				RFIMStruct->d_U + (i * RFIMStruct->h_UBatchOffset), RFIMStruct->h_valuesPerSample,
				RFIMStruct->d_VT + (i * RFIMStruct->h_VTBatchOffset), RFIMStruct->h_valuesPerSample,
				RFIMStruct->d_eigenWorkingSpace + (i * RFIMStruct->h_eigenWorkingSpaceBatchOffset),
				RFIMStruct->h_singleEigWorkingSpaceByteSize,
				NULL,
				RFIMStruct->d_devInfo + (i * RFIMStruct->h_devInfoBatchOffset));
	*/
	}

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



void Device_EigenReductionAndFiltering(RFIMMemoryStruct* RFIMStruct, float* d_originalSignalMatrices, float* d_filteredSignals)
{


	//Set the appropriate number of columns to zero
	uint64_t eigenvectorZeroByteSize = sizeof(float) * RFIMStruct->h_valuesPerSample * RFIMStruct->h_eigenVectorDimensionsToReduce;

	uint64_t cudaStreamIterator = 0;

	for(uint64_t i = 0; i < RFIMStruct->h_batchSize; ++i)
	{
		cudaMemsetAsync(RFIMStruct->d_U + (i * RFIMStruct->h_UBatchOffset),
				0, eigenvectorZeroByteSize, RFIMStruct->h_cudaStreams[cudaStreamIterator]);

		cudaStreamIterator += 1;
		if(cudaStreamIterator >= RFIMStruct->h_cudaStreamsLength)
		{
			cudaStreamIterator = 0;
		}

		/*
		//TODO: DEBUG REMOVE
		cudaError_t cudaError = cudaDeviceSynchronize();
		cublasStatus_t cublasError = cublasGetError();

		if(cudaError != cudaSuccess || cublasError != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "Device_EigenReductionAndFiltering 1 error\n");
		}
		*/


	}



	cublasStatus_t cublasStatus;

	//Projected signal matrix
	//Ps = (Er Transposed) * Os
	float alpha = 1;
	float beta = 0;

	uint64_t originalSignalBatchOffset = RFIMStruct->h_valuesPerSample * RFIMStruct->h_numberOfSamples;

	cudaStreamIterator = 0;


	//Do the projection
	for(uint64_t i = 0; i < RFIMStruct->h_batchSize; ++i)
	{
		//Set the stream
		cublasSetStream_v2(*RFIMStruct->cublasHandle, RFIMStruct->h_cudaStreams[cudaStreamIterator]);



		//compute
		cublasStatus = cublasSgemm_v2(*RFIMStruct->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
				RFIMStruct->h_valuesPerSample, RFIMStruct->h_numberOfSamples, RFIMStruct->h_valuesPerSample,
				&alpha, RFIMStruct->d_U + (i * RFIMStruct->h_UBatchOffset), RFIMStruct->h_valuesPerSample,
				d_originalSignalMatrices + (i * originalSignalBatchOffset), RFIMStruct->h_valuesPerSample, &beta,
				RFIMStruct->d_projectedSignalMatrix + (i * RFIMStruct->h_projectedSignalBatchOffset), RFIMStruct->h_valuesPerSample);


		//Check request status codes
		if(cublasStatus != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "Device_EigenReductionAndFiltering: error calculating the projected signal\n");
			exit(1);
		}


		//Iterate the stream
		cudaStreamIterator += 1;
		if(cudaStreamIterator >= RFIMStruct->h_cudaStreamsLength)
		{
			cudaStreamIterator = 0;
		}

		/*
		//TODO: DEBUG REMOVE
		cudaError_t cudaError = cudaDeviceSynchronize();

		cublasStatus = cublasGetError();

		if(cudaError != cudaSuccess || cublasStatus != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "Device_EigenReductionAndFiltering 2 error\n");
		}
		*/
	}



	//Do the reprojection back
	//final signal matrix
	// Fs = Er * Ps

	cudaStreamIterator = 0;

	for(uint64_t i = 0; i < RFIMStruct->h_batchSize; ++i)
	{

		//Set the stream
		cublasSetStream_v2(*RFIMStruct->cublasHandle, RFIMStruct->h_cudaStreams[cudaStreamIterator]);


		cublasStatus_t = cublasSgemm_v2(*RFIMStruct->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
				RFIMStruct->h_valuesPerSample, RFIMStruct->h_numberOfSamples, RFIMStruct->h_valuesPerSample,
				&alpha, RFIMStruct->d_U + (i * RFIMStruct->h_UBatchOffset), RFIMStruct->h_valuesPerSample,
				RFIMStruct->d_projectedSignalMatrix + (i * RFIMStruct->h_projectedSignalBatchOffset), RFIMStruct->h_valuesPerSample, &beta,
				d_filteredSignals + (i * originalSignalBatchOffset), RFIMStruct->h_valuesPerSample);




		if(cublasStatus != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "Device_EigenReductionAndFiltering: error calculating the filtered signal\n");
			exit(1);
		}


		//Iterate the stream
		cudaStreamIterator += 1;
		if(cudaStreamIterator >= RFIMStruct->h_cudaStreamsLength)
		{
			cudaStreamIterator = 0;
		}

		/*
		//TODO: DEBUG REMOVE
		cudaError_t cudaError = cudaDeviceSynchronize();
		cublasStatus = cublasGetError();

		if(cudaError != cudaSuccess || cublasStatus != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "Device_EigenReductionAndFiltering 3 error\n");
		}
		*/
	}



}




void Device_EigenReductionAndFilteringBatched(RFIMMemoryStructBatched* RFIMStruct, float** d_originalSignalMatrices, float** d_filteredSignals)
{
	//Set the appropriate number of columns to zero
	uint64_t eigenvectorZeroByteSize = sizeof(float) * RFIMStruct->h_valuesPerSample * RFIMStruct->h_eigenVectorDimensionsToReduce;

	//use streams 1 or greater for memcopies and eigenvalue solving
	uint64_t cudaStreamIterator = 1;

	for(uint64_t i = 0; i < RFIMStruct->h_batchSize; ++i)
	{
		cudaMemsetAsync(RFIMStruct->d_U + (i * RFIMStruct->h_UBatchOffset),
				0, eigenvectorZeroByteSize, RFIMStruct->h_cudaStreams[cudaStreamIterator]);

		cudaStreamIterator += 1;
		if(cudaStreamIterator >= RFIMStruct->h_cudaStreamsLength)
		{
			cudaStreamIterator = 1;
		}

		/*
		//TODO: DEBUG REMOVE
		cudaError_t cudaError = cudaDeviceSynchronize();
		cublasStatus_t cublasError = cublasGetError();

		if(cudaError != cudaSuccess || cublasError != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "Device_EigenReductionAndFiltering 1 error\n");
		}
		*/
	}


	//Wait for all the memsets to complete before stating the compute
	for(uint64_t i = 1; i < RFIMStruct->h_cudaStreamsLength; ++i)
	{
		cudaStreamSynchronize(RFIMStruct->h_cudaStreams[i]);
	}



	cublasStatus_t cublasStatus;

	//Projected signal matrix
	//Ps = (Er Transposed) * Os
	float alpha = 1;
	float beta = 0;


	//Do the projection
	//compute

	//Set the stream to stream zero, so it's not on the default stream
	cublasSetStream_v2(*RFIMStruct->cublasHandle, RFIMStruct->h_cudaStreams[0]);

	cublasStatus = cublasSgemmBatched(*RFIMStruct->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
			RFIMStruct->h_valuesPerSample, RFIMStruct->h_numberOfSamples, RFIMStruct->h_valuesPerSample,
			&alpha, (const float**)RFIMStruct->d_UBatched, RFIMStruct->h_valuesPerSample,
			(const float**)d_originalSignalMatrices, RFIMStruct->h_valuesPerSample, &beta,
			RFIMStruct->d_projectedSignalMatrixBatched, RFIMStruct->h_valuesPerSample,
			RFIMStruct->h_batchSize);


	//Check request status codes
	if(cublasStatus != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "Device_EigenReductionAndFiltering: error calculating the projected signal\n");
		exit(1);
	}


	/*
	//TODO: DEBUG REMOVE
	cudaError_t cudaError = cudaDeviceSynchronize();

	cublasStatus = cublasGetError();

	if(cudaError != cudaSuccess || cublasStatus != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "Device_EigenReductionAndFiltering 2 error\n");
	}
	*/



	//Do the reprojection back
	//final signal matrix
	// Fs = Er * Ps



	cublasStatus_t = cublasSgemmBatched(*RFIMStruct->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			RFIMStruct->h_valuesPerSample, RFIMStruct->h_numberOfSamples, RFIMStruct->h_valuesPerSample,
			&alpha, (const float**)RFIMStruct->d_UBatched, RFIMStruct->h_valuesPerSample,
			(const float**)RFIMStruct->d_projectedSignalMatrixBatched, RFIMStruct->h_valuesPerSample, &beta,
			d_filteredSignals, RFIMStruct->h_valuesPerSample,
			RFIMStruct->h_batchSize);




	if(cublasStatus != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "Device_EigenReductionAndFiltering: error calculating the filtered signal\n");
		exit(1);
	}

	/*
	//TODO: DEBUG REMOVE
	cudaError_t cudaError = cudaDeviceSynchronize();
	cublasStatus = cublasGetError();

	if(cudaError != cudaSuccess || cublasStatus != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "Device_EigenReductionAndFiltering 3 error\n");
	}
	*/

}




void Device_EigenReductionAndFilteringComplex(RFIMMemoryStructComplex* RFIMStruct, cuComplex* d_originalSignalMatrices, cuComplex* d_filteredSignals)
{
	//Set the appropriate number of columns to zero
	uint64_t eigenvectorZeroByteSize = sizeof(cuComplex) * RFIMStruct->h_valuesPerSample * RFIMStruct->h_eigenVectorDimensionsToReduce;

	uint64_t cudaStreamIterator = 0;

	for(uint64_t i = 0; i < RFIMStruct->h_batchSize; ++i)
	{
		cudaMemsetAsync(RFIMStruct->d_U + (i * RFIMStruct->h_UBatchOffset),
				0, eigenvectorZeroByteSize, RFIMStruct->h_cudaStreams[cudaStreamIterator]);

		cudaStreamIterator += 1;
		if(cudaStreamIterator >= RFIMStruct->h_cudaStreamsLength)
		{
			cudaStreamIterator = 0;
		}

	}


	cublasStatus_t cublasStatus;

	//Projected signal matrix
	//Ps = (Er Transposed) * Os
	cuComplex alpha = make_cuComplex(1, 0);
	cuComplex beta = make_cuComplex(0, 0);

	uint64_t originalSignalBatchOffset = RFIMStruct->h_valuesPerSample * RFIMStruct->h_numberOfSamples;

	cudaStreamIterator = 0;


	//Do the projection
	for(uint64_t i = 0; i < RFIMStruct->h_batchSize; ++i)
	{
		//Set the stream
		cublasSetStream_v2(*RFIMStruct->cublasHandle, RFIMStruct->h_cudaStreams[cudaStreamIterator]);



		//compute
		cublasStatus = cublasCgemm_v2(*RFIMStruct->cublasHandle, CUBLAS_OP_C, CUBLAS_OP_N,
				RFIMStruct->h_valuesPerSample, RFIMStruct->h_numberOfSamples, RFIMStruct->h_valuesPerSample,
				&alpha, RFIMStruct->d_U + (i * RFIMStruct->h_UBatchOffset), RFIMStruct->h_valuesPerSample,
				d_originalSignalMatrices + (i * originalSignalBatchOffset), RFIMStruct->h_valuesPerSample, &beta,
				RFIMStruct->d_projectedSignalMatrix + (i * RFIMStruct->h_projectedSignalBatchOffset), RFIMStruct->h_valuesPerSample);


		//Check request status codes
		if(cublasStatus != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "Device_EigenReductionAndFiltering: error calculating the projected signal\n");
			exit(1);
		}


		//Iterate the stream
		cudaStreamIterator += 1;
		if(cudaStreamIterator >= RFIMStruct->h_cudaStreamsLength)
		{
			cudaStreamIterator = 0;
		}

	}



	//Do the reprojection back
	//final signal matrix
	// Fs = Er * Ps

	cudaStreamIterator = 0;

	for(uint64_t i = 0; i < RFIMStruct->h_batchSize; ++i)
	{

		//Set the stream
		cublasSetStream_v2(*RFIMStruct->cublasHandle, RFIMStruct->h_cudaStreams[cudaStreamIterator]);


		cublasStatus_t = cublasCgemm_v2(*RFIMStruct->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
				RFIMStruct->h_valuesPerSample, RFIMStruct->h_numberOfSamples, RFIMStruct->h_valuesPerSample,
				&alpha, RFIMStruct->d_U + (i * RFIMStruct->h_UBatchOffset), RFIMStruct->h_valuesPerSample,
				RFIMStruct->d_projectedSignalMatrix + (i * RFIMStruct->h_projectedSignalBatchOffset), RFIMStruct->h_valuesPerSample, &beta,
				d_filteredSignals + (i * originalSignalBatchOffset), RFIMStruct->h_valuesPerSample);




		if(cublasStatus != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "Device_EigenReductionAndFiltering: error calculating the filtered signal\n");
			exit(1);
		}


		//Iterate the stream
		cudaStreamIterator += 1;
		if(cudaStreamIterator >= RFIMStruct->h_cudaStreamsLength)
		{
			cudaStreamIterator = 0;
		}

	}



}



void Host_EigenReductionAndFiltering(RFIMMemoryStructCPU* RFIMStruct, float* h_originalSignalMatrices, float* h_filtredSignalMatrices)
{

	//Set the appropriate number of columns to zero
	uint64_t eigenvectorZeroByteSize = sizeof(float) * RFIMStruct->h_valuesPerSample * RFIMStruct->h_eigenVectorDimensionsToReduce;

	for(uint64_t i = 0; i < RFIMStruct->h_batchSize; ++i)
	{
		memset(RFIMStruct->h_U + (i * RFIMStruct->h_UBatchOffset), 0, eigenvectorZeroByteSize);

		/*
		cudaMemsetAsync(RFIMStruct->d_U + (i * RFIMStruct->h_UBatchOffset),
				0, eigenvectorZeroByteSize, RFIMStruct->h_cudaStreams[cudaStreamIterator]);
		*/
	}




	float alpha = 1;
	float beta = 0;

	uint64_t originalSignalBatchOffset = RFIMStruct->h_valuesPerSample * RFIMStruct->h_numberOfSamples;


	//Do the projection
	for(uint64_t i = 0; i < RFIMStruct->h_batchSize; ++i)
	{

		//Projected signal matrix
		//Ps = (Er Transposed) * Os
		cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
				RFIMStruct->h_valuesPerSample, RFIMStruct->h_numberOfSamples, RFIMStruct->h_valuesPerSample,
				alpha, RFIMStruct->h_U + (i * RFIMStruct->h_UBatchOffset), RFIMStruct->h_valuesPerSample,
				h_originalSignalMatrices + (i * originalSignalBatchOffset), RFIMStruct->h_valuesPerSample, beta,
				RFIMStruct->h_projectedSignalMatrix + (i * RFIMStruct->h_projectedSignalBatchOffset), RFIMStruct->h_valuesPerSample);


		/*
		cublasStatus = cublasSgemm_v2(*RFIMStruct->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
				RFIMStruct->h_valuesPerSample, RFIMStruct->h_numberOfSamples, RFIMStruct->h_valuesPerSample,
				&alpha, RFIMStruct->d_U + (i * RFIMStruct->h_UBatchOffset), RFIMStruct->h_valuesPerSample,
				d_originalSignalMatrices + (i * originalSignalBatchOffset), RFIMStruct->h_valuesPerSample, &beta,
				RFIMStruct->d_projectedSignalMatrix + (i * RFIMStruct->h_projectedSignalBatchOffset), RFIMStruct->h_valuesPerSample);

		*/


		//Do the reprojection back
		//final signal matrix
		// Fs = Er * Ps

		cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
				RFIMStruct->h_valuesPerSample, RFIMStruct->h_numberOfSamples, RFIMStruct->h_valuesPerSample,
				alpha,  RFIMStruct->h_U + (i * RFIMStruct->h_UBatchOffset), RFIMStruct->h_valuesPerSample,
				RFIMStruct->h_projectedSignalMatrix + (i * RFIMStruct->h_projectedSignalBatchOffset), RFIMStruct->h_valuesPerSample, beta,
				h_filtredSignalMatrices + (i * originalSignalBatchOffset), RFIMStruct->h_valuesPerSample);


		/*
		cublasStatus_t = cublasSgemm_v2(*RFIMStruct->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
				RFIMStruct->h_valuesPerSample, RFIMStruct->h_numberOfSamples, RFIMStruct->h_valuesPerSample,
				&alpha, RFIMStruct->d_U + (i * RFIMStruct->h_UBatchOffset), RFIMStruct->h_valuesPerSample,
				RFIMStruct->d_projectedSignalMatrix + (i * RFIMStruct->h_projectedSignalBatchOffset), RFIMStruct->h_valuesPerSample, &beta,
				d_filteredSignals + (i * originalSignalBatchOffset), RFIMStruct->h_valuesPerSample);

		*/
	}





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


