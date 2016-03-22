

#include "../Header/Kernels.h"

#include <stdint.h>

/*
row_index(i, M):
    ii = M(M+1)/2-1-i
    K = floor((sqrt(8ii+1)-1)/2)
    return M-1-K

or


unsigned int row_index( unsigned int i, unsigned int M ){
    double m = M;
    double row = (-2*m - 1 + sqrt( (4*m*(m+1) - 8*(double)i - 7) )) / -2;
    if( row == (double)(int) row ) row -= 1;
    return (unsigned int) row;
}


unsigned int column_index( unsigned int i, unsigned int M ){
    unsigned int row = row_index( i, M);
    return  i - M * row + row*(row+1) / 2;
}
*/



//Assumes square matrices
__device__ __host__ uint64_t upperTriangularLength(unsigned int numRows)
{
	return (numRows * (numRows + 1)) / 2;
}



//Calculates the mean of an input array in parallel, in place in the input array
__global__ void parallelMeanUnroll2(float* d_inputArray, uint64_t inputLength, float* d_outputMean)
{
	uint32_t localThreadIndex = threadIdx.x;
	uint32_t sumDataIndex = blockIdx.x * blockDim.x * 2 + localThreadIndex; //The index of the piece of data that I will sum into this current block
	uint32_t globalThreadIndex = blockDim.x * blockIdx.x + localThreadIndex;

	//calculate a pointer to this threadBlocks data
	float* localBlockPointer = d_inputArray + blockIdx.x * blockDim.x * 2;

	//Add the next blockDim.x's worth of data into this block before we start reducing
	//Bounds checking
	if(sumDataIndex + blockDim.x < inputLength)
	{
		d_inputArray[sumDataIndex] += d_inputArray[sumDataIndex + blockDim.x];
	}

	//Wait for all threads on this block to complete
	__syncthreads();

	//Start reducing
	//In-place, strided, reduction
	for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (localThreadIndex < stride)
		{
			localBlockPointer[localThreadIndex] += localBlockPointer[localThreadIndex + stride];
		}
	}

	//Wait for all threads on this block to complete
	__syncthreads();

	//If this is the thread with the global index of one, calculate the mean
	if(globalThreadIndex == 0)
	{
		//Clear the output just incase it isn't already
		*d_outputMean = 0;

		for(uint32_t i = 0; i < gridDim.x; ++i)
		{
			*d_outputMean += d_inputArray[ i * blockDim.x * 2]; //Times 2 because we take in 'two blocks' worth of data for each actual block
		}

		*d_outputMean =  *d_outputMean / (inputLength - 1);

		//printf("Mean: %f\n", *d_outputMean);
	}

}



__global__ void subtractMean(float* d_inputArray, uint64_t inputLength, float d_mean)
{
	uint32_t globalThreadIndex = blockDim.x * blockIdx.x + threadIdx.x;

	if(globalThreadIndex >= inputLength)
		return;

	d_inputArray[globalThreadIndex] -= d_mean;
}



__global__ void outerProductSmartBruteForce(float* resultMatrix, float* vec, int vectorLength)
{
	int col = (blockIdx.x * blockDim.x) + threadIdx.x; //column
	int row = (blockIdx.y * blockDim.y) + threadIdx.y; //row


	//check bounds
	if(row >= vectorLength || col >= vectorLength || row > col)
		return;

	int index = (row * vectorLength + col) - (row * (row + 1)) / 2;

	resultMatrix[index] += vec[row] * vec[col];

}


__global__ void outerProductSmartBruteForceLessThreads(float* resultMatrix, float* vec, uint64_t vectorLength)
{
	int col = (blockIdx.x * blockDim.x) + threadIdx.x; //column
	int row = (blockIdx.y * blockDim.y) + threadIdx.y; //row

	//check bounds
	if(row >= vectorLength || col >= vectorLength)
		return;

	//transpose
	if(row > col)
	{
		row = vectorLength - row;
		col = row + col;
	}

	int index = (row * vectorLength + col) - (row * (row + 1)) / 2;

	resultMatrix[index] = vec[row] * vec[col];
}


//Specialised outer product for DSPSR
__global__ void outerProductUpperTri(cuFloatComplex* resultMatrix, cuFloatComplex* vec, unsigned int vectorLength)
{
	int col = (blockIdx.x * blockDim.x) + threadIdx.x; //column
	int row = (blockIdx.y * blockDim.y) + threadIdx.y; //row

	//check bounds
	if(row >= vectorLength || col >= vectorLength)
		return;

	//transpose
	if(row > col)
	{
		row = vectorLength - row;
		col = row + col;
	}

	int index = (row * vectorLength + col) - (row * (row + 1)) / 2;

	resultMatrix[index] = cuCaddf(resultMatrix[index], cuCmulf(vec[row], vec[col]));
}


__global__ void normalise(float* result, unsigned int resultLength, float* amps, unsigned int* hits)
{
	int absoluteThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;

	if(absoluteThreadIdx > resultLength)
		return;

	result[absoluteThreadIdx] = amps[absoluteThreadIdx] / hits[absoluteThreadIdx / 4];
}



__global__ void setDiagonalToZero(float* d_matrix, uint64_t columnsAndRows)
{
	int absoluteThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;

	//Check for out of bounds
	if(absoluteThreadIdx >= columnsAndRows)
		return;

	//set diagonal element to zero
	int matrixIndex = absoluteThreadIdx * columnsAndRows + absoluteThreadIdx;
	d_matrix[matrixIndex] = 0;
}



