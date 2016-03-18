/*
 * RFIMHelperFunctions.h
 *
 *  Created on: 17/03/2016
 *      Author: vincentvillani
 */

#ifndef RFIMHELPERFUNCTIONS_H_
#define RFIMHELPERFUNCTIONS_H_

#include <stdint.h>

/*
	Description:
		Generates a signal containing only white noise.

	Params:

		h_valuesPerSample: The number of values generated for each sample.

		h_numberOfSamples: The number of samples in the signal

		The total length of the signal can be found by multiplying h_valuesPerSample and h_numberOfSamples

	Returns:

		Returns an array of floats on the DEVICE. This contains the generated signal.
*/

float* Device_GenerateWhiteNoiseSignal(uint64_t h_valuesPerSample, uint64_t h_numberOfSamples);



/*
	Description:
		Calculates the covariance matrix for a given signal

	Params:

		d_signalMatrix:
			An array on the DEVICE containing the signal values, it's total length can be calculated by valuesPerSample * sampleNum
			This signal is unchanged by calling this function.
			THIS SIGNAL SHOULD BE A COLUMN-MAJOR MATRIX

		h_valuesPerSample:
			The number of values for a given sample in the signal, intended for use with a multibeam instrument.
			This is needed because there can be more than one measurement/value per sample. (multibeam instruments)
			I.E. for each 'sample', there will be 'valuesPerSample' number of values

		h_numberOfSamples:
			The number of samples in the signal. Each sample contains 'valuesPerSample' values per sample.


	Returns:

		An array of floats on the DEVICE that make up the covariance matrix. This matrix should be a symmetric h_valuesPerSample x h_valuesPerSample matrix.
		Only the upper trianglar contains values, the lower trianglar part is just zero (memory is allocated to store the zeroes and the lower trianglar part is set to zero)

*/
float* Device_CalculateCovarianceMatrix(const float* d_signalMatrix, uint64_t h_valuesPerSample, uint64_t h_numberOfSamples);





/*
	Description:
		Performs an matrix transpose

	Params:

		d_matrix: The matri

	returns:
		Device pointer to the transposed matrix. Remember to free this when you are done with it.

 */

float* Device_MatrixTranspose(float* d_matrix, uint64_t rowNum, uint64_t colNum);



//TODO: Debug - Remove this
float* DEBUG_CALCULATE_MEAN_MATRIX(float* d_signalMatrix, uint64_t h_valuesPerSample, uint64_t h_numberOfSamples);





#endif /* RFIMHELPERFUNCTIONS_H_ */