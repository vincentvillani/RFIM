/*
 * UtilityFunctions.h
 *
 *  Created on: 17/03/2016
 *      Author: vincentvillani
 */

#ifndef UTILITYFUNCTIONS_H_
#define UTILITYFUNCTIONS_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>

#include <string>


//Write a host signal matrix to a file

float* Utility_GenerateWhiteNoiseHost(uint64_t length, float mean, float stdDev);
float Utility_GenerateSingleWhiteNoiseValueHost(float mean, float stdDev);

float Utility_Mean(float* h_signal, uint64_t signalLength);
float* Utility_SubSignalMean(float* h_signal, uint64_t h_valuesPerSample, uint64_t h_numberOfSamples);

float Utility_SignalToNoiseRatio(float* h_signal, uint64_t signalLength, float signalAmplitude);
float* Utility_SubSignalSignalToNoiseRatio(float* h_signal, uint64_t h_valuesPerSample, uint64_t h_numberOfSamples, float signalAmplitude);

//coefficent of cross correlation
//Doesn't support delay
//Doesn't support signals of different lengths
//Does only one job basically
//float Utility_CoefficentOfCrossCorrelation(float* h_firstSignal, float* h_secondSignal, uint64_t h_signalLength);
float* Utility_CoefficentOfCrossCorrelation(float* h_multiplexedSignal, float* h_secondSignal,
		uint64_t h_valuesPerSample, uint64_t h_numberOfSamples, uint64_t h_secondSignalLength);

float Utility_Variance(float* h_signal, uint64_t signalLength);
float* Utility_SubSignalVariance(float* h_signal, uint64_t h_valuesPerSample, uint64_t h_numberOfSamples);

void Utility_WriteSignalMatrixToFile(const std::string filename, float* h_rowMajorSignalMatrix, uint64_t rows, uint64_t columns);
void Utility_DeviceWriteSignalMatrixToFile(const std::string filename, float* d_rowMajorSignalMatrix, uint64_t rows, uint64_t columns, bool transpose);




#endif /* UTILITYFUNCTIONS_H_ */
