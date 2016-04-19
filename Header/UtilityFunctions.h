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

float Utility_SignalToNoiseRatio(float* h_signal, uint64_t signalLength, float signalAmplitude);
float Utility_Variance(float* h_signal, uint64_t signalLength);

void Utility_WriteSignalMatrixToFile(const std::string filename, float* h_rowMajorSignalMatrix, uint64_t rows, uint64_t columns);
void Utility_DeviceWriteSignalMatrixToFile(const std::string filename, float* d_rowMajorSignalMatrix, uint64_t rows, uint64_t columns, bool transpose);




#endif /* UTILITYFUNCTIONS_H_ */
