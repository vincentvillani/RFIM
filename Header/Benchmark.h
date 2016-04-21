/*
 * Benchmark.h
 *
 *  Created on: 31 Mar 2016
 *      Author: vincentvillani
 */

#ifndef BENCHMARK_H_
#define BENCHMARK_H_

#include "RFIMMemoryStruct.h"

void Benchmark();
void BenchmarkComplex();

//Benchmark the actual effectiveness of removing RFIM

//Adds a sine wave with constant amplitudes, frequency and equal phase to multiple beams.
//3 beams, 6, beams, 9 beams, 12 beams
//13 beams total
void BenchmarkRFIMConstantInterferor();

//Add different sine waves with different freq and amplitudes and try removing different amounts of eigenvectors
void BenchmarkRFIMVariableInterferorVariableEigenvectorRemoval();



#endif /* BENCHMARK_H_ */
