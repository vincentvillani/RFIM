/*
 * Benchmark.h
 *
 *  Created on: 31 Mar 2016
 *      Author: vincentvillani
 */

#ifndef BENCHMARK_H_
#define BENCHMARK_H_

#include "RFIMMemoryStruct.h"

void Benchmark(RFIMMemoryStruct* RFIM, float* d_signal, float* d_filteredSignal, uint32_t calculationNum, uint32_t iterations);



#endif /* BENCHMARK_H_ */
