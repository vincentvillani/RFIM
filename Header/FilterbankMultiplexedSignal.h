/*
 * MutliplexedSignal.h
 *
 *  Created on: 3 May 2016
 *      Author: vincent
 */

#ifndef MUTLIPLEXEDSIGNAL_H_
#define MUTLIPLEXEDSIGNAL_H_

#include <stdio.h>
#include <stdint.h>

#include "Sigproc/SigprocFilterbank.h"


//Represents a multiplexed filterbank signal
typedef struct FilterbankMultiplexedSignal
{
	uint32_t signalNum; //number of signals combined into this one signal
	uint64_t frequencyChannelNum;
	uint64_t perSignalSampleNum; //Number of samples per signal
	uint64_t totalSampleNum; //Total number of samples across the signal

	float* signalData;


}MultiplexedSignal;






FilterbankMultiplexedSignal* FMS_Create(uint32_t signalNum, uint64_t frequencyChannelNum, uint64_t perSignalSampleNum);
void FMS_Destroy(FilterbankMultiplexedSignal* FMS);

void FMS_AddSigprocFilterbankSignal(FilterbankMultiplexedSignal* FMS, SigprocFilterbank* sigprocSignal, uint32_t signalIndex);


#endif /* MUTLIPLEXEDSIGNAL_H_ */
