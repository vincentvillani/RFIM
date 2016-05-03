

#include "../Header/FilterbankMultiplexedSignal.h"


FilterbankMultiplexedSignal* FMS_Create(uint32_t signalNum, uint64_t frequencyChannelNum, uint64_t perSignalSampleNum)
{
	FilterbankMultiplexedSignal* result = (FilterbankMultiplexedSignal*)malloc(sizeof(FilterbankMultiplexedSignal));

	result->signalNum = signalNum;
	result->frequencyChannelNum = frequencyChannelNum;
	result->perSignalSampleNum = perSignalSampleNum;
	result->totalSampleNum = signalNum * frequencyChannelNum * perSignalSampleNum;

	result->signalData = (float*)malloc(sizeof(float) * result->totalSampleNum);

	return result;
}



void FMS_Destroy(FilterbankMultiplexedSignal* FMS)
{
	free(FMS->signalData);
	free(FMS);
}



void FMS_AddSigprocFilterbankSignal(SigprocFilterbank* signal, uint32_t signalIndex)
{

}
