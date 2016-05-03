

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



void FMS_AddSigprocFilterbankSignal(FilterbankMultiplexedSignal* FMS, SigprocFilterbank* sigprocSignal, uint32_t signalIndex)
{
	uint32_t nChan = sigprocSignal->get_nchans(); //Number of frequency channels in the signal
	uint64_t nSample = sigprocSignal->get_nsamps(); //Number of time samples in the signal

	unsigned char* sigprocData = sigprocSignal->get_data();


	//Go through the sigproc filterbank data and multiplex it into the FMS signal
	//It goes through the sigproc file linearly to get some good cache performance, can't really do anything about the FMS data in terms of cache?

	//For each sigproc sample
	for(uint64_t currentSigprocSampleIndex = 0; currentSigprocSampleIndex < nSample; ++currentSigprocSampleIndex)
	{
		//For each frequency value for the current sample
		for(uint64_t currentSigprocFreqIndex = 0; currentSigprocFreqIndex < nChan; ++currentSigprocFreqIndex)
		{

			//Time sample offset + frequency channel offset
			uint64_t sigprocIndex = currentSigprocSampleIndex * nChan + currentSigprocFreqIndex;

			//Frequency channel offset + time sample offset + signal offset
			uint64_t FMSIndex = (currentSigprocFreqIndex * nSample * FMS->signalNum) + (currentSigprocSampleIndex * FMS->signalNum) + signalIndex;

			//multiplex the current sigproc data into the FMS signal
			FMS->signalData[FMSIndex] = (float)sigprocData[sigprocIndex];
		}

	}


	/*
	//For each frequency channel
	for(uint32_t currentFreqIndex = 0; currentFreqIndex < nChan; ++currentFreqIndex)
	{
		//For each sample in the channel
		for(uint64_t currentSampleIndex = 0; currentSampleIndex < nSample; ++currentSampleIndex)
		{
			//Frequency channel offset + current sample offset + signal index
			uint64_t currentFMSIndex = (currentFreqIndex * nSample) + (currentSampleIndex * FMS->signalNum) + signalIndex;\
			uint64_t currentSigprocIndex = (curr)

			FMS->signalData[currentFMSIndex]
		}
	}
	*/

}
