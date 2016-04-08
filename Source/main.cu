
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>


#include <string>
#include <stdint.h>

#include "../Header/UnitTests.h"
#include "../Header/RFIMHelperFunctions.h"
#include "../Header/UtilityFunctions.h"
#include "../Header/RFIM.h"
#include "../Header/Benchmark.h"


//TODO: Look into ways of using less memory
//TODO: Write a unit test to test eigenvector removal gives the correct result



int main(int argc, char **argv)
{
	//Run all the unit tests
	RunAllUnitTests();

	/*
	uint32_t h_valuesPerSample = 26;
	uint32_t h_numberOfSamples = 1024;
	uint32_t h_batchSize = 512;
	uint32_t h_batchNum = 1;
	*/





	//1. Run RFIM benchmark
	//--------------------------
	//Benchmark();



	//2. Free everything
	//--------------------------



	return 0;
}
