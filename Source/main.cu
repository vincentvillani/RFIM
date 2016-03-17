
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

#include <cuda.h>
#include <curand.h>
#include <cublas.h>

#include <stdint.h>

#include "../Header/Kernels.h"
#include "../Header/UnitTests.h"
#include "../Header/CudaMacros.h"



int main(int argc, char **argv)
{
	//Run all the unit tests
	RunAllUnitTests();




	return 0;
}
