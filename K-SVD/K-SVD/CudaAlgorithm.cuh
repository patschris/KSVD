/*------------------------------*/
/*  CUDA ALGORITHM BASE CLASS   */
/*------------------------------*/
#pragma once
#ifndef _CUALG_
#define _CUALG_


// Base class inherited from all 
// algorithms implemented in the GPU
class CudaAlgorithm {		

	protected:
		// Constructor
		CudaAlgorithm(DeviceMemory<datatype>*, HostMemory<datatype>*);
		// Destructor
		~CudaAlgorithm();

		/*-------------------*/
		/* Protected Members */
		/*-------------------*/
		// References to the device and
		// host memory objects so that all
		// functions have access
		DeviceMemory<datatype>&	deviceMemory;
		HostMemory<datatype>&	hostMemory;
		// Cublas Handle
		cublasHandle_t			CBhandle;
		// Error flag
		bool					valid;

		/*-------------------*/
		/* Protected Methods */
		/*-------------------*/
		// Evaluates the flag for
		// normal operation
		bool	isValid();

};

#endif // !_CUALG_