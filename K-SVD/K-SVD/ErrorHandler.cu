/*------------------------------*/
/*   ERROR HANDLING NAMESPACE   */
/*       IMPLEMENTATION         */
/*------------------------------*/
#include "GlobalDeclarations.cuh"
#include "curand.h"
#include "cusolverDn.h"
#include "ErrorHandler.cuh"


/*-------------------------------*/
/* Error string from cublas call */
/*-------------------------------*/
const char* ErrorHandler::cublasGetErrorString(cublasStatus_t status) {
	switch (status) {
		case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
		case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
		case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
		case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
		case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
		case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
		case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
		case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
	}
	return "UNKNOWN ERROR";
}

/*---------------------------------*/
/* Error string from cusolver call */
/*---------------------------------*/
const char* cusolverGetErrorString(cusolverStatus_t status) {
	switch (status) {
		case CUSOLVER_STATUS_SUCCESS: return "CUSOLVER_STATUS_SUCCESS";
		case CUSOLVER_STATUS_NOT_INITIALIZED: return "CUSOLVER_STATUS_NOT_INITIALIZED";
		case CUSOLVER_STATUS_ALLOC_FAILED: return "CUSOLVER_STATUS_ALLOC_FAILED";
		case CUSOLVER_STATUS_INVALID_VALUE: return "CUSOLVER_STATUS_INVALID_VALUE";
		case CUSOLVER_STATUS_ARCH_MISMATCH: return "CUSOLVER_STATUS_ARCH_MISMATCH";
		case CUSOLVER_STATUS_EXECUTION_FAILED: return "CUSOLVER_STATUS_execution failed";
		case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSOLVER_STATUS_Matrix not supported";
		case CUSOLVER_STATUS_INTERNAL_ERROR: return "CUSOLVER_STATUS_INTERNAL_ERROR";
	}
	return "UNKNOWN ERROR";
}

/*-------------------------------*/
/* Error string from curand call */
/*-------------------------------*/
const char* curandGetErrorString(curandStatus_t error) {
	switch (error) {
		case CURAND_STATUS_SUCCESS: return "CURAND_STATUS_SUCCESS";
		case CURAND_STATUS_VERSION_MISMATCH: return "CURAND_STATUS_VERSION_MISMATCH";
		case CURAND_STATUS_NOT_INITIALIZED: return "CURAND_STATUS_NOT_INITIALIZED";
		case CURAND_STATUS_ALLOCATION_FAILED: return "CURAND_STATUS_ALLOCATION_FAILED";
		case CURAND_STATUS_TYPE_ERROR: return "CURAND_STATUS_TYPE_ERROR";
		case CURAND_STATUS_OUT_OF_RANGE: return "CURAND_STATUS_OUT_OF_RANGE";
		case CURAND_STATUS_LENGTH_NOT_MULTIPLE: return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
		case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
		case CURAND_STATUS_LAUNCH_FAILURE: return "CURAND_STATUS_LAUNCH_FAILURE";
		case CURAND_STATUS_PREEXISTING_FAILURE: return "CURAND_STATUS_PREEXISTING_FAILURE";
		case CURAND_STATUS_INITIALIZATION_FAILED: return "CURAND_STATUS_INITIALIZATION_FAILED";
		case CURAND_STATUS_ARCH_MISMATCH: return "CURAND_STATUS_ARCH_MISMATCH";
		case CURAND_STATUS_INTERNAL_ERROR: return "CURAND_STATUS_INTERNAL_ERROR";
	}
	return "UNKNOWN ERROR";
}
