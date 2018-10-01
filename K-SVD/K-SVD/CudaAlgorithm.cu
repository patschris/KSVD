/*------------------------------*/
/*   CUDA ALGORITHM BASE CLASS  */
/*       IMPLEMENTATION         */
/*------------------------------*/
#include <vector>
#include <thrust/device_vector.h>
#include "Globals.cuh"
#include "GlobalDeclarations.cuh"
#include "FileManager.cuh"
#include "HostMemory.cuh"
#include "DeviceMemory.cuh"
#include "CudaAlgorithm.cuh"
#include "curand.h"
#include "cusolverDn.h"
#include "ErrorHandler.cuh"

using namespace std;


/*---------------*/
/* Constructor   */
/*---------------*/
CudaAlgorithm::CudaAlgorithm(DeviceMemory<datatype>* devptr, HostMemory<datatype>* hostptr)
	: deviceMemory(*devptr), hostMemory(*hostptr) {

	valid = true;
	// Create handle
	cublasStatus_t cubstat;
	if ((cubstat = cublasCreate(&(this->CBhandle))) != CUBLAS_STATUS_SUCCESS) {
		cerr << "Cublas Create failed: " << ErrorHandler::cublasGetErrorString(cubstat) << endl;
		valid = false;
	}
}

/*--------------*/
/* Destructor   */
/*--------------*/
CudaAlgorithm::~CudaAlgorithm() {
	// Destroy handle
	cublasStatus_t cubstat;
	if ((cubstat = cublasDestroy(this->CBhandle)) != CUBLAS_STATUS_SUCCESS) {
		cerr << "Cublas Destroy failed: " << ErrorHandler::cublasGetErrorString(cubstat) << endl;
	}
 }

/*--------------------------*/
/* Check if error occured   */
/*--------------------------*/
bool CudaAlgorithm::isValid() {
	return(valid);
}