/******************************/
/* CUDA IMPLEMENTATION - KSVD */
/******************************/
#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC  
#include <stdlib.h>  
#include <crtdbg.h>
#endif
#include <vector>
#include <iostream>
#include <iomanip>
#include <windows.h>
#include <thrust/device_vector.h>
#include <thrust/version.h>
#include "GlobalDeclarations.cuh"
#include "Globals.cuh"
#include "FileManager.cuh"
#include "HostMemory.cuh"
#include "DeviceMemory.cuh"
#include "Timer.cuh"
#include "Utilities.cuh"
#include "CudaAlgorithm.cuh"
#include "OMP.cuh"
#include "DictionaryUpdate.cuh"
#include "KSVD.cuh"
#include "Algorithms.cuh"


/*****************
*   NAMESPACES   *
*	__________   *
*****************/
using namespace		std;
using DeviceSpace	= DeviceMemory<datatype>;
using HostSpace		= HostMemory<datatype>;
using kSVD			= KSVD;

/***********************
*   FUNCTIONS - main   *
*	________________   *
************************/
int main(int argc, char** argv)
{
	#ifdef _DEBUG
		{
	#endif
	cout << "Initializing! Please wait..." << endl;{
	/*------------------*/
	/* Required objects */
	/*------------------*/
	// Files' manipulator
	FileManager fileManager;
	// Host memory space
	HostSpace hostMemory(fileManager);
	if (hostMemory.get(HostSpace::MAX_ELEMENTS) == NULL) {
		cerr << "Host Memory initialization error! Aborting..." << endl;
		return -1;
	}
	// Device memory space
	DeviceSpace deviceMemory(hostMemory);
	if (deviceMemory.get(DeviceSpace::MAX_ELEMENTS) == NULL) {
		cerr << "Device Memory initialization error! Aborting..." << endl;
		return -1;
	}
	// Timer for profiling
	Timer timer;
	if (!timer.isValid()) {
		cerr << "Timer initialization error! Aborting..." << endl;
		return -1;
	}
	// Class wrapper for the K-SVD execution
	kSVD KSVD(&deviceMemory, &hostMemory);
	if (!KSVD.isValid()) {
		cerr << "KSVD initialization error! Aborting..." << endl;
		return -1;
	}
	/*----------------*/
	/* Initialization */
	/*----------------*/
	// Set reduction buffer now
	// that we have device memory initialized
	if (Algorithms::Reduction::setReductionBuffer(
			deviceMemory.get(DeviceSpace::REDUCTION_BUFFER)) == false){

		cerr << "Reduction buffer initialization error! Aborting..." << endl;
		return -1;
	}
	// Set also the random generator buffer now...
	if (Algorithms::RandomPermutation::setGeneratorBuffer(
		hostMemory.get_RND_GNRT_buffer(), Globals::colsD ) == false) {

		cerr << "Random Generator buffer initialization error! Aborting..." << endl;
		return -1;
	}
	// Print mode of operation
	#ifdef DOUBLE
		cout << "Double precision mode." << endl;
	#else
		cout << "Single precision mode." << endl;
	#endif
	// Print sizes
	cout << "Dictionary size: " << Globals::rowsD << " x " << Globals::colsD << endl;
	cout << "Input data size: " << Globals::rowsX << " x " << Globals::colsX << endl;
	/*-----------*/
	/* Execution */
	/*-----------*/
	cout << "Running..." << endl << endl;
	timer.GPUtic();
	if (KSVD.ExecuteIterations(NUMBERofITERATIONS) == false) {
		cerr << "An error occured while executing KSVD!...Aborting" << endl;
		return -1;
	}
	float time = timer.GPUtoc();
	/*-----------*/
	/* Printing  */
	/*-----------*/
	cout << endl << "Final Execution Time =>\t\t" << setprecision(3) << std::fixed
		<< time << " ms (" << setprecision(2) << std::fixed
		<< time / 1000.0 << " sec)" << endl;
	cout << "Total Memory Used (program) =>\t" << setprecision(2) << std::fixed
		<< deviceMemory.getConsumed() / 1024.0 / 1024.0 << " MB" << endl;
	cout << "Total Available Memory =>\t" << setprecision(2) << std::fixed
		<< deviceMemory.getTotal() / 1024.0 / 1024.0 << " MB" << endl;
	cout << "Total Memory Used (OS) =>\t" << setprecision(2) << std::fixed
		<< deviceMemory.getUsed() / 1024.0 / 1024.0 << " MB" << endl;
	cout << "Total Free Memory =>\t\t" << setprecision(2) << std::fixed
		<< deviceMemory.getFree() / 1024.0 / 1024.0 << " MB" << endl;
	cout << "Memory Utilization =>\t\t" << setprecision(1) << std::fixed
		<< 100 * (deviceMemory.getConsumed() / deviceMemory.getFree()) << " %" << endl;
	/*------------*/
	/* Write back */
	/*------------*/
	thrust::host_vector<datatype> temp(Globals::colsD*Globals::colsX);
	thrust::copy(
		deviceMemory.getThrust(DeviceSpace::SELECTED_ATOMS).begin(),
		deviceMemory.getThrust(DeviceSpace::SELECTED_ATOMS).end(),
		temp.begin());
	datatype* ptr = thrust::raw_pointer_cast(&(temp[0]));
	if (fileManager.writeTemporary(ptr,	Globals::colsD, Globals::colsX, "CUDA_Gamma") == false) {
		cerr << "Cannot write file for temporary array! Aborting..." << endl;
		return -1;
	}
	/*---------*/
	/* Cleanup */
	/*---------*/
	cudaDeviceSynchronize();
	cout << "----------------------------------------------------" << endl;
	cudaError_t err;
	while( (err = cudaGetLastError()) != cudaSuccess){
		cerr << "Error: An error occured during the execution of the program: " <<
			cudaGetErrorString(err) << endl;
	}}
	cout << "# Exiting!" << endl;	
	#ifdef _DEBUG
		}
		_CrtDumpMemoryLeaks();
	#endif
	return 0;
}