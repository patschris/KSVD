/*------------------------------*/
/*      TIMING OPERATIONS       */
/*       IMPLEMENTATION         */
/*------------------------------*/
#include <iostream>
#include <windows.h>
#include "GlobalDeclarations.cuh"
#include "Timer.cuh"

using namespace std;


/*---------------*/
/* Constructor   */
/*---------------*/
Timer::Timer() {
	this->GPUstart = NULL;
	this->GPUstop = NULL;
	cudaError_t cuda_error;
	if ( (cuda_error = cudaEventCreate(&this->GPUstart)) != cudaSuccess) {
		cerr << "Cuda event create for GPUstart failed: " << cudaGetErrorString(cuda_error) << endl;
	}
	if ( (cuda_error = cudaEventCreate(&this->GPUstop)) != cudaSuccess) {
		cerr << "Cuda event create for GPUstop failed: " << cudaGetErrorString(cuda_error) << endl;
	}
}

/*--------------*/
/* Destructor   */
/*--------------*/
Timer::~Timer() {
	if (this->GPUstart) {
		cudaEventDestroy(this->GPUstart);
	}
	if (this->GPUstop) {
		cudaEventDestroy(this->GPUstop);
	}
}

/*---------------------------*/
/* Check if timer is valid   */
/*---------------------------*/
bool Timer::isValid() {
	return (this->GPUstart != NULL && this->GPUstop != NULL);
}

/*-------------------*/
/* Start CPU timer   */
/*-------------------*/
void Timer::tic() {
	QueryPerformanceFrequency(&this->frequency);
	QueryPerformanceCounter(&this->CPUstart);
}

/*--------------------*/
/* Stop CPU timer and */
/* return in sec!     */
/*--------------------*/
double Timer::toc() {
	QueryPerformanceCounter(&this->CPUstop);
	return ((double)(this->CPUstop.QuadPart - this->CPUstart.QuadPart) / this->frequency.QuadPart);
}

/*--------------------*/
/* Stop CPU timer and */
/* return in ms!      */
/*--------------------*/
double Timer::tocmilli() {
	QueryPerformanceCounter(&this->CPUstop);
	return ((double)(this->CPUstop.QuadPart - this->CPUstart.QuadPart) / this->frequency.QuadPart * 1000.0);
}

/*-------------------*/
/* Start GPU timer   */
/*-------------------*/
void Timer::GPUtic() {
	cudaEventRecord(this->GPUstart);
}

/*--------------------*/
/* Stop GPU timer and */
/* return in ms!      */
/*--------------------*/
float Timer::GPUtoc() {	
	cudaEventRecord(this->GPUstop);
	cudaEventSynchronize(this->GPUstop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, this->GPUstart, this->GPUstop);
	return milliseconds;
}