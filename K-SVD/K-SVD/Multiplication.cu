/*-------------------------------*/
/*    ALGORITHMIC TOOLS CLASS    */
/*  - MULTIPLICATION SUBCLASS -  */
/*        IMPLEMENTATION         */
/*-------------------------------*/
#include <iostream>
#include "GlobalDeclarations.cuh"
#include "Algorithms.cuh"
#include "ErrorHandler.cuh"
#include "ReductionBase.cuh"

using namespace std;


/*********************************
*         DECLARATIONS           *
*	      ____________           *
*********************************/
__global__
void device_A_times_x_modified(datatype*, datatype*, datatype*, unsigned int, unsigned int,	datatype*, 
								datatype*, datatype*);
__global__
void device_normal_x_times_A_modified(datatype*, datatype*, datatype*, unsigned int, datatype*, datatype*,
										datatype*, datatype*, unsigned int,	datatype*, datatype*, datatype*,
										unsigned int, unsigned int);


/*************************************
*         MEMBER FUNCTIONS           *
*	      ________________           *
*************************************/
/*-----------------------------------*/
/* Matrix-Matrix Multiplication Set  */
/* __________________________________*/
/* Library: Cublas                   */
/*		{ using cublas<T>GEMM }      */
/* Operation performed format        */
/*		==> C = A'*B or AT*B         */
/*-----------------------------------*/
bool Algorithms::Multiplication::AT_times_B(cublasHandle_t handle,
	datatype* A, int rowsA, int colsA,
	datatype* B, int rowsB, int colsB,
	datatype* C) {

	cublasStatus_t status;
	const datatype alpha = 1.0;
	const datatype beta = 0.0;
	if ((
		#ifdef DOUBLE
			status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
				colsA, colsB, rowsB, &alpha, A, rowsA, B, rowsB, &beta, C, colsA)
		#else
			status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
				colsA, colsB, rowsB, &alpha, A, rowsA, B, rowsB, &beta, C, colsA)
		#endif
	) != cudaSuccess) {

		cerr << "Cublas multiplication (AT*B) failed: " << ErrorHandler::cublasGetErrorString(status) << endl;
		return false;
	}
	
	return true;
}

/*-----------------------------------*/
/* Matrix-Matrix Multiplication Set  */
/* __________________________________*/
/* Library: Cublas                   */
/*		{ using cublas<T>GEMM }      */
/* Operation performed format        */
/*		==>   C = A*B                */
/*-----------------------------------*/
bool Algorithms::Multiplication::A_times_B(cublasHandle_t handle,
	datatype* A, int rowsA, int colsA,
	datatype* B, int rowsB, int colsB,
	datatype* C) {

	cublasStatus_t status;
	const datatype alpha = 1.0;
	const datatype beta = 0.0;
	if ((
		#ifdef DOUBLE
			status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
				rowsA, colsB, rowsB, &alpha, A, rowsA, B, rowsB, &beta, C, rowsA)
		#else
			status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
				rowsA, colsB, rowsB, &alpha, A, rowsA, B, rowsB, &beta, C, rowsA)
		#endif
	) != cudaSuccess) {

		cerr << "Cublas multiplication (A*B) failed: " << ErrorHandler::cublasGetErrorString(status) << endl;
		return false;
	}

	return true;
}

/*-----------------------------------*/
/* Matrix-Vector Multiplication Set  */
/* __________________________________*/
/* Operation performed format        */
/*		==>   C -= A*x               */
/*-----------------------------------*/
bool Algorithms::Multiplication::A_times_x_SUBTRACT(
	datatype* A, int rowsA,
	datatype* x, int length,
	datatype* C,
	datatype* D, datatype* gamma_J, datatype* counter,
	unsigned int recommended) {

	device_A_times_x_modified << <1,recommended >> > (
		A, x, C, rowsA, length, D, gamma_J, counter);

	return true;
}

/*-----------------------------------*/
/* Matrix-Vector Multiplication Set  */
/* __________________________________*/
/* Operation performed format        */
/*		==>   C = AT*x               */
/*-----------------------------------*/
bool Algorithms::Multiplication::AT_times_x(cublasHandle_t handle,
	datatype* A, int rowsA,
	datatype* x, int length,
	datatype* C) {

	cublasStatus_t status;
	const datatype alpha = 1.0;
	const datatype beta = 0.0;
	if ((
		#ifdef DOUBLE
			status = cublasDgemv(handle, CUBLAS_OP_T,
					rowsA, length, &alpha, A, rowsA, x, 1, &beta, C, 1)
		#else
			status = cublasSgemv(handle, CUBLAS_OP_T,
					rowsA, length, &alpha, A, rowsA, x, 1, &beta, C, 1)
		#endif
	) != cudaSuccess) {

		cerr << "Cublas multiplication (AT*x) failed: " << ErrorHandler::cublasGetErrorString(status) << endl;
		return false;
	}

	return true;
}

/*-----------------------------------*/
/* Matrix-Vector Multiplication Set  */
/* __________________________________*/
/* Operation performed format        */
/*		==>   C = x*A               */
/*-----------------------------------*/
bool Algorithms::Multiplication::normal_x_times_A_modified(
	datatype* A, datatype* Gamma, datatype* columns, unsigned int colsD,
	datatype* gammaJ, datatype* rowlincomb, datatype* constant, datatype* norm, unsigned int pJ,
	datatype* counter, unsigned int colsX, datatype* D, datatype* in, unsigned int N) {

	// block.x => No. of cols direction
	dim3 block(512);
	dim3 grid((colsX + N + block.x - 1) / block.x);
	device_normal_x_times_A_modified << <grid, block >> > (
		A, Gamma, columns, colsD, gammaJ, rowlincomb,
		constant, norm, pJ, counter, D, in, N, colsX);
	return true;
}


/*=======================================================================*/
/*========================        KERNELS       =========================*/
/*=======================================================================*/


/*--------------------------------*/
/* Operation: C = C - A*x         */
/*--------------------------------*/
__global__ void device_A_times_x_modified(
	datatype* A, datatype* x, datatype* out,
	unsigned int rowsA, unsigned int N, datatype* D,
	datatype* gammaJ, datatype* counters) {

	datatype sum;
	if (*((int*)counters) != 0) {
		if (threadIdx.x < rowsA) {
			sum = 0;
			for (int i = 0; i < N; i++) {
				sum += A[i*rowsA + threadIdx.x] * x[i];
			}
			out[threadIdx.x] = out[threadIdx.x] - sum +
				D[threadIdx.x] * (*gammaJ);
		}
	}
}

/*--------------------*/
/* Operation: C = x*A */
/*--------------------*/
__global__ void device_normal_x_times_A_modified(
	datatype* A, datatype* Gamma, datatype* columns, unsigned int colsD,
	datatype* gammaJ, datatype* rowlincomb, datatype* constant,
	datatype* norm, unsigned int pJ, datatype* counter,
	datatype* D, datatype* in, unsigned int N, unsigned int colsX) {

	unsigned int No_cols;
	if ((No_cols = *((int*)counter)) != 0) {
		unsigned int myCol = blockIdx.x*blockDim.x + threadIdx.x;
		if (myCol < No_cols) {
			datatype sum = 0;
			for (int i = 0; i < colsD; i++) {
				sum += Gamma[(int)columns[myCol] * colsD + i] * A[i];
			}
			Gamma[(int)columns[myCol] * colsD + pJ] = (rowlincomb[myCol] - sum + (*constant)*gammaJ[myCol]) / (*norm);
		}
		else if (myCol >= colsX) {
			// Now update the Dictionary!
			// OPERATIONS:               
			//		D(:,p(j)) = atom     
			//
			myCol -= colsX;
			if (myCol < N) {
				D[myCol] = in[myCol] / (*norm);
			}
		}
	}
}