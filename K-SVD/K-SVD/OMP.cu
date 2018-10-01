/*------------------------------*/
/*   BATCH-OMP CUDA ALGORITHM   */
/*       IMPLEMENTATION         */
/*------------------------------*/
#include <vector>
#include <thrust/device_vector.h>
#include "GlobalDeclarations.cuh"
#include "Globals.cuh"
#include "FileManager.cuh"
#include "HostMemory.cuh"
#include "DeviceMemory.cuh"
#include "CudaAlgorithm.cuh"
#include "OMP.cuh"
#include "Algorithms.cuh"
#include "Utilities.cuh"

using namespace std;


/*****************
*   NAMESPACES   *
*	__________   *
*****************/
using DeviceSpace	= DeviceMemory<datatype>;
using HostSpace		= HostMemory<datatype>;

/*******************
*   DECLARATIONS   *
*	____________   *
*******************/
inline void parallelOMP(datatype*, datatype*, datatype*, datatype*, int, int);

/*---------------*/
/*  Constructor  */
/*---------------*/
OMP::OMP() : CudaAlgorithm(NULL,NULL){
}

/*---------------*/
/* OMP function  */
/*---------------*/
bool OMP::BatchOMP() {
	// DtX = D'*data  ==>  DtX = D'*X  ==>  DtX = D_temp'*X_ARRAY
	if (Algorithms::Multiplication::AT_times_B(this->CBhandle,
			this->deviceMemory.get(DeviceSpace::TEMP_ROWSD_BY_COLSD),
			Globals::rowsD, Globals::colsD,
			this->deviceMemory.get(DeviceSpace::X_ARRAY),
			Globals::rowsX, Globals::colsX,
			this->deviceMemory.get(DeviceSpace::DtX)
		) == false) {

		return false;
	}
	// Start Batch-OMP in parallel fashion now!
	if (this->parallel_signals_operations() == false) {
		return false;
	}
	return true;
}

/*--------------------------------*/
/* Implementation of the parallel */
/* iterations of the OMP alg.     */
/*--------------------------------*/
bool OMP::parallel_signals_operations() {
	parallelOMP(
		this->deviceMemory.get(DeviceSpace::DtX),
		this->deviceMemory.get(DeviceSpace::SELECTED_ATOMS),
		this->deviceMemory.get(DeviceSpace::G),
		this->deviceMemory.get(DeviceSpace::c),
		Globals::colsD, Globals::colsX);
	return true;
}

///////////////////////////////////////////////////
////////////    HELPER FUNCTIONS   ////////////////
///////////////////////////////////////////////////

/************************************
*           DECLARATIONS            *
*	        ____________            *
************************************/
__global__
void parOMP(datatype*, datatype*, datatype*, datatype*, int, int);

/************************************
*            WRAPPERS               *
*	         ________               *
************************************/

/*------------------------------------*/
/* OPERATION:                         */
/* ----------                         */
/* for (signum=0; signum<L; ++signum) */
/* {                                  */
/*	 ...                              */
/* }                                  */
/* *in parallel threads*              */
/*------------------------------------*/
inline void parallelOMP(
		datatype* DtX, datatype* selected_atoms,
		datatype* G, datatype* c,
		int colsD, int colsX) {

	// We set 512 threads per block as
	// it was experimentally found that
	// there are not enough resources on the
	// GPU to utilize this kernel.
	parOMP << < (colsX + 512 - 1) / 512, 512 >> > (
		DtX, selected_atoms, G,
		c, colsD, colsX);
	
}

/************************************
*            KERNELS                *
*	         _______                *
************************************/

/*---------------------------------*/
/* This function computes the back */
/* substitution of a linear system */
/* L'*x = b where L is product of  */
/* a Cholesky factorization.       */
/*---------------------------------*/
__inline__ __device__
void backsubst_Lt(datatype* L, datatype* b, datatype* x, int n, int k) {
	unsigned int i, j;
	datatype rhs;

	for (i = k; i >= 1; --i) {
		rhs = b[i - 1];
		for (j = i; j<k; ++j) {
			rhs -= L[(i - 1)*n + j] * x[j];
		}
		x[i - 1] = rhs / L[(i - 1)*n + i - 1];
	}
}

/*---------------------------------*/
/* This function computes the back */
/* substitution of a linear system */
/* L*x = b where L is product of a */
/* Cholesky factorization. It also */
/* uses indexed input.             */
/*---------------------------------*/
__inline__ __device__
void backsubt_indexed_L(datatype* L, datatype* b, datatype* x, unsigned int* ind, int n, int k) {
	unsigned int i, j;
	datatype rhs;

	for (i = 0; i<k; ++i) {
		rhs = b[ind[i]];
		for (j = 0; j<i; ++j) {
			rhs -= L[j*n + i] * x[j];
		}
		x[i] = rhs / L[i*n + i];
	}
}

/*---------------------------------*/
/* This function computes the back */
/* substitution of a linear system */
/* L*x = b where L is product of a */
/* Cholesky factorization. It also */
/* uses indexed input and in-place */
/* output.                         */
/*---------------------------------*/
__inline__ __device__
void backsubt_indexed_append_L(datatype* L, datatype* b, unsigned int* ind, int n, int k, datatype* temp_storage) {
	unsigned int i, j;
	datatype rhs;

	for (i = 0; i<k; ++i) {
		rhs = b[ind[i]];
		for (j = 0; j<i; ++j) {
			rhs -= L[j*n + i] * temp_storage[j];
		}
		temp_storage[i] = rhs / L[i*n + i];
		L[i*n + k] = temp_storage[i];
	}
}

/*---------------------------------*/
/* Helper function to zero-out the */
/* specified array.                */
/*---------------------------------*/
__inline__ __device__
void zeroOut(datatype* array, int colsD) {
	for (int k = 0; k < colsD; k++) {
		array[k] = 0.0;
	}
}

/*-------------------------------*/
/* Helper function to return the */
/* index of the element of the   */
/* maximum absolute value inside */
/* the specified array.          */
/*-------------------------------*/
__inline__ __device__
unsigned int maxabs(datatype* array, int size) {
	datatype max = SQR(*array), value;
	unsigned int index = 0;
	for (int k = 1; k < size; k++) {
		value = SQR(array[k]);
		if (value > max) {
			max = value;
			index = k;
		}
	}
	return index;
}

/*-------------------------------*/
/* Helper function to return the */
/* sum of the square values of   */
/* the elements in the specified */
/* array.                        */
/*-------------------------------*/
__inline__ __device__
datatype sumSquareValues(datatype* array, int size, int stride) {
	datatype sum = 0;
	for (int k = 0; k < size; k++) {
		sum += SQR(array[k*stride + size]);
	}
	return (1 - sum);
}

/*---------------------------*/
/* Helper function for:      */
/*    alpha = D'*residual    */
/*---------------------------*/
__inline__ __device__
void addMul(
	datatype* A, int rowsA, int colsA, datatype* x,
	unsigned int* ind, datatype* alpha, datatype* alpha_original) {

	datatype sum;
	for (int i = 0; i < rowsA; i++) {
		sum = 0;
		for (int j = 0; j < colsA; j++) {
			sum += A[ind[j] * rowsA + i] * x[j];
		}
		alpha[i] = alpha_original[i] - sum;
	}
}

/*-----------------------------------*/
/* This kernel executes the parallel */
/* loops for calculating the sparse  */
/* representation of each signal.    */
/*-----------------------------------*/
__global__
void parOMP(
		datatype* DtX, datatype* selected_atoms, datatype* G, datatype* c,
		int colsD, int colsX){
	// >
	// Start Batch-OMP...
	//....................................
	//  Parallel operations from now on!
	//....................................
	// Index of the current signal
	unsigned int signum = blockIdx.x * blockDim.x + threadIdx.x;
	if (signum < colsX) {
		// Local Batch-O.M.P. matrices 
		datatype cholBuffer[Tdata];
		unsigned int ind[Tdata];
		datatype Lchol[Tdata*Tdata];
		datatype alpha[32];		// colsD <= 32 as in the rest of the program
		// Increasing factors
		const unsigned int inc_colsD = signum*colsD;
		const unsigned int inc_Tdata = signum*Tdata;
		// We reset the 'selected atoms' array to zero because 
		// of remnants from previous iterations of K-SVD.
		zeroOut(selected_atoms + inc_colsD, colsD);
		// Stack variables
		unsigned int i = 0, pos;
		datatype sum;

		/* >>> O.M.P. - Main Iteration <<< */
		while (i < Tdata) {
			
			if (i == 0) {
				/* Avoid copy and use DtX directly*/

				// Index of next atom
				// OPERATION :=> pos = maxabs(alpha, m)
				pos = maxabs(DtX + inc_colsD, colsD);
				
				// Stop criterion: selected same atom twice, or inner product too small
				if ((selected_atoms + inc_colsD)[pos] || SQR((DtX + inc_colsD)[pos]) < 1e-14) {
					break;
				}
			}
			else {
				/* Use alpha array for any subsequent iteration */

				// Index of next atom
				// OPERATION :=> pos = maxabs(alpha, m)
				pos = maxabs(alpha, colsD);
				
				// Stop criterion: selected same atom twice, or inner product too small
				if ((selected_atoms + inc_colsD)[pos] || SQR(alpha[pos]) < 1e-14) {
					break;
				}
			}

			// Mark selected atom
			ind[i] = pos;
			///////////////// MODIFIED /////////////////
			/* (see note at the end of this function) */
			(selected_atoms + inc_colsD)[pos] = i + 1;

			// Cholesky update
			if (i == 0) {
				*Lchol = 1;
			}
			else {
				/* incremental Cholesky decomposition: compute next row of Lchol */
				backsubt_indexed_append_L(
					Lchol,
					G + pos*colsD,
					ind,
					Tdata, i,
					cholBuffer);

				/* compute Lchol(i,i) */
				sum = sumSquareValues(Lchol, i, Tdata);

				if (sum <= 1e-14) {
					break;
				}
	
				Lchol[i*Tdata + i] = sqrt(sum);
			}		

			i++;

			/* perform orthogonal projection and compute sparse coefficients */
			
			// The two following functions together 
			// compute the Cholesky factorization
			// of a given array and solve the 
			// corresponding set of linear systems
			// by executing types of back substitution.              
			backsubt_indexed_L(
				Lchol,
				DtX + inc_colsD,
				cholBuffer,
				ind,
				Tdata, i);
			backsubst_Lt(
				Lchol,
				cholBuffer,
				c + inc_Tdata,
				Tdata, i);


			/* update alpha = D'*residual */

			// Here we make the assumption that 'colsD' cannot be more than 32 atoms
			// (i.e. signal sources ). Also 'i' cannot be more than colsD because the
			// loop would have 'breaked' in line 316
			addMul(
				G, colsD, i,
				c + inc_Tdata,
				ind,
				alpha,
				DtX + inc_colsD
			);
		}
		/* >>> O.M.P. - Post Iteration Finalization <<< */
		/*
		  Note: At this point the CPU equivalent of OMP produces a
				sparse array Gamma using array 'c' and indices array
				'ind'. However the next step ( i.e the first step of
				Dictionary update phase) commands that only non-zero
				elements of Gamma can be used and their respective
				indices. That can be easily implemented by launching
				a new grid to recover the data and as such the need
				for a temporary storage is no longer neccessary. For
				this to happen, however, a slight modification has to
				be made so that the 'selected atoms' array informs us 
				not only whether a non-zero element is present at a 
				specific position but also its index in the output
				array ('c').
		*/
		/*
		  Modification summary: 
				ORIGINAL selected_atoms ARRAY => stores 0 or 1 to indicate 
												 presence/absence of element
				MODIFIED VERSION => stores index of element in range {1...Tdata}
									or 0 to indicate zero-element

		  A non-zero element can later be found through row and column indices
		  of Gamma: 
				For a *non-zero* element at Full_Gamma[i][j] we get its value at:
					(c + j*Tdata) [ (int) ( (selected_atoms + j*colsD) [ i ] - 1 ) ]
		*/
	}
	//....................................
	//      END OF PARALLEL SECTION
	//....................................
}