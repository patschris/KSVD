/*---------------------------------*/
/* k-MEANS DICTIONARY UPDATE STAGE */
/*         IMPLEMENTATION          */
/*---------------------------------*/
#include <vector>
#include <thrust/device_vector.h>
#include "GlobalDeclarations.cuh"
#include "Globals.cuh"
#include "FileManager.cuh"
#include "HostMemory.cuh"
#include "DeviceMemory.cuh"
#include "CudaAlgorithm.cuh"
#include "DictionaryUpdate.cuh"
#include "Utilities.cuh"
#include "Algorithms.cuh"
#include "ReductionBase.cuh"

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
inline void sprow(datatype*, datatype*, datatype*, datatype*, datatype*, int, int);
inline void special_case_update(datatype*, datatype*, datatype*, unsigned int, datatype*, datatype*, datatype*,
								datatype*, datatype*, unsigned int, unsigned int);

/*--------------*/
/* Constructor  */
/*--------------*/
DictionaryUpdate::DictionaryUpdate() : CudaAlgorithm(NULL, NULL) {
}

/*--------------------*/
/* Dictionary Update  */
/*--------------------*/
bool DictionaryUpdate::updateDictionary(unsigned int* p) {
	// We retrieve all the non-zero elements
	// along with their indices for every row
	// of the O.M.P. output Gamma. That way we
	// have eliminated the operation:
	//    [gamma_j, data_indices] = 
	//            sprow(Gamma, j)
	//
	// in each iteration.
	// Output stored in 'DtX' and 'alpha'.
	if (this->Sprow() == false) {
		return false;
	}
	/**/
	/* Analyze the data obtained by sprow in parallel */
	/**/
	// We pre-calculate the sum of squares for
	// each gamma_j because it remains constant
	// for every iteration. Thus we eliminate the
	// calculation of: 
	//          (gamma_j*gamma_j')
	//
	// for each row's loop. Output stored in 'G'.
	// Simultaneously, we pre-calculate the linear
	// combination of the columns of X using gamma_j
	// as the coefficients matrix. Essentially this
	// operation is:
	//			Y = X(:, COLS)*gamma_j'
	//
	// But each gamma_j remains constant
	// for every iteration. Thus we eliminate the
	// calculation of:
	//    collincomb(X,data_indices,gamma_j')
	//
	// for each row's loop.
	// Output stored in 'TEMP_COLSD_BY_ROWSX'.
	if (Algorithms::Reduction::reduce1D_Batched_Sum_TOGETHER_collincomb(
			this->deviceMemory.get(DeviceSpace::DtX),
			this->deviceMemory.get(DeviceSpace::G),
			this->deviceMemory.get(DeviceSpace::TEMP_1_BY_COLSD),
			Globals::colsD, Globals::colsX,
			this->deviceMemory.get(DeviceSpace::X_ARRAY),
			this->deviceMemory.get(DeviceSpace::alpha),
			this->deviceMemory.get(DeviceSpace::TEMP_COLSD_BY_ROWSX),
			Globals::rowsX)
		== false){

		return false;
	}

	////////////////////////////////////////
	// Dictionary Update - Main Iteration //
	////////////////////////////////////////
	unsigned int pJ;
	for (unsigned int j = 0; j < Globals::colsD; j++) {
		/*-- Optimize atom --*/
		pJ = p[j];
		// First we handle the special case where sprow()
		// returned zero for this row of Gamma i.e. no
		// non-zero elements were found
		if (specialCase(pJ) == false) {
			return false;
		}
		// Matrix-vector multiplication
		// Operation:
		//		(smallGamma*gamma_j')
		//
		if (Algorithms::Reduction::reduce2D_dot_products_modified(
				this->deviceMemory.get(DeviceSpace::SELECTED_ATOMS),
				this->deviceMemory.get(DeviceSpace::alpha) + pJ*Globals::colsX,
				this->deviceMemory.get(DeviceSpace::DtX) + pJ*Globals::colsX,
				this->deviceMemory.get(DeviceSpace::G) + Globals::colsD,
				this->deviceMemory.get(DeviceSpace::TEMP_1_BY_COLSD) + pJ,
				pJ,
				Globals::colsD, Globals::colsX)
			== false) {

			return false;
		}
		// Matrix-vector multiplication
		// Operation:
		//		D*(smallGamma*gamma_j')
		//
		if (Algorithms::Multiplication::A_times_x_SUBTRACT(
				this->deviceMemory.get(DeviceSpace::TEMP_ROWSD_BY_COLSD),
				Globals::rowsD,
				this->deviceMemory.get(DeviceSpace::G) + Globals::colsD ,
				Globals::colsD,
				this->deviceMemory.get(DeviceSpace::TEMP_COLSD_BY_ROWSX) + pJ * Globals::rowsD,
				this->deviceMemory.get(DeviceSpace::TEMP_ROWSD_BY_COLSD) + pJ * Globals::rowsD,
				this->deviceMemory.get(DeviceSpace::G) + pJ,
				this->deviceMemory.get(DeviceSpace::TEMP_1_BY_COLSD) + pJ,
				Globals::TPB_rowsX)
			== false) {

			return false;
		}
		// We calculate the linear combination
		// of the rows of X using 'atom' as the
		// coefficients matrix. Essentially this
		// operation is:
		//			Y = X'*A(ROWS,:)
		//
		// We also calculate the norm of the new
		// atom in parallel as well as execute the
		// multiplication: (atom'*D).
		if (Algorithms::Reduction::reduce2D_rowlincomb_plus_nrm2_plus_mul(
				this->deviceMemory.get(DeviceSpace::X_ARRAY),
				this->deviceMemory.get(DeviceSpace::TEMP_COLSD_BY_ROWSX) + pJ * Globals::rowsX,
				this->deviceMemory.get(DeviceSpace::TEMP_SINGLE_VALUE),
				this->deviceMemory.get(DeviceSpace::alpha) + pJ * Globals::colsX,
				Globals::rowsX, Globals::colsX,
				this->deviceMemory.get(DeviceSpace::TEMP_1_BY_COLSD) + pJ,
				Globals::TPB_rowsX,
				this->deviceMemory.get(DeviceSpace::c),
				this->deviceMemory.get(DeviceSpace::TEMP_ROWSD_BY_COLSD), 
				this->deviceMemory.get(DeviceSpace::G) + Globals::colsD,
				Globals::colsD
			)
			== false) {

			return false;
		}
		//
		// We finally calculate:
		//
		//		(atom'*D)*smallGamma
		//
		// using reduction and simultaneously
		// update D and Gamma with the new values.
		// These two operations can be performed
		// in parallel!		
		if (Algorithms::Multiplication::normal_x_times_A_modified(
				this->deviceMemory.get(DeviceSpace::G) + Globals::colsD,
				this->deviceMemory.get(DeviceSpace::SELECTED_ATOMS),
				this->deviceMemory.get(DeviceSpace::alpha) + pJ * Globals::colsX,
				Globals::colsD,
				this->deviceMemory.get(DeviceSpace::DtX) + pJ * Globals::colsX,
				this->deviceMemory.get(DeviceSpace::TEMP_SINGLE_VALUE),
				this->deviceMemory.get(DeviceSpace::G) + Globals::colsD + pJ,
				this->deviceMemory.get(DeviceSpace::c),
				pJ,
				this->deviceMemory.get(DeviceSpace::TEMP_1_BY_COLSD) + pJ,
				Globals::colsX,
				this->deviceMemory.get(DeviceSpace::TEMP_ROWSD_BY_COLSD) + pJ*Globals::rowsX,
				this->deviceMemory.get(DeviceSpace::TEMP_COLSD_BY_ROWSX) + pJ*Globals::rowsX,
				Globals::rowsX
			)
			== false) {

			return false;
		}
	}
	////////////////////////////////////////
	//       End Of Main Iteration        //
	////////////////////////////////////////

	return true;
}

/*-------------------------------------*/
/* SPROW: Store all non-zero elements, */
/* then store their indices ( source:  */
/* array Gamma [ i.e. 'c' ] )          */
/*-------------------------------------*/
bool DictionaryUpdate::Sprow() {
	// Initialize counters to zero
	if (cudaMemset(
			this->deviceMemory.get(DeviceSpace::TEMP_1_BY_COLSD),
			0,
			Globals::colsD * sizeof(datatype)
		) != cudaSuccess) {

		return false;
	}
	// Execute sprow in a single pass
	sprow(
		this->deviceMemory.get(DeviceSpace::c),
		this->deviceMemory.get(DeviceSpace::SELECTED_ATOMS),
		this->deviceMemory.get(DeviceSpace::TEMP_1_BY_COLSD),
		this->deviceMemory.get(DeviceSpace::DtX),
		this->deviceMemory.get(DeviceSpace::alpha),
		Globals::colsD, Globals::colsX
	);
	return true;
}

/*----------------------------------------*/
/* In this function we handle the special */
/* case when a row of Gamma is not used   */
/* by any atom and thus sprow() returned  */
/* an empty set.                          */
/*----------------------------------------*/
bool DictionaryUpdate::specialCase(unsigned int pJ) {
	// Compute (D*Gamma) using CUBLAS 
	// for performance
	if (Algorithms::Multiplication::A_times_B(this->CBhandle,
			this->deviceMemory.get(DeviceSpace::TEMP_ROWSD_BY_COLSD), Globals::rowsD, Globals::colsD,
			this->deviceMemory.get(DeviceSpace::SELECTED_ATOMS), Globals::colsD, Globals::colsX,
			this->deviceMemory.get(DeviceSpace::TEMP_ROWSX_BY_COLSX)
		)
		== false) {

		return false;
	}
	// Now reduce over the result and calculate
	// the error: 
	//        E = sum{ (X - X~)^2 }
	//
	// Then we calculate:
	//         [d,i] = max(E);
	//
	// Output 'i' stored in *TEMP_ROWSX_BY_COLSX.
	if (Algorithms::Reduction::reduce_ERROR_for_special_case(
			this->deviceMemory.get(DeviceSpace::X_ARRAY),
			this->deviceMemory.get(DeviceSpace::TEMP_ROWSX_BY_COLSX),
			this->deviceMemory.get(DeviceSpace::ERR),
			Globals::rowsX, Globals::colsX,
			this->deviceMemory.get(DeviceSpace::TEMP_1_BY_COLSD) + pJ,
			this->deviceMemory.get(DeviceSpace::UNUSED_SIGS),
			this->deviceMemory.get(DeviceSpace::UNUSED_SIGS_COUNTER),
			Globals::TPB_rowsX
		)
		== false) {

		return false;
	}
	// We should now calculate the norm of the
	// selected signal in our data.
	if (Algorithms::Reduction::SCase_EU_norm(
			this->deviceMemory.get(DeviceSpace::X_ARRAY),
			this->deviceMemory.get(DeviceSpace::TEMP_ROWSX_BY_COLSX),
			Globals::rowsX, Globals::TPB_rowsX,
			this->deviceMemory.get(DeviceSpace::TEMP_1_BY_COLSD) + pJ,
			this->deviceMemory.get(DeviceSpace::UNUSED_SIGS),
			Globals::colsX
		) 
		== false) {

		return false;
	}
	// We finally update the dictionary with the new
	// atom divided by its norm
	special_case_update(
		this->deviceMemory.get(DeviceSpace::X_ARRAY),
		this->deviceMemory.get(DeviceSpace::TEMP_ROWSD_BY_COLSD) + pJ * Globals::rowsD,
		this->deviceMemory.get(DeviceSpace::TEMP_ROWSX_BY_COLSX),
		Globals::rowsX,
		this->deviceMemory.get(DeviceSpace::TEMP_1_BY_COLSD) + pJ,
		this->deviceMemory.get(DeviceSpace::UNUSED_SIGS),
		this->deviceMemory.get(DeviceSpace::UNUSED_SIGS_COUNTER),
		this->deviceMemory.get(DeviceSpace::UNUSED_SIGS_BITMAP),
		this->deviceMemory.get(DeviceSpace::REPLACED_ATOMS) + pJ,
		Globals::colsX,
		Globals::TPB_rowsX
	);
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
void sprow_kernel(datatype*, datatype*, datatype*, datatype*, datatype*, int, int);
__global__
void SCupdate(datatype*, datatype*, datatype*, unsigned int, datatype*, datatype*, datatype*, datatype*,
				datatype*, unsigned int, unsigned int);

/************************************
*            WRAPPERS               *
*	         ________               *
************************************/

/*------------------------------*/
/* OPERATION:                   */
/*  [gamma_j, data_indices] =   */
/*       sprow(Gamma, j);       */
/*                              */
/* { for every j in 1...colsD } */
/*------------------------------*/
inline void sprow(
	datatype* c, datatype* selected_atoms, datatype* counters,
	datatype* out_values, datatype* out_indices,
	int colsD, int colsX) {

	// Once again we assume colsD <= 32 in 
	// order to fit in a single warp
	dim3 block(16,32); // = 512 threads per block
	dim3 grid((colsX + block.x - 1) / block.x);
	sprow_kernel << < grid, block >> > ( 
		c, selected_atoms, counters,
		out_values, out_indices,
		colsD, colsX);
}

/*------------------------------------*/
/* OPERATIONS:                        */
/*	 atom = X(:,unused_sigs(perm(i))) */
/* and                                */
/*     atom = atom./norm(atom)        */
/* and                                */
/*		 D(:,p(j)) = atom             */
/*                                    */
/*------------------------------------*/
inline void special_case_update(
	datatype* X, datatype* Dj, datatype* temp, unsigned int N,
	datatype* counters, datatype* unused, datatype* UScounter,
	datatype* bitmap, datatype* replaced, unsigned int colsX,
	unsigned int recommended) {

	// We assume length i.e. rowsD <= 1024
	SCupdate << <1, recommended >> > (X, Dj, temp, N, counters,
		unused, UScounter, bitmap, replaced, colsX, MIN(colsX, MAX_SIGNALS));
}

/************************************
*            KERNELS                *
*	         _______                *
************************************/

/*-----------------------------------*/
/* This kernel uses an array of      */
/* counters to simultaneously count, */
/* index and store the non-zero      */
/* elements of input, multiplexed by */
/* the selected atoms array, to two  */
/* output arrays.                    */
/*-----------------------------------*/
__global__ void sprow_kernel(
	datatype* c, datatype* selected_atoms, datatype* counters, 
	datatype* out_values, datatype* out_indices,
	int colsD, int colsX) {

	// threadIdx.x = my column
	// threadIdx.y = my row
	unsigned int column = blockIdx.x * blockDim.x + threadIdx.x, ind, pos;
	if (column < colsX && threadIdx.y < colsD) {
		ind = (selected_atoms + column*colsD)[threadIdx.y];
		if (ind) {
			pos = atomicAdd((int*)(counters + threadIdx.y), 1);
			*(out_values + threadIdx.y*colsX + pos) = (c + column*Tdata)[ind - 1];
			*(out_indices + threadIdx.y*colsX + pos) = column;
			(selected_atoms + column*colsD)[threadIdx.y] = (c + column*Tdata)[ind - 1];
		}
	}
}

/*------------------------------------*/
/* OPERATIONS:                        */
/*	 atom = X(:,unused_sigs(perm(i))) */
/* and                                */
/*     atom = atom./norm(atom)        */
/* and                                */
/*		 D(:,p(j)) = atom             */
/*                                    */
/*------------------------------------*/
__global__ void SCupdate(
	datatype* X, datatype* Dj, datatype* temp, unsigned int N,
	datatype* counters, datatype* unused, datatype* UScounter,
	datatype* bitmap, datatype* replacedJ, unsigned int colsX,
	unsigned int dim) {
	 
	unsigned int offset, counter;
	if (*((int*)counters) == 0) {
		if (threadIdx.x < N) {
			offset = (unsigned int)unused[myGenerator((unsigned int)(*temp), dim)];
			X += offset*N;
			Dj[threadIdx.x] = X[threadIdx.x] / (*(temp + 1));
			if (threadIdx.x == 0) {
				counter = (*UScounter);
				unused[myGenerator((unsigned int)(*temp), dim)] = unused[colsX - counter - 1];
				(*UScounter) = counter + 1;
				*replacedJ = 1;
				bitmap[offset] = 0;
			}
		}
	}
}