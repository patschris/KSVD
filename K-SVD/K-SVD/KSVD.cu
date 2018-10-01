/*------------------------------*/
/*       KSVD-OPERATION         */
/*       IMPLEMENTATION         */
/*------------------------------*/
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include "GlobalDeclarations.cuh"
#include "Globals.cuh"
#include "FileManager.cuh"
#include "HostMemory.cuh"
#include "DeviceMemory.cuh"
#include "CudaAlgorithm.cuh"
#include "OMP.cuh"
#include "DictionaryUpdate.cuh"
#include "KSVD.cuh"
#include "ErrorHandler.cuh"
#include "ThrustOperators.cuh"
#include "Utilities.cuh"
#include "Algorithms.cuh"


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
inline void diagX_Mul(datatype*,datatype*,datatype*,int,int);
inline bool use_count(datatype*, datatype*, unsigned int, unsigned int);
inline void replace(datatype*, datatype*, datatype*, unsigned int, unsigned int);
inline void initialize(datatype*, datatype*, datatype*, datatype*, unsigned int, unsigned int);

/*---------------*/
/* Constructor   */
/*---------------*/
KSVD::KSVD(DeviceSpace* devptr, HostSpace* hostptr)
	: CudaAlgorithm(devptr, hostptr) {
}

/*--------------*/
/* Destructor   */
/*--------------*/
KSVD::~KSVD() {
}

/*-------------------------------*/
/*  Perform the K-SVD algorithm  */
/*-------------------------------*/
bool KSVD::ExecuteIterations(unsigned int iternum) {
	// D_temp = normcols(D);
	if (normcols() == false) {
		return false;
	}
	// Main Loop
	unsigned int* all_perms;
	for (unsigned int iter = 0; iter < iternum; iter++) {
		// G = D'*D  ==>  G = D_temp'*D_temp
		if (Algorithms::Multiplication::AT_times_B(this->CBhandle,
				this->deviceMemory.get(DeviceSpace::TEMP_ROWSD_BY_COLSD),
				Globals::rowsD, Globals::colsD,
				this->deviceMemory.get(DeviceSpace::TEMP_ROWSD_BY_COLSD),
				Globals::rowsD, Globals::colsD,
				this->deviceMemory.get(DeviceSpace::G)
			) == false) {

			return false;
		}
		
		//////////////////////////
		// Sparse coding stage! //
		//////////////////////////
		// Execute Batch-OMP stage now!
		// Equivalent: Gamma = sparsecode(data,D,XtX,G,thresh)
		if (this->BatchOMP() == false) {
			return false;
		}
		/*
			Note:
				Now array Gamma is not present but its elements are accessed
				using the following formula:

			 For a *non-zero* element at (i,j) we get its value at:
				(c + j*Tdata) [ (int) ( (selected_atoms + j*colsD) [ i ] - 1 ) ]
		*/

		/////////////////////////////////
		// Reset variables for the new //
		// iteration before Dictionary //
		// Update stage.               //
		/////////////////////////////////
		initialize(
			this->deviceMemory.get(DeviceSpace::UNUSED_SIGS),
			this->deviceMemory.get(DeviceSpace::REPLACED_ATOMS),
			this->deviceMemory.get(DeviceSpace::UNUSED_SIGS_COUNTER),
			this->deviceMemory.get(DeviceSpace::UNUSED_SIGS_BITMAP),
			Globals::colsX, Globals::colsD
		);

		//////////////////////////////
		// Dictionary update stage! //
		//////////////////////////////
		// Generate a random permutation of indices in the
		// range 1...colsD. Equivalent: p = randperm(dictsize); 
		all_perms = Algorithms::RandomPermutation::generateRandPerm(Globals::colsD);
		// Execute Dictionary Update stage now!
		if (this->updateDictionary(all_perms) == false) {
			return false;
		}

		////////////////////////
		// Compute error now! //
		////////////////////////
		if (this->errorComputation(iter + 1) == false) {
			return false;
		}
		
		////////////////////////
		//  Clear Dictionary  //
		////////////////////////
		if (this->clearDict() == false) {
			return false;
		}
	}
	return true;
}

/*-----------------------------------------*/
/* Clear the dictionary of unused atoms    */
/* or atoms having error above a threshold */
/* or atoms that few samples use them!     */
/*-----------------------------------------*/
bool KSVD::clearDict() {
	// Count how many elements in every row of Gamma
	// have absolute value above 1e-7
	if (use_count(
			this->deviceMemory.get(DeviceSpace::SELECTED_ATOMS),
			this->deviceMemory.get(DeviceSpace::TEMP_1_BY_COLSD),
			Globals::colsD, Globals::colsX
		)
		== false) {

		return false;
	}
	// Iteration to clear every atom that satisfies 
	// a certain condition implemented in maximum
	// function below. Matlab equivalent:
	//
	// for j = 1:dictsize
	// |	% compute G(:, j)
	// |	% replace atom
	// end
	//
	for (unsigned int j = 0; j < Globals::colsD; j++) {
		// Now we compute:
		//
		//      Gj = D'*D(:,j);
		//
		if (Algorithms::Multiplication::AT_times_x(this->CBhandle,
				this->deviceMemory.get(DeviceSpace::TEMP_ROWSD_BY_COLSD),
				Globals::rowsD,
				this->deviceMemory.get(DeviceSpace::TEMP_ROWSD_BY_COLSD) + j*Globals::rowsD,
				Globals::colsD,
				this->deviceMemory.get(DeviceSpace::G)
			)
			== false) {

			return false;
		}
		// Now we compute the maximum (square) value
		// of Gj. Operation performed:
		//
		//            (max(Gj.^2))
		//
		// We also apply the condition.
		if (Algorithms::Reduction::Maximum_squared(
				this->deviceMemory.get(DeviceSpace::G),
				Globals::colsD,
				this->deviceMemory.get(DeviceSpace::TEMP_1_BY_COLSD) + j,
				this->deviceMemory.get(DeviceSpace::REPLACED_ATOMS) + j,
				j	
			)
			== false) {

			return false;
		}		
		// We now find the signal with the maximum error.
		// Matlab equivalent:
		//
		//      [~,i] = max(err);
		//
		if (Algorithms::Reduction::Maximum(
				this->deviceMemory.get(DeviceSpace::ERR),
				Globals::colsX,
				this->deviceMemory.get(DeviceSpace::G)
			)
			== false) {

			return false;
		}
		// We should now calculate the norm of the
		// selected signal in our data.
		if (Algorithms::Reduction::euclidean_norm(
				this->deviceMemory.get(DeviceSpace::X_ARRAY),
				this->deviceMemory.get(DeviceSpace::G),
				Globals::rowsX, Globals::TPB_rowsX
			)
			== false) {

			return false;
		}
		// Finally replace atom.
		replace(
			this->deviceMemory.get(DeviceSpace::TEMP_ROWSD_BY_COLSD) + j * Globals::rowsD,
			this->deviceMemory.get(DeviceSpace::X_ARRAY),
			this->deviceMemory.get(DeviceSpace::G),
			Globals::rowsX, Globals::TPB_rowsX
		);
	}
	return true;
}

/*------------------------------------*/
/* Compute the residual error         */
/* using the formula:                 */
/*                                    */
/*  sqrt(sum[(X-D*Gamma)^2]/numel(X)) */
/*                                    */
/*------------------------------------*/
bool KSVD::errorComputation(unsigned int iter) {
	// Compute (D*Gamma) using CUBLAS 
	// for performance
	if (Algorithms::Multiplication::A_times_B(this->CBhandle,
			this->deviceMemory.get(DeviceSpace::TEMP_ROWSD_BY_COLSD), Globals::rowsD, Globals::colsD,
			this->deviceMemory.get(DeviceSpace::SELECTED_ATOMS), Globals::colsD, Globals::colsX,
			this->deviceMemory.get(DeviceSpace::TEMP_ROWSX_BY_COLSX))
		== false) {

		return false;
	}
	// Now reduce over the result and calculate
	// the error: 
	//         sum{ (X - X~)^2 }
	//
	if (Algorithms::Reduction::reduce_RMSE(
			this->deviceMemory.get(DeviceSpace::X_ARRAY),
			this->deviceMemory.get(DeviceSpace::TEMP_ROWSX_BY_COLSX),
			this->deviceMemory.get(DeviceSpace::ERR),
			Globals::rowsX , Globals::colsX, iter,
			Globals::TPB_rowsX,
			this->deviceMemory.get(DeviceSpace::UNUSED_SIGS_BITMAP))
		== false) {

		return false;
	}
	return true;
}

/*-------------------------*/
/*  Normalize columns of D */
/*-------------------------*/
bool KSVD::normcols() {
	// Calculating ==> 1./sqrt(sum(D.*D))
	if (Algorithms::Reduction::reduce2D_Sum_SQUAREDelements(
			this->deviceMemory.get(DeviceSpace::D_ARRAY),
			Globals::rowsD, Globals::colsD,
			this->deviceMemory.get(DeviceSpace::TEMP_1_BY_COLSD),
			Globals::TPB_rowsX
		) == false) {

		return false;
	}
	// D = D*spdiag( 1./sqrt(sum(D.*D)) ) ==> D = D*spdiag(TEMP_1_BY_COLSD)
	diagX_Mul(
		this->deviceMemory.get(DeviceSpace::D_ARRAY),
		this->deviceMemory.get(DeviceSpace::TEMP_ROWSD_BY_COLSD),
		this->deviceMemory.get(DeviceSpace::TEMP_1_BY_COLSD),
		Globals::rowsD * Globals::colsD, Globals::rowsD);

	return true;
}

/*-------------------------*/
/* Check if error occured  */
/*-------------------------*/
bool KSVD::isValid() {
	return(CudaAlgorithm::isValid());
}

///////////////////////////////////////////////////
////////////    HELPER FUNCTIONS   ////////////////
///////////////////////////////////////////////////

/************************************
*           DECLARATIONS            *
*	        ____________            *
************************************/
__global__
void diagMul(datatype*,datatype*, datatype*, int, int);
__global__
void use_count_kernel(datatype*, datatype*, unsigned int, unsigned int);
__global__
void replaceDj(datatype*, datatype*, datatype*, unsigned int);
__global__
void init(datatype*, datatype*, datatype*, datatype*, unsigned int, unsigned int);

/************************************
*            WRAPPERS               *
*	         ________               *
************************************/

/*---------------------------------*/
/* OPERATION:                      */
/*      out = in*diag(coeff)       */
/*---------------------------------*/
inline void diagX_Mul(datatype* in, datatype* out, datatype* coeff, int size, int rows) {
	dim3 block(1024);
	dim3 grid((size + block.x - 1) / block.x);
	diagMul <<< grid, block >>> (in, out, coeff, size, rows);
}

/*------------------------------*/
/* OPERATION:                   */
/*                              */
/*  usecount =                  */
/*   sum(abs(Gamma)>1e-7, 2);   */
/*                              */
/*------------------------------*/
inline bool use_count(
	datatype* gamma, datatype* counters,
	unsigned int colsD, unsigned int colsX) {

	// Initialize counters to zero
	if (cudaMemset(counters, 0, colsD * sizeof(datatype)) != cudaSuccess) {
		return false;
	}

	// Once again we assume colsD <= 32
	dim3 block(16, colsD); // => max. 512 threads per block
	dim3 grid((colsX + block.x - 1) / block.x);
	use_count_kernel << < grid, block >> > (gamma, counters, colsD, colsX);

	return true;
}

/*---------------------------------*/
/* OPERATION:                      */
/*   D(:,j) = X(:,unused_sigs(i))  */
/*     / norm(X(:,unused_sigs(i))) */
/*---------------------------------*/
inline void replace(
	datatype* Dj, datatype* X, datatype* G,
	unsigned int size, unsigned int recommended) {

	replaceDj << < 1, recommended >> > (Dj, X, G, size);
}

/*-----------------------------------*/
/* OPERATIONS:                       */
/*  replaced_atoms = zeros(dictsize) */
/*  unused_sigs = 1:size(data,2);    */
/*-----------------------------------*/
inline void initialize(
	datatype* un, datatype* rep, datatype* counter, datatype* bitmap,
	unsigned int N, unsigned int colsD) {

	dim3 block(1024);
	dim3 grid((N + block.x - 1) / block.x);
	init << <grid, block>> > (un, rep, counter, bitmap, N, colsD);
}

/************************************
*            KERNELS                *
*	         _______                *
************************************/

/*---------------------------*/
/* This kernel multiplies a  */
/* vector transformed into a */
/* diagonal matrix with some */
/* other matrix.             */
/*---------------------------*/
__global__
void diagMul(datatype* in, datatype* out, datatype* coeff, int size, int rows) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		int col = index / rows;
		out[index] = in[index] * coeff[col];
	}
}

/*-----------------------------------*/
/* This kernel uses an array of      */
/* counters to simultaneously count  */
/* elements of every row of input    */
/* that satisfy a certain condition. */
/*-----------------------------------*/
__global__ void use_count_kernel(
	datatype* gamma, datatype* counters,
	unsigned int colsD, unsigned int colsX) {

	// threadIdx.x = my column
	// threadIdx.y = my row
	unsigned int column = blockIdx.x * blockDim.x + threadIdx.x;
	if (column < colsX) {
		if (SQR((gamma + column*colsD)[threadIdx.y]) > 1e-7) {
			atomicAdd((unsigned int*)(counters + threadIdx.y), 1);
		}
	}
}

/*---------------------------------*/
/* This kernel performs the simple */
/* task of:                        */
/*  D(:,j) = X(:,i) / norm(X(:,i)) */
/*---------------------------------*/
__global__ void replaceDj(
	datatype* Dj, datatype* X, datatype* G,
	unsigned int size) {

	if (!(*G)) {
		// Our atom does not need replacement
		return;
	}
	if (threadIdx.x < size) {
		X += (unsigned int)(*(G + 1))*size;
		Dj[threadIdx.x] = X[threadIdx.x] / (*(G + 2));
	}
}

/*-----------------------------------*/
/* This kernel is the equivalent of: */
/*                                   */
/*  replaced_atoms = zeros(dictsize) */
/*  unused_sigs = 1:size(data,2)     */
/*-----------------------------------*/
__global__ void init(
	datatype* un, datatype* rep, datatype* counter, datatype* bitmap,
	unsigned int N, unsigned int N2) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < N) {
		un[index] = index;
		bitmap[index] = 1.0;
		if (index < N2) {
			rep[index] = 0.0;
		}
		if (index == 0) {
			*counter = 0.0;
		}
	}
}