/*----------------------------*/
/*   ALGORITHMIC TOOLS CLASS  */
/*   - REDUCTIONS SUBCLASS -  */
/*       IMPLEMENTATION       */
/*----------------------------*/
#include <iostream>
#include "GlobalDeclarations.cuh"
#include "Algorithms.cuh"
#include "ReductionKernels.cuh"

using namespace std;


/*********************************
*         DECLARATIONS           *
*	      ____________           *
*********************************/
extern __device__
datatype* experimental_array;

/*************************************
*         MEMBER FUNCTIONS           *
*	      ________________           *
*************************************/
/*--------------------------------------------*/
/* Reduction in 1D layout (i.e. vector )      */
/* of __M__A__X__I__M__U__M__ squared value.  */
/*--------------------------------------------*/
__host__
bool Algorithms::Reduction::Maximum_squared(
	datatype* in, unsigned int colsD, datatype* usecount,
	datatype* replaced, unsigned int j) {

	// We again assume colsD <= 32 to utilize a single warp
	device_max << <1, 32 >> > (in, colsD, usecount, replaced, j);
	return true;
}

/*-------------------------------------------*/
/* Reduction in 1D layout (i.e. vector) of   */
/* __M__A__X__I__M__U__M__ value of its      */
/* elements as well as their indices.        */
/*-------------------------------------------*/
__host__
bool Algorithms::Reduction::Maximum(datatype* in, int size, datatype* out) {
	dim3 block(1024);
	dim3 grid((size + block.x - 1) / block.x);
	device_max_err_STAGE1 << < grid, block >> > (in, size, out);
	int N;
	while (grid.x > 1) {
		N = grid.x;
		grid.x = (N + block.x - 1) / block.x;
		device_max_err_STAGE2 << < grid, block >> > (in, N, out);
	}
	return true;
}

/*-----------------------------------------------*/
/* Reduction in 2D layout (i.e. matrix columns)  */
/* of __S__U__M__ of squared values of elements. */
/* Result is then square rooted and inverted.    */
/*-----------------------------------------------*/
__host__
bool Algorithms::Reduction::reduce2D_Sum_SQUAREDelements(
	datatype* in, int rows, int cols, datatype* out,
	unsigned int recommended) {

	dim3 block(recommended);
	dim3 grid(cols);
	deviceReduceKernel_2D_square_values << <grid, block >> > (in, rows, out);
	return true;
}

/*--------------------------------------*/
/* Reduction in 1D layout (i.e. vector) */
/* of __S__U__M__ of squared values of  */
/* elements (more than 1024).           */
/*--------------------------------------*/
__host__
bool Algorithms::Reduction::reduce1D_Batched_Sum_TOGETHER_collincomb(
	datatype* in, datatype* out, datatype* counters, int colsD, int colsX,
	datatype *A, datatype* columns, datatype* out2,	unsigned int rowsX) {

	dim3 block(1024);
	dim3 grid((colsX + block.x - 1) / block.x, colsD);
	multi_reduction_squared_sum_STAGE1 << <grid, block >> > (in, out, counters, colsX);
	grid.x = colsD;
	grid.y = 2;
	multi_reduction_squared_sum_STAGE2_together_collincomb << <grid, block >> >(
		in, out, counters, colsX,
		A, columns, out2, rowsX);
	return true;
}

/*------------------------------------------------*/
/* Reduction in 2D layout (i.e. matrix columns)   */
/* of linear combination of input matrix and      */
/* given columns (filtered by the specified rows) */
/*------------------------------------------------*/
__host__
bool Algorithms::Reduction::reduce2D_rowlincomb_plus_nrm2_plus_mul(
	datatype *A, datatype* x, datatype* out, datatype* cols,
	unsigned int rowsX, unsigned int colsX,	datatype* counters,
	unsigned int recommended, datatype* out2, datatype* D,
	datatype* out3,	unsigned int colsD) {

	rowlincomb << <colsX + colsD + 1, recommended >> > (
		A, x, out, cols, rowsX, counters,
		out2, D, out3, colsX);
	return true;
}

/*-------------------------------------------*/
/* Reduction in 2D layout (i.e. matrix rows) */
/* of dot products with given matrix         */
/* (filtered by the specified cols).         */
/*-------------------------------------------*/
__host__
bool Algorithms::Reduction::reduce2D_dot_products_modified(
	datatype* Gamma, datatype* columns, datatype* gammaJ, datatype* out, datatype* counters,
	unsigned int pJ, unsigned int colsD, unsigned int colsX) {

	// We also assume that colsD <= 32
	dim3 block(1024);
	dim3 grid((colsX + block.x - 1) / block.x, colsD);
	multi_reduction_smallGamma_times_gammaJ_STAGE1 << <grid, block >> > (
		Gamma, columns, gammaJ, out,
		counters, pJ);
	multi_reduction_smallGamma_times_gammaJ_STAGE2 << <colsD, block >> > (
		out, counters, pJ);
	return true;
}

/*-------------------------------*/
/* Euclidean norm of the vector. */
/*-------------------------------*/
__host__
bool Algorithms::Reduction::euclidean_norm(
	datatype* in, datatype* out,
	unsigned int length, unsigned int recommended) {

	// We assume length i.e. rowsX <= 1024
	EUnorm << <1, recommended >> > (in, out, length);
	return true;
}

/*-----------------------------------------------*/
/* Compute the Round Mean Square Error between a */
/* matrix X and its approximation matrix X~      */
/*-----------------------------------------------*/
__host__
bool Algorithms::Reduction::reduce_RMSE(
	datatype* X, datatype* Xappr, datatype* out,
	unsigned int rows, unsigned int cols, unsigned int iter,
	unsigned int recommended_threads, datatype* bitmap) {

	dim3 block(recommended_threads);
	dim3 grid(cols);
	RMSE_stage1 << <grid, block >> > (X, Xappr, rows, out);
	/********************/
	block.x = 512;
	grid.x = (cols + block.x - 1) / block.x;
	RMSE_stage2 << < grid, block >> > (out, cols, rows*cols, iter, bitmap);
	/********************/
	int N;
	while (grid.x > 1) {
		N = grid.x;
		grid.x = (N + block.x - 1) / block.x;
		RMSE_stage3 << < grid, block >> > (N, rows*cols, iter);
	}
	return true;
}

/*------------------------------------------------*/
/* Compute the Round Mean Square Error between a  */
/* matrix X and its approximation matrix X~       */
/* ( modified for use only in Dictionary Update's */
/* special case )                                 */
/*------------------------------------------------*/
__host__
bool Algorithms::Reduction::reduce_ERROR_for_special_case(
	datatype* X, datatype* Xappr, datatype* out,
	unsigned int rows, unsigned int cols, datatype* counters,
	datatype* unused, datatype* unsig_counter,
	unsigned int recommended_threads) {

	dim3 block(recommended_threads);
	dim3 grid(MIN(cols,MAX_SIGNALS));
	SCase_err_stage1 << <grid, block >> > (X, Xappr, rows, cols, out, counters, unused, unsig_counter);
	/********************/
	int N = grid.x;
	block.x = 1024;
	grid.x = (N + block.x - 1) / block.x;
	SCase_err_stage2 << < grid, block >> > (out, N, Xappr, cols, counters, unsig_counter);
	/********************/
	while (grid.x > 1) {
		N = grid.x;
		grid.x = (N + block.x - 1) / block.x;
		SCase_err_stage3 << < grid, block >> > (N, Xappr, counters);
	}
	return true;
}

/*-------------------------------*/
/* Euclidean norm of the vector. */
/* ONLY for the special case!    */
/*-------------------------------*/
__host__
bool Algorithms::Reduction::SCase_EU_norm(
	datatype* in, datatype* out, unsigned int length,
	unsigned int recommended, datatype* counters, datatype* unused,
	unsigned int colsX) {

	// We assume length <= 1024
	SCase_norm << <1, recommended >> > (in, out, length, counters, unused, MIN(colsX, MAX_SIGNALS));
	return true;
}

/*--------------------------------------*/
/* Set the size of the reduction buffer */
/* to be used in the calculations.      */
/*--------------------------------------*/
bool Algorithms::Reduction::setReductionBuffer(datatype* b) {
	// Implicit Synchronization
	cudaError_t cet;
	if ((cet = cudaMemcpyToSymbol(
			experimental_array, &b, sizeof(datatype*), 0, cudaMemcpyHostToDevice)
		) != cudaSuccess) {

		cerr << "CudaMemcpyToSymbol (reduction buffer) failed: " << cudaGetErrorString(cet) << endl;
		return false;
	}
	return true;
}