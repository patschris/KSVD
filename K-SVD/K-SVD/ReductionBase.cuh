/*------------------------------*/
/*  REDUCTION ONLY HEADER FILE  */
/*------------------------------*/
#pragma once
#ifndef _REDBASE_
#define _REDBASE_


/////////////////////////////////////////////
// This file serves as a separator between //
// reduction base functions and reduction  //
// higher-level kernels.                   //
/////////////////////////////////////////////

/************************************
*   REDUCTION-SPECIFIC USER TYPES   *
*	_____________________________   *
************************************/

/*-------------------------------------*/
/* Finding the index of the maximum    */
/* or minimum value of a set of values */
/* requires interleaved, multiple      */
/* reduction units per operation (one  */
/* for max and one for index).         */
/*-------------------------------------*/
struct DoubleReductionType {
	datatype	value;
	int			index;
};

/************************************
*     REDUCTION BASE FUNCTIONS      *
*	  ________________________      *
************************************/

////////////////////////////////////
// Reduction within a single warp //
////////////////////////////////////
__inline__ __device__
DoubleReductionType warpReduceMaximumIndex(DoubleReductionType val) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2) {
		datatype shfl_val = __shfl_down(val.value, offset);
		int shfl_ind = __shfl_down(val.index, offset);
		if (shfl_val > val.value) {
			val.value = shfl_val;
			val.index = shfl_ind;
		}
	}
	return val;
}

////////////////////////////////////
// Reduction within a single warp //
////////////////////////////////////
__inline__ __device__
datatype warpReduceMax(datatype val) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2) {
		datatype shfl_val = __shfl_down(val, offset);
		if (shfl_val > val)
			val = shfl_val;
	}
	return val;
}

////////////////////////////////////
// Reduction within a single warp //
////////////////////////////////////
__inline__ __device__
datatype warpReduceSum(datatype val) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
		val += __shfl_down(val, offset);
	return val;
}

/////////////////////////////////////
// Reduction within a single block //
/////////////////////////////////////
__inline__ __device__
datatype blockReduceSum(datatype val) {

	// Max. block size = 1024 (32 x 32) = 32 warps of 32 threads 
	static __shared__
		datatype shared[32];	// Shared mem for 32 partial sums
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	val = warpReduceSum(val);	// Each warp performs partial reduction

	if (lane == 0)				// Write reduced value to shared memory
		shared[wid] = val;

	__syncthreads();			// Wait for all partial reductions

								//read from shared memory only if that warp existed
	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

	if (wid == 0)				//Final reduce within first warp
		val = warpReduceSum(val);

	return val;
}

/////////////////////////////////////
// Reduction within a single block //
// (maximum version)               //
/////////////////////////////////////
__inline__ __device__
DoubleReductionType blockReduceMaximumIndex(DoubleReductionType val) {

	// Max. block size = 1024 (32 x 32) = 32 warps of 32 threads 
	static __shared__
		DoubleReductionType shared[32];	// Shared mem for 32 partial sums
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	val = warpReduceMaximumIndex(val);	// Each warp performs partial reduction

	if (lane == 0)				// Write reduced value to shared memory
		shared[wid] = val;

	__syncthreads();			// Wait for all partial reductions

								//read from shared memory only if that warp existed
	if (threadIdx.x < blockDim.x / warpSize) {
		val = shared[lane];
	}
	else {
		val.value = 0;
	}

	if (wid == 0)				//Final reduce within first warp
		val = warpReduceMaximumIndex(val);

	return val;
}

/************************************
*         UNARY FUNCTIONS           *
*	      _______________           *
************************************/

/*---------------------------------*/
/* Action performed on data before */
/*  reduction of sum of elements!  */
/*     'SUM OF SQUARE VALUES'      */
/*---------------------------------*/
__inline__ __device__
datatype sqval(datatype val) {
	return val*val;
}

/*---------------------------------*/
/* Action performed on data after  */
/*  reduction of sum of elements!  */
/*     'INVERTED SQUARE ROOT'      */
/*---------------------------------*/
__inline__ __device__
datatype invSQRT(datatype val) {
	return 1.0 / sqrt(val);
}

/*******************************
*         GENERATORS           *
*	      __________           *
*******************************/
// This functions acts as a random shuffle generator
// on the device by taking an index and then returning
// the new index that the specific position maps to.

/*-----------------------------------------------------*/
/* This function acts as a random shuffle generator    */
/* on the device by taking an index and then returning */
/* the new index that the specific position maps to.   */
/*-----------------------------------------------------*/
__inline__ __device__
unsigned int myGenerator(unsigned int pos, unsigned int N) {
	// We return the permutation: (1:length) so that every position maps to its self.
	return pos;
}

/***********************
*         END          *
*	      ___          *
***********************/

#endif // !REDBASE