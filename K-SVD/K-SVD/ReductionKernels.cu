/*----------------------------*/
/* REDUCTION KERNEL FUNCTIONS */
/*      IMPLEMENTATION        */
/*----------------------------*/
#include "GlobalDeclarations.cuh"
#include "ReductionBase.cuh"
#include "ReductionKernels.cuh"
#include "Algorithms.cuh"
#include <cstdio>

using namespace std;


/*********************************
*         DECLARATIONS           *
*	      ____________           *
*********************************/
__device__
datatype* experimental_array; 


/************************************
*    BATCHED REDUCTION WRAPPERS     *
*	 __________________________     *
************************************/

/*--------------------------------*/
/*  2-STAGE MULTIBLOCK REDUCTION  */
/*          *STAGE 1*             */
/*--------------------------------*/
__global__ void multi_reduction_squared_sum_STAGE1(
	datatype* in, datatype* out, datatype* counter, int stride) {

	// blockIdx.y = row = reduction id
	// blockIdx.x = reduction axis
	int N = *((int*)(counter + blockIdx.y));
	datatype sum = 0;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < N) {
		// Retrieve value filtered through function
		sum = sqval(in[blockIdx.y*stride + index]);
	}
	sum = blockReduceSum(sum);
	if (threadIdx.x == 0) {
		// Write to output
		if (gridDim.x > 1) {
			experimental_array[blockIdx.y * 1024 + blockIdx.x] = sum;
		}
		else {
			*(out + blockIdx.y) = sum;
		}
	}
}

/*--------------------------------*/
/*  2-STAGE MULTIBLOCK REDUCTION  */
/*           *STAGE 2*            */  
/* ALSO COLLINCOMB FUSED IN KERNEL*/
/* TO INCREASE CONCURRENCY.       */
/*--------------------------------*/
__global__ void multi_reduction_squared_sum_STAGE2_together_collincomb(
	datatype* in, datatype* out, datatype* counter, int stride,
	datatype *A, datatype* columns, datatype* out2,
	unsigned int rowsX) {

	if (blockIdx.y == 0) {
		/**/
		/*             *STAGE 2*            */
		/**/
		int N = (*((int*)(counter + blockIdx.x)) + 1024 - 1) / 1024;
		datatype sum = 0;
		if (threadIdx.x < N) {
			sum = experimental_array[blockIdx.x * 1024 + threadIdx.x];
		}
		sum = blockReduceSum(sum);
		if (threadIdx.x == 0) {
			// Write to output
			*(out + blockIdx.x) = sum;
		}
	}
	else {
		/**/
		/*           *COLLINCOMB*            */
		/**/
		if (threadIdx.x < rowsX) {
			// Multi-Reduction sub-id: blockIdx.x
			columns += blockIdx.x * stride;
			in += blockIdx.x * stride;
			out2 += blockIdx.x * rowsX;
			unsigned int N = *((int*)(counter + blockIdx.x));
			datatype sum = 0;
			for (int i = 0; i < N; i++) {
				sum += A[((int)columns[i])*rowsX + threadIdx.x] * in[i];
			}
			*(out2 + threadIdx.x) = sum;
		}
	}
}

/*------------------------------------*/
/* 2D + 2-STAGE REDUCTION FOR MATRIX- */
/* VECTOR PRODUCT FILTERED BY COLUMNS */
/*             *STAGE 1*              */
/*------------------------------------*/
__global__ void multi_reduction_smallGamma_times_gammaJ_STAGE1(
	datatype *Gamma, datatype* columns, datatype* gammaJ, datatype* out,
	datatype* counters, unsigned int pJ) {

	// blockIdx.x - reduction axis
	// blockIdx.y - reduction ID axis
	int N = *((int*)counters);
	if (N != 0) {
		if (pJ == blockIdx.y) {
			if (blockIdx.x == 0 && threadIdx.x == 0) {
				*(out + pJ) = *(out + pJ - gridDim.y);
			}
			return;
		}
		datatype sum = 0;
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < N) {
			// Retrieve value filtered through columns
			sum = Gamma[((int)columns[index])*gridDim.y + blockIdx.y] * gammaJ[index];
		}
		sum = blockReduceSum(sum);
		if (threadIdx.x == 0) {
			// Write to output
			N = (N + 1024 - 1) / 1024;
			if (N > 1) {
				experimental_array[blockIdx.y * 1024 + blockIdx.x] = sum;
			}
			else {
				if (blockIdx.x == 0) {
					*(out + blockIdx.y) = sum;
				}
			}
		}
	}
}

/*------------------------------------*/
/* 2D + 2-STAGE REDUCTION FOR MATRIX- */
/* VECTOR PRODUCT FILTERED BY COLUMNS */
/*             *STAGE 2*              */
/*------------------------------------*/
__global__ void multi_reduction_smallGamma_times_gammaJ_STAGE2(
	datatype* out, datatype* counters, unsigned int pJ) {

	int N = *((int*)counters);
	if (N != 0) {
		if (pJ == blockIdx.x) {
			return;
		}
		N = (N + 1024 - 1) / 1024;
		if (N > 1) {
			datatype sum = 0;
			if (threadIdx.x < N) {
				// Retrieve partial sum
				sum = experimental_array[blockIdx.x * 1024 + threadIdx.x];
			}
			sum = blockReduceSum(sum);
			if (threadIdx.x == 0) {
				// Write to output
				*(out + blockIdx.x) = sum;
			}
		}
	}
}


/************************************
*    REDUCTION STAGES FUNCTIONS     *
*	 __________________________     *
************************************/

//////////////////////////////////////
// Reduction across multiple blocks //
//////////////////////////////////////
__global__ void deviceReduceKernel_2D_square_values(datatype *in, int N, datatype* out) {
	datatype sum = 0;
	if (threadIdx.x < N) {
		// Retrieve value filtered through function
		sum = sqval(in[blockIdx.x*N + threadIdx.x]);
	}
	sum = blockReduceSum(sum);
	if (threadIdx.x == 0) {
		// Write to output
		out[blockIdx.x] = invSQRT(sum);
	}
}

//////////////////////////////////////
// - 1 STAGE MULTIBLOCK REDUCTION - //
// -  LINEAR COMBINATION OF ROWS  - //
//////////////////////////////////////
__global__ void rowlincomb(
	datatype *A, datatype* x, datatype* out, datatype* cols,
	unsigned int rowsX, datatype* counters, datatype* out2,
	datatype* D, datatype* out3, unsigned int colsX) {

	unsigned int N;
	if ( (N = *((int*)counters)) != 0) {
		if (blockIdx.x < N) {
			datatype sum = 0;
			if (threadIdx.x < rowsX) {
				// Retrieve value filtered through columns and rows
				sum = A[(int)cols[blockIdx.x] * rowsX + threadIdx.x] * x[threadIdx.x];
			}
			sum = blockReduceSum(sum);
			if (threadIdx.x == 0) {
				// Write to output
				out[blockIdx.x] = sum;
			}
		}
		else if (blockIdx.x >= colsX) {
			if (blockIdx.x == gridDim.x - 1) {
				// Calculate the norm of the new atom
				// by combining the previous results.
				//
				datatype sum = 0;
				if (threadIdx.x < rowsX) {
					// Retrieve value filtered through sqval
					sum = sqval(x[threadIdx.x]);
				}
				sum = blockReduceSum(sum);
				if (threadIdx.x == 0) {
					// Write to output
					*out2 = sqrt(sum);
				}
			}
			else {
				// We now calculate the following
				// vector-matrix multiplication in
				// parallel:
				//            (atom'*D)
				//
				datatype sum = 0;
				if (threadIdx.x < rowsX) {
					// Retrieve column
					sum = D[(blockIdx.x - colsX)*rowsX + threadIdx.x] * x[threadIdx.x];
				}
				sum = blockReduceSum(sum);
				if (threadIdx.x == 0) {
					// Write to output
					out3[blockIdx.x - colsX] = sum;
				}
			}
		}
	}
}

//////////////////////////////////////
// - 3 STAGE MULTIBLOCK REDUCTION - //
// -       RMSE CALCULATION       - //
// -         *STAGE 1*            - //  
//////////////////////////////////////
__global__ void RMSE_stage1(datatype* X, datatype* Xappr, unsigned int N, datatype* out) {

	datatype sum = 0;
	if (threadIdx.x < N) {
		// Retrieve value and compute difference
		sum = sqval(X[blockIdx.x*N + threadIdx.x] - Xappr[blockIdx.x*N + threadIdx.x]);
	}
	sum = blockReduceSum(sum);
	if (threadIdx.x == 0) {
		// Write to output
		out[blockIdx.x] = sum;
	}
}

//////////////////////////////////////
// - 3 STAGE MULTIBLOCK REDUCTION - //
// -       RMSE CALCULATION       - //
// -          *STAGE 2*           - //  
//////////////////////////////////////
__global__ void RMSE_stage2(
	datatype* in, unsigned int N, unsigned int size, unsigned int iter,
	datatype* bitmap) {

	datatype sum = 0;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < N) {
		// Retrieve value
		sum = in[index];
		in[index] = sum * bitmap[index];
	}
	sum = blockReduceSum(sum);
	if (threadIdx.x == 0) {
		 //Write to output
		if (gridDim.x > 1) {
			experimental_array[blockIdx.x] = sum;
		}
		else {
			printf("Iteration %02d / %d complete, RMSE = %.5f\n", iter, NUMBERofITERATIONS, sqrt(sum / size));
		}
	}
}

//////////////////////////////////////
// - 3 STAGE MULTIBLOCK REDUCTION - //
// -       RMSE CALCULATION       - //
// -          *STAGE 3*           - //  
//////////////////////////////////////
__global__ void RMSE_stage3(unsigned int N, unsigned int size, unsigned int iter) {

	datatype sum = 0;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < N) {
		// Retrieve value
		sum = experimental_array[index];
	}
	sum = blockReduceSum(sum);
	if (threadIdx.x == 0) {
		//Write to output
		if (gridDim.x > 1) {
			experimental_array[blockIdx.x] = sum;
		}
		else {
			printf("Iteration %02d / %d complete, RMSE = %.5f\n", iter, NUMBERofITERATIONS, sqrt(sum / size));
		}
	}
}

//////////////////////////////////////////
// -  MULTI-STAGE REDUCTION TO BUFFER  -//
// - BLOCK-WIDE MAXIMUM VALUE & INDEX  -//
// -            *STAGE 1*             - //  
//////////////////////////////////////////
__global__ void device_max_err_STAGE1(datatype* in, int N, datatype* out) {

	if (!(*out)) {
		// Our atom does not need replacement
		return;
	}
	DoubleReductionType myElement;
	myElement.value = 0;
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < N) {
		// Retrieve value
		myElement.value = in[index];
		// Store index
		myElement.index = index;
	}
	myElement = blockReduceMaximumIndex(myElement);
	if (threadIdx.x == 0) {
		// Write to output
		if (gridDim.x > 1) {
			((DoubleReductionType*)experimental_array)[blockIdx.x] = myElement;
		}
		else {
			*(out + 1) = myElement.index;
			in[myElement.index] = 0;
		}
	}
}

//////////////////////////////////////////
// -  MULTI-STAGE REDUCTION TO BUFFER  -//
// - BLOCK-WIDE MAXIMUM VALUE & INDEX  -//
// -            *STAGE 2*             - //  
//////////////////////////////////////////
__global__ void device_max_err_STAGE2(datatype* in, int N, datatype* out) {

	if (!(*out)) {
		// Our atom does not need replacement
		return;
	}
	DoubleReductionType myElement;
	myElement.value = 0;
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < N) {
		// Retrieve value
		myElement = ((DoubleReductionType*)experimental_array)[index];
	}
	myElement = blockReduceMaximumIndex(myElement);
	if (threadIdx.x == 0) {
		// Write to output
		if (gridDim.x > 1) {
			((DoubleReductionType*)experimental_array)[blockIdx.x] = myElement;
		}
		else {
			*(out + 1) = myElement.index;
			in[myElement.index] = 0;
		}
	}
}

///////////////////////////////////////////
// -  SINGLE-STAGE IN-PLACE REDUCTION   -//
// -  COLUMN-WISE MAXIMUM SQUARE VALUE  -//
// -           SINGLE WARP              -//
///////////////////////////////////////////
__global__ void device_max(
	datatype* G, unsigned int colsD, datatype* usecount, datatype* replaced,
	unsigned int j) {

	datatype max = 0;
	if (threadIdx.x < colsD && threadIdx.x != j) {
		// Retrieve value
		max = sqval( G[threadIdx.x] );
	}
	max = warpReduceMax(max);
	if (threadIdx.x == 0) {
		// Write to output
		*G = ( ( max > SQR_muTHRESH ) || ( *((unsigned int*)usecount) < USE_THRESH ) ) 
				&& (*replaced == 0);
	}
}

//////////////////////////////////////////
// - 1 STAGE EUCLEDIAN NORM REDUCTION - //
//////////////////////////////////////////
__global__ void EUnorm(datatype* in, datatype* out, unsigned int N) {

	if (!(*out)) {
		// Our atom does not need replacement
		return;
	}
	in += (unsigned int)(*(out + 1))*N;
	datatype sum = 0;
	if (threadIdx.x < N) {
		// Retrieve value filtered through sqval
		sum = sqval(in[threadIdx.x]);
	}
	sum = blockReduceSum(sum);
	if (threadIdx.x == 0) {
		// Write to output
		*(out + 2) = sqrt(sum);
	}
}

////////////////////////////////////////
// - 1 STAGE SPECIAL CASE REDUCTION - //
// -       ERROR CALCULATION        - //
////////////////////////////////////////
__global__ void SCase_err_stage1(
	datatype* X, datatype* Xappr, unsigned int N, unsigned int colsX,
	datatype* out, datatype* counters, datatype* unused, 
	datatype* UScounter) {

	if (*((int*)counters) == 0) {
		if (blockIdx.x < (colsX - *UScounter)) {
			unsigned int column = unused[myGenerator(blockIdx.x, blockDim.x)];
			datatype sum = 0;
			if (threadIdx.x < N) {
				// Retrieve value and compute difference
				sum = sqval(X[column*N + threadIdx.x] - Xappr[column*N + threadIdx.x]);
			}
			sum = blockReduceSum(sum);
			if (threadIdx.x == 0) {
				// Write to output
				out[blockIdx.x] = sum;
			}
		}
	}
}

////////////////////////////////////////
// - 2 STAGE SPECIAL CASE REDUCTION - //
// -      MAXIMUM CALCULATION       - //
////////////////////////////////////////
__global__ void SCase_err_stage2(
	datatype* in, int N, datatype* out, unsigned int colsX,
	datatype* counters, datatype* UScounter) {

	if (*((int*)counters) == 0) {
	
		DoubleReductionType myElement;
		myElement.value = 0;
		int index = blockIdx.x*blockDim.x + threadIdx.x;
		if (index < N && index < (colsX - *UScounter) ) {
			// Retrieve value
			myElement.value = in[index];
			// Store index
			myElement.index = index;
		}
		myElement = blockReduceMaximumIndex(myElement);
		if (threadIdx.x == 0) {
			// Write to output
			if (gridDim.x > 1) {
				((DoubleReductionType*)experimental_array)[blockIdx.x] = myElement;
			}
			else {
				*out = myElement.index;
			}
		}
	}
}

////////////////////////////////////////
// - 2 STAGE SPECIAL CASE REDUCTION - //
// -      MAXIMUM CALCULATION       - //
////////////////////////////////////////
__global__ void SCase_err_stage3(
	int N, datatype* out, datatype* counters) {

	if (*((int*)counters) == 0) {

		DoubleReductionType myElement;
		myElement.value = 0;
		int index = blockIdx.x*blockDim.x + threadIdx.x;
		if (index < N) {
			// Retrieve value
			myElement = ((DoubleReductionType*)experimental_array)[index];
		}
		myElement = blockReduceMaximumIndex(myElement);
		if (threadIdx.x == 0) {
			// Write to output
			if (gridDim.x > 1) {
				((DoubleReductionType*)experimental_array)[blockIdx.x] = myElement;
			}
			else {
				*out = myElement.index;
			}
		}
	}
}

//////////////////////////////////////////
// - 1 STAGE EUCLEDIAN NORM REDUCTION - //
//////////////////////////////////////////
__global__ void SCase_norm(
	datatype* in, datatype* out, unsigned int N,
	datatype* counters, datatype* unused, unsigned int dim) {

	if(*((int*)counters) == 0) {
		in += (unsigned int)unused[myGenerator((unsigned int)(*out), dim)] * N;
		datatype sum = 0;
		if (threadIdx.x < N) {
			// Retrieve value filtered through sqval
			sum = sqval(in[threadIdx.x]);
		}
		sum = blockReduceSum(sum);
		if (threadIdx.x == 0) {
			// Write to output
			*(out + 1) = sqrt(sum);
		}
	}
}