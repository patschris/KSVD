/*------------------------------*/
/*  REDUCTION ONLY HEADER FILE  */
/*------------------------------*/
#pragma once
#ifndef _REDHF_
#define _REDHF_


//////////////////////////////////////////////
// This file serves as a separator between  //
// reduction kernel functions and reduction //
// high-level API ( in Reduction.cu )       //
//////////////////////////////////////////////

/***************************
*         MACROS           *
*	      ______           *
***************************/
// Macro for finding the minimum of 2 numbers
#define MINIMUM(a,b) ( a < b ? a : b )
#define MAXIMUM(a,b) ( a > b ? a : b )

/*********************************
*         DECLARATIONS           *
*	      ____________           *
*********************************/
// K-SVD operations
__global__
void deviceReduceKernel_2D_square_values(datatype *, int, datatype*);
// Dict. Upd. operations
__global__
void multi_reduction_squared_sum_STAGE1(datatype*, datatype*, datatype*, int);
__global__
void multi_reduction_squared_sum_STAGE2_together_collincomb(datatype*, datatype*, datatype*, int, datatype*, 
															datatype*, datatype*, unsigned int);
__global__
void multi_reduction_smallGamma_times_gammaJ_STAGE1(datatype *, datatype*, datatype*, datatype*, datatype*, unsigned int);
__global__
void multi_reduction_smallGamma_times_gammaJ_STAGE2(datatype*, datatype*, unsigned int);
__global__
void rowlincomb(datatype*, datatype*, datatype*, datatype*, unsigned int, datatype*, datatype*, datatype*,
				datatype*, unsigned int);
__global__
void SCase_err_stage1(datatype*, datatype*, unsigned int, unsigned int, datatype*, datatype*, datatype*, datatype*);
__global__
void SCase_err_stage2(datatype*, int, datatype*, unsigned int, datatype*, datatype*);
__global__
void SCase_err_stage3(int, datatype*, datatype*);
__global__
void SCase_norm(datatype*, datatype*,unsigned int, datatype*, datatype*, unsigned int);
// Error computation operations
__global__
void RMSE_stage1(datatype*, datatype*, unsigned int, datatype*);
__global__
void RMSE_stage2(datatype*, unsigned int, unsigned int, unsigned int, datatype*);
__global__
void RMSE_stage3(unsigned int, unsigned int, unsigned int);
// Clear Dict. operations
__global__
void device_max_err_STAGE1(datatype*, int, datatype*);
__global__
void device_max_err_STAGE2(datatype*, int, datatype*);
__global__
void device_max(datatype*, unsigned int, datatype*, datatype*, unsigned int);
__global__
void EUnorm(datatype*, datatype*, unsigned int);

/***********************
*         END          *
*	      ___          *
***********************/

#endif // !REDHF