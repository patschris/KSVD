/*-------------------------------*/
/* GLOBAL CONSTANTS - STRUCTURES */
/*-------------------------------*/
#pragma once
#ifndef __DECLARATIONS__
#define __DECLARATIONS__


/***************
*   INCLUDES   *
*	________   *
***************/
// These should be included from every 
// compilation unit
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include "ProgramParameters.cuh"

/**************************
*   TYPEDEFS - SWITCHES   *
*	___________________	  *
**************************/
// Comment out the following #define
// for single-precision operations
// or leave as-is for double precision 
#define DOUBLE
// The type of data used in the
// program ( can be either float 
// or double)
#ifdef	DOUBLE
typedef double datatype;
#else
typedef float datatype;
#endif // !DOUBLE

/******************************
*   DEFINITIONS - CONSTANTS   *
*	_______________________	  *
******************************/
// Number of active threads per reduction.
// (Half of the number of elements per single reduction,
// since only half of them are needed).
// Current: 512 threads/1024 elements
#define REDUCTION_BLCK_SIZE	512
// Random generator mode of operation:
// fully random ( 0 ) or pre-defined 
// numbers for comparison to MATLAB (1).
#define STATIC_GENERATION	0
// At least this number of samples must use the
// atom to be kept and not be cleared. This 
// threshold is used when clearing the dictionary
// in the last step of the KSVD alogorithm.
#define USE_THRESH			4
// We use the square value of the Mutual incoherence
// threshold in the program so we #define it explicitly
// as well
#define SQR_muTHRESH		muTHRESH * muTHRESH
// Maximum number of signals used in the special case that
// sprow() did not find any non-zero element in a row of 
// Gamma. Consequently we use at most that many signals 
// to construct a new atom.
#define MAX_SIGNALS			5000

/************
*   PATHS   *
*	_____	*
************/
#define	Xarray_in_PATH		"./input/X.mat"
#define	Darray_in_PATH		"./input/D.mat"
#define	Darray_out_PATH		"./output/D.mat"
#define	temp_out_PATH		"./output/temp.mat"
#define	MATLAB_include		D:\0_SSD_Program_Files\MATLAB\extern\include\\

/*************
*   MACROS   *
*	______	 *
*************/
#define QUOTEME(x)			QUOTEME_1(x)
#define QUOTEME_1(x)		#x
#define INCLUDE_F(x,y)		QUOTEME(x ## y)
#define INCLUDE_FILE(x,y)	INCLUDE_F(x,y)
#define SINGLE_THREAD(x,y)	x ## <<<1,1>>> ## y
#define	SQR(x)				(x)*(x)
#define MIN(a,b)			( a < b ? a : b)

#endif // !__DECLARATIONS__