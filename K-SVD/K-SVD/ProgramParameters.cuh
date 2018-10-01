/*---------------------------------*/
/* PROGRAM PARAMETERS - EQUIVALENT */
/* OF MATLAB SCRIPT PARAMETERS FOR */
/* THE EXECUTION OF KSVD ALGORITHM */
/*---------------------------------*/
#pragma once
#ifndef __PROGPARAMS__
#define __PROGPARAMS__


/*****************************/
/*    Program parameters     */
/*****************************/
// Number of iterations of the K-SVD algorithm
// MATLAB variable: params.iternum = 30
// Implementation maximum: 1024
#define NUMBERofITERATIONS	30
// Sparsity level threshold
// MATLAB variable: params.Tdata = 6
#define Tdata				6
// Mutual incoherence threshold
// MATLAB variable: params.muthresh = 0.8
#define muTHRESH			0.8


#endif // !__PROGPARAMS__