/*------------------------------*/
/* GLOBAL VARIABLES DEFINITIONS */
/*------------------------------*/
#include "Globals.cuh"


/*-----------------------*/
/* Rows of input array X */
/*-----------------------*/
unsigned int Globals::rowsX = 0;

/*--------------------------*/
/* Columns of input array X */
/*--------------------------*/
unsigned int Globals::colsX = 0;

/*-----------------------*/
/* Rows of input array D */
/*-----------------------*/
unsigned int Globals::rowsD = 0;

/*--------------------------*/
/* Columns of input array D */
/*--------------------------*/
unsigned int Globals::colsD = 0;

/*-------------------------------*/
/* Recommended threads per block */
/* for a kernel that uses colsX. */
/*-------------------------------*/
unsigned int Globals::TPB_rowsX = 0;

/*-------------------------------*/
/* Recommended threads per block */
/* for a kernel that uses colsD. */
/*-------------------------------*/
unsigned int Globals::TPB_colsD = 0;