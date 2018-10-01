/*------------------------------*/
/*       FILE OPERATIONS        */
/*       IMPLEMENTATION         */
/*------------------------------*/
#include <iostream>
#include "GlobalDeclarations.cuh"
#include "Globals.cuh"
#include "FileManager.cuh"

using namespace std;


// Declarations
bool readMat(datatype**,mxArray**,unsigned int*,unsigned int*,char*,char*);
bool writeMat(datatype*,unsigned int,unsigned int, char*, char*);
unsigned int next_pow_2(unsigned int);

// Macros
#define MIN2(a,b) (a < b ? a : b)

/*---------------*/
/* Constructor   */
/*---------------*/
FileManager::FileManager() {
	this->aX = NULL;
	this->aD = NULL;
}

/*--------------*/
/* Destructor   */
/*--------------*/
FileManager::~FileManager() {
	if (this->aX) {
		mxDestroyArray(this->aX);
	}
	if (this->aD) {
		mxDestroyArray(this->aD);
	}
}

/*-------------------------------*/
/* Read X array stored in file   */
/*-------------------------------*/
bool FileManager::readXarray(datatype** array) {
	if (readMat(
		array, &(this->aX),
		&Globals::rowsX, &Globals::colsX,
		Xarray_in_PATH, "X") == false) {

		cerr << "An error occured while reading file!..." << endl;
		return false;
	}
	Globals::TPB_rowsX = MIN2(next_pow_2(Globals::rowsX), 1024);
	return true;
}

/*-------------------------------*/
/* Read D array stored in file   */
/*-------------------------------*/
bool FileManager::readDarray(datatype** array) {
	if (readMat(
		array, &(this->aD),
		&Globals::rowsD, &Globals::colsD,
		Darray_in_PATH, "D") == false) {

		cerr << "An error occured while reading file!..." << endl;
		return false;
	}
	Globals::TPB_colsD = MIN2(next_pow_2(Globals::colsD), 32);
	return true;
}

/*-------------------------*/
/* Write D array to file   */
/*-------------------------*/
bool FileManager::writeDarray(datatype* array) {
	if ( writeMat(array, Globals::rowsD, Globals::colsD, Darray_out_PATH,"FinalD") == false) {
		cerr << "An error occured while writing file!..." << endl;
		return false;
	}
	return true;
}

/*-----------------------------*/
/* Write given array to file   */
/*-----------------------------*/
bool FileManager::writeTemporary(datatype* array, unsigned int rows, unsigned int cols,char* array_name) {
	if (writeMat(array, rows, cols, temp_out_PATH, array_name) == false) {
		cerr << "An error occured while writing file!..." << endl;
		return false;
	}
	return true;
}

/*---------------------------*/
/* Read MATLAB file format   */
/*---------------------------*/
bool readMat(datatype** array_dbl,mxArray **mArray,
			unsigned int* rows_ptr,unsigned int* cols_ptr,
			char* file,char* array_name) {

	MATFile *pmat;
	pmat = matOpen(file, "r");
	if (pmat == NULL) {
		cerr << "Error opening file " << file << " ..." << endl;
		return(false);
	}
	*mArray = matGetVariable(pmat, array_name);
	if (*mArray == NULL) {
		cerr << "Error reading existing matrix X..." << endl;
		matClose(pmat);
		return(false);
	}
	#ifdef DOUBLE
		*array_dbl = mxGetPr(*mArray);
	#else
		*array_dbl = (datatype*)mxGetData(*mArray);
	#endif
	*rows_ptr = (unsigned int) mxGetM(*mArray);
	*cols_ptr = (unsigned int) mxGetN(*mArray);
	if (matClose(pmat) != 0) {
		cerr << "Error closing file " << file << "..." << endl;
		return(false);
	}
	return true;
}

/*----------------------------*/
/* Write MATLAB file format   */
/*----------------------------*/
bool writeMat(datatype* array,unsigned int rows,unsigned int cols, char* file, char* array_name) {
	MATFile *pmat;
	pmat = matOpen(file, "w");
	if (pmat == NULL) {
		cerr << "Error opening file " << file << " ..." << endl;
		return(false);
	}
	
	mxArray *pa2 = mxCreateDoubleMatrix(rows, cols, mxREAL);
	if (pa2 == NULL) {
		cerr << "Error: Out of memory while writing existing matrix X..." << endl;
		matClose(pmat);
		return(false);
	}
	memcpy((void *)(mxGetPr(pa2)), (void *)array, rows * cols * sizeof(datatype));

	int status = matPutVariableAsGlobal(pmat, array_name, pa2);
	if (status != 0) {
		cerr << "Error using matPutVariableAsGlobal..." << endl;
		matClose(pmat);
		return(false);
	}
	/* clean up */
	mxDestroyArray(pa2);
	if (matClose(pmat) != 0) {
		cerr << "Error closing file " << file << "..." << endl;
		return(false);
	}
	return true;
}

/*-------------------------------*/
/* Helper function to find the   */
/* next power of 2 for the given */
/* number.                       */
/*-------------------------------*/
unsigned int next_pow_2(unsigned int v) {
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
}