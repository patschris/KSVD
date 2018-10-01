/*------------------------------*/
/*     UTILITIES  NAMESPACE     */
/*       IMPLEMENTATION         */
/*------------------------------*/
#include <iostream>
#include <iomanip>
#include "GlobalDeclarations.cuh"
#include "Utilities.cuh"

using namespace std;


// Declarations - functions
__global__ void kernelPrint(datatype*, unsigned int, unsigned int, unsigned int, unsigned int);
__global__ void kernelPrintTransposed(datatype*, unsigned int, unsigned int, unsigned int, unsigned int);

/*----------------*/
/* Print an array */
/*----------------*/
void Utilities::print(datatype* array, unsigned int rows, unsigned int cols,
						unsigned int limit_rows, unsigned int limit_cols) {

	cout << "_____________________________________________________" << endl;
	for (size_t i = 0; i < rows && i < limit_rows; i++)
	{
		cout << "# " << setfill('0') << setw(3) << i + 1 << " #   ";
		for (size_t j = 0; j < cols && j < limit_cols; j++)
		{
			cout << setprecision(10) << std::fixed << array[i*cols + j] << "\t";
		}
		cout << endl;
	}
	cout << "_____________________________________________________" << endl;
}

/*-------------------------------*/
/* Print an array (transposed)   */
/*-------------------------------*/
void Utilities::printTransposed(datatype* array, unsigned int rows, unsigned int cols,
								unsigned int limit_rows, unsigned int limit_cols) {

	cout << "_____________________________________________________" << endl;
	for (size_t i = 0; i < rows && i < limit_rows; i++)
	{
		cout << "# " << setfill('0') << setw(3) << i + 1 << " #   ";
		for (size_t j = 0; j < cols && j < limit_cols; j++)
		{
			cout << setprecision(10) << std::fixed << array[i + j*rows] << "\t";
		}
		cout << endl;
	}
	cout << "_____________________________________________________" << endl;
}

/*----------------------*/
/* Print a device array */
/*----------------------*/
void Utilities::printDevice(datatype* array, unsigned int rows, unsigned int cols,
							unsigned int limit_rows, unsigned int limit_cols) {

	SINGLE_THREAD(kernelPrint, (array, rows, cols, limit_rows, limit_cols) );
}

/*-----------------------------------*/
/* Print a device array (transposed) */
/*-----------------------------------*/
void Utilities::printDeviceTransposed(datatype* array, unsigned int rows, unsigned int cols,
										unsigned int limit_rows, unsigned int limit_cols) {

	SINGLE_THREAD(kernelPrintTransposed,(array, rows, cols, limit_rows, limit_cols));
}

/*-------------------------*/
/* Print a device array in */
/* a kernel on the device  */
/*-------------------------*/
__global__
void kernelPrint(datatype* array, unsigned int rows, unsigned int cols,
				unsigned int limit_rows, unsigned int limit_cols) {

	printf("_____________________________________________________\n");
	for (size_t i = 0; i < rows && i < limit_rows; i++)
	{
		printf("#  %03d #   ",i + 1);
		for (size_t j = 0; j < cols && j < limit_cols; j++)
		{
			printf("%.10f\t",array[i*cols + j]);
		}
		printf("\n");
	}
	printf("_____________________________________________________\n");
}

/*-----------------------------------*/
/* Print a device array (transposed) */
/* in a kernel on the device         */
/*-----------------------------------*/
__global__
void kernelPrintTransposed(datatype* array, unsigned int rows, unsigned int cols,
							unsigned int limit_rows, unsigned int limit_cols) {

	printf("_____________________________________________________\n");
	for (size_t i = 0; i < rows && i < limit_rows; i++)
	{
		printf("#  %03d #   ", i + 1);
		for (size_t j = 0; j < cols && j < limit_cols; j++)
		{
			printf("%.10f\t", array[i + j*rows]);
		}
		printf("\n");
	}
	printf("_____________________________________________________\n");
}
