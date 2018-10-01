/*------------------------------*/
/* ALL DEVICE MEMORY OPERATIONS */
/*       IMPLEMENTATION         */
/*		  (HIGH LEVEL)			*/
/*------------------------------*/
#include <iostream>


/*****************
*   NAMESPACES   *
*	__________   *
*****************/
using namespace		std;
using HostSpace		= HostMemory<datatype>;

/*-------------*/
/* Constructor */
/*-------------*/
template < typename T>
DeviceMemory<T>::DeviceMemory(HostSpace& hostMemory)
	: POINTERarray(DeviceMemory<T>::MAX_ELEMENTS + 1, 0), THRUSTarray(DeviceMemory<T>::MAX_ELEMENTS + 1) {

	// Set up memory metrics first!
	// Total used memory
	this->consumed =
		(Globals::rowsX * Globals::colsX) +
		(Globals::rowsD * Globals::colsD) +
		(Globals::colsD) +
		(Globals::colsX) +
		(Globals::rowsD * Globals::colsD * 1024) +
		(Globals::colsD * Globals::colsD) +
		(Globals::colsD * Globals::colsX) +
		(Globals::colsD * Globals::colsX) +
		(Tdata * Globals::colsX) +
		(Globals::colsD * Globals::colsX) +
		(Globals::colsD * Globals::rowsX) +
		(Globals::rowsD * Globals::colsD) +
		(Globals::rowsX * Globals::colsX) +
		(Globals::colsX) +
		(Globals::colsD) +
		(1) + 
		(Globals::colsX);

	this->consumed = this->consumed * sizeof(datatype);
	// Total available global memory
	size_t free_byte;
	size_t total_byte;
	cudaMemGetInfo(&free_byte, &total_byte);
	this->free = (double)free_byte;
	this->total = (double)total_byte;
	this->used = this->total - this->free;

	// Now the arrays!
	// X_ARRAY (rowsX * colsX)
	this->THRUSTarray[X_ARRAY] = thrust::device_vector<T>(
		hostMemory.get(HostSpace::X_ARRAY),
		hostMemory.get(HostSpace::X_ARRAY) + Globals::rowsX*Globals::colsX);
	this->POINTERarray[X_ARRAY] = thrust::raw_pointer_cast(&(this->THRUSTarray[X_ARRAY])[0]);
	/*_______________*/
	/*  More arrays  */
	/*_______________*/
	// TEMP_ROWSD_BY_COLSD
	this->THRUSTarray[TEMP_ROWSD_BY_COLSD] = thrust::device_vector<T>(Globals::rowsD*Globals::colsD);
	this->POINTERarray[TEMP_ROWSD_BY_COLSD] = thrust::raw_pointer_cast(
		&(this->THRUSTarray[TEMP_ROWSD_BY_COLSD])[0]);
	// TEMP_1_BY_COLSD
	this->THRUSTarray[TEMP_1_BY_COLSD] = thrust::device_vector<T>(Globals::colsD);
	this->POINTERarray[TEMP_1_BY_COLSD] = thrust::raw_pointer_cast(
		&(this->THRUSTarray[TEMP_1_BY_COLSD])[0]);
	// TEMP_SINGLE_VALUE ( 1 !by colsX threads )
	this->THRUSTarray[TEMP_SINGLE_VALUE] = thrust::device_vector<T>(Globals::colsX);
	this->POINTERarray[TEMP_SINGLE_VALUE] = thrust::raw_pointer_cast(
		&(this->THRUSTarray[TEMP_SINGLE_VALUE])[0]);
	// REDUCTION_BUFFER ( rowsD * colsD * 1024 )
	// The maximum buffer needed! ( in collincomb )
	this->THRUSTarray[REDUCTION_BUFFER] = thrust::device_vector<T>(Globals::colsD*Globals::rowsD*1024);
	this->POINTERarray[REDUCTION_BUFFER] = thrust::raw_pointer_cast(
		&(this->THRUSTarray[REDUCTION_BUFFER])[0]);
	// G ( colsD * colsD)
	this->THRUSTarray[G] = thrust::device_vector<T>(Globals::colsD*Globals::colsD);
	this->POINTERarray[G] = thrust::raw_pointer_cast(
		&(this->THRUSTarray[G])[0]);
	// DtX ( colsD * colsX)
	this->THRUSTarray[DtX] = thrust::device_vector<T>(Globals::colsD*Globals::colsX);
	this->POINTERarray[DtX] = thrust::raw_pointer_cast(
		&(this->THRUSTarray[DtX])[0]);
	cudaMemset(this->POINTERarray[DtX], 0, Globals::colsD*Globals::colsX);
	// SELECTED_ATOMS ( colsD !by colsX threads )
	this->THRUSTarray[SELECTED_ATOMS] = thrust::device_vector<T>(Globals::colsD*Globals::colsX);
	this->POINTERarray[SELECTED_ATOMS] = thrust::raw_pointer_cast(
		&(this->THRUSTarray[SELECTED_ATOMS])[0]);
	cudaMemset(this->POINTERarray[SELECTED_ATOMS], 0, Globals::colsD*Globals::colsX);
	// c ( Tdata !by colsX threads )
	this->THRUSTarray[c] = thrust::device_vector<T>(Tdata*Globals::colsX);
	this->POINTERarray[c] = thrust::raw_pointer_cast(
		&(this->THRUSTarray[c])[0]);
	cudaMemset(this->POINTERarray[c], 0, Tdata*Globals::colsX);
	// alpha ( colsD !by colsX threads )
	this->THRUSTarray[alpha] = thrust::device_vector<T>(Globals::colsD*Globals::colsX);
	this->POINTERarray[alpha] = thrust::raw_pointer_cast(
		&(this->THRUSTarray[alpha])[0]);
	// TEMP_COLSD_BY_ROWSX ( colsD * rowsX )
	this->THRUSTarray[TEMP_COLSD_BY_ROWSX] = thrust::device_vector<T>(Globals::colsD*Globals::rowsX);
	this->POINTERarray[TEMP_COLSD_BY_ROWSX] = thrust::raw_pointer_cast(
		&(this->THRUSTarray[TEMP_COLSD_BY_ROWSX])[0]);
	// TEMP_ROWSX_BY_COLSX ( rowsX * colsX )
	this->THRUSTarray[TEMP_ROWSX_BY_COLSX] = thrust::device_vector<T>(Globals::rowsX*Globals::colsX);
	this->POINTERarray[TEMP_ROWSX_BY_COLSX] = thrust::raw_pointer_cast(
		&(this->THRUSTarray[TEMP_ROWSX_BY_COLSX])[0]);
	// ERR ( 1 * colsX )
	this->THRUSTarray[ERR] = thrust::device_vector<T>(Globals::colsX);
	this->POINTERarray[ERR] = thrust::raw_pointer_cast(
		&(this->THRUSTarray[ERR])[0]);
	// UNUSED_SIGS ( 1 * colsX )
	this->THRUSTarray[UNUSED_SIGS] = thrust::device_vector<T>(Globals::colsX);
	this->POINTERarray[UNUSED_SIGS] = thrust::raw_pointer_cast(
		&(this->THRUSTarray[UNUSED_SIGS])[0]);
	// REPLACED_ATOMS ( 1 * colsD)
	this->THRUSTarray[REPLACED_ATOMS] = thrust::device_vector<T>(Globals::colsD);
	this->POINTERarray[REPLACED_ATOMS] = thrust::raw_pointer_cast(
		&(this->THRUSTarray[REPLACED_ATOMS])[0]);
	// UNUSED_SIGS_COUNTER ( 1 )
	this->THRUSTarray[UNUSED_SIGS_COUNTER] = thrust::device_vector<T>(1);
	this->POINTERarray[UNUSED_SIGS_COUNTER] = thrust::raw_pointer_cast(
		&(this->THRUSTarray[UNUSED_SIGS_COUNTER])[0]);
	// UNUSED_SIGS_BITMAP ( 1 * colsX )
	this->THRUSTarray[UNUSED_SIGS_BITMAP] = thrust::device_vector<T>(Globals::colsX);
	this->POINTERarray[UNUSED_SIGS_BITMAP] = thrust::raw_pointer_cast(
		&(this->THRUSTarray[UNUSED_SIGS_BITMAP])[0]);
	/*_______________*/
	/* End of arrays */
	/*_______________*/
	// D_ARRAY
	this->THRUSTarray[D_ARRAY] = thrust::device_vector<T>(
		hostMemory.get(HostSpace::D_ARRAY),
		hostMemory.get(HostSpace::D_ARRAY) + Globals::rowsD*Globals::colsD);
	this->POINTERarray[D_ARRAY] = thrust::raw_pointer_cast(&(this->THRUSTarray[D_ARRAY])[0]);
}

/*------------*/
/* Destructor */
/*------------*/
template < typename T>
DeviceMemory<T>::~DeviceMemory() {
	cout << "# Device Memory Cleared!" << endl;
}

/*----------------*/
/* Public Methods */
/*----------------*/

// Return an array pointer
template < typename T> inline
T* DeviceMemory<T>::get(unsigned int index) {
	return this->POINTERarray.at(index);
}

// Return a thrust device array pointer
template < typename T> inline
thrust::device_vector<T>&
DeviceMemory<T>::getThrust(unsigned int index) {
	return this->THRUSTarray.at(index);
}

// Return total available device memory
template < typename T> inline
double DeviceMemory<T>::getTotal() {
	return this->total;
}

// Return total memory used by the OS
template < typename T> inline
double DeviceMemory<T>::getUsed() {
	return this->used;
}

// Return free global memory
template < typename T> inline
double DeviceMemory<T>::getFree() {
	return this->free;
}

// Return total memory used by our program
template < typename T> inline
double DeviceMemory<T>::getConsumed() {
	return this->consumed;
}