/*------------------------------*/
/* ALL HOST MEMORY OPERATIONS   */
/*       IMPLEMENTATION         */
/*		  (HIGH LEVEL)			*/
/*------------------------------*/
#include <iostream>

using namespace std;


/*-------------*/
/* Constructor */
/*-------------*/
template < typename T>
HostMemory<T>::HostMemory(FileManager& fm) : array(HostMemory<T>::MAX_ELEMENTS + 1, 0) {
	/* Arrays in vector */
	if (fm.readXarray(&this->array[X_ARRAY]) == false) {
		cerr << "Cannot read file for X! Aborting..." << endl;
		return;
	}
	if (fm.readDarray(&this->array[D_ARRAY]) == false) {
		cerr << "Cannot read file for D! Aborting..." << endl;
		return;
	}
	/* End of arrays in vector */

	if ((this->rand_buffer = new(nothrow) unsigned int[Globals::colsD]) == NULL) {
		cerr << "Allocation for rand_gen_buffer failed!..." << endl;
		return;
	}
}

/*------------*/
/* Destructor */
/*------------*/
template < typename T>
HostMemory<T>::~HostMemory() {
	this->array[X_ARRAY] = NULL;
	this->array[D_ARRAY] = NULL;
	for (int i = 0; i <= HostMemory<T>::MAX_ELEMENTS; i++) {
		if (this->array[i]){
			delete[] this->array[i];
		}
	}
	if (this->rand_buffer) {
		delete[] this->rand_buffer;
	}
	cout << "# Host Memory Cleared!" << endl;
}

/*----------------*/
/* Public Methods */
/*----------------*/
// Return an array pointer
template < typename T> inline
T* HostMemory<T>::get(unsigned int index) {
	return this->array.at(index);
}

//Return the random generator buffer
template < typename T> inline
unsigned int* HostMemory<T>::get_RND_GNRT_buffer() {
	return this->rand_buffer;
}