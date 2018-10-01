/*------------------------------*/
/* ALL DEVICE MEMORY OPERATIONS */
/*		  (HIGH LEVEL)			*/
/*------------------------------*/
#pragma once
#ifndef _DEVMEM_
#define _DEVMEM_


// Class wrapper for device memory
// allocations
template < typename T>
class DeviceMemory {

	private:
		// Private members
		std::vector< T* >							POINTERarray;
		std::vector< thrust::device_vector<T> >		THRUSTarray;
		double										total, free, used, consumed;

	public:

		// Constructor/Destructor
		DeviceMemory(HostMemory<datatype>&);
		~DeviceMemory();

		// Public Methods 
		T*							get(unsigned int);
		thrust::device_vector<T>&	getThrust(unsigned int);
		double						getTotal();
		double						getFree();
		double						getUsed();
		double						getConsumed();

		// Public Types
		enum {
			X_ARRAY = 0,
			/*_______________*/
			/*  More arrays  */
			/*_______________*/
			// TEMPORARY
			TEMP_ROWSD_BY_COLSD,
			TEMP_1_BY_COLSD,
			TEMP_SINGLE_VALUE,
			// CONSTANT
			REDUCTION_BUFFER,
			// INTERMEDIATE RESULTS - batch.OMP
			G,DtX,SELECTED_ATOMS,
			c,alpha,
			// INTERMEDIATE RESULTS - Dictionary Update
			TEMP_COLSD_BY_ROWSX,
			UNUSED_SIGS,
			UNUSED_SIGS_COUNTER,
			UNUSED_SIGS_BITMAP,
			REPLACED_ATOMS,
			// INTERMEDIATE RESULTS - Error computation
			TEMP_ROWSX_BY_COLSX,
			ERR,
			/*_______________*/
			/* End of arrays */
			/*_______________*/
			D_ARRAY,
			MAX_ELEMENTS = D_ARRAY
		};

};

// Include the implementation because 
// separate templated methods won't link
#include "DeviceMemory.cu"

#endif // !_DEVMEM_