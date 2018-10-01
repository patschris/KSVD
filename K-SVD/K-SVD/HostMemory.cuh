/*----------------------------*/
/* ALL HOST MEMORY OPERATIONS */
/*		  (HIGH LEVEL)		  */
/*----------------------------*/
#pragma once
#ifndef _HOSTMEM_
#define _HOSTMEM_


// Class wrapper for memory allocations
template < typename T>
class HostMemory {

	private:
		// Private members
		std::vector< T* >	array;
		unsigned int*		rand_buffer;

	public:

		// Constructor/Destructor
		HostMemory(FileManager&);
		~HostMemory();

		// Public Methods
		T*		get(unsigned int);
		unsigned
		int*	get_RND_GNRT_buffer();

		// Public Types
		enum {
			X_ARRAY = 0,
			D_ARRAY,			
			MAX_ELEMENTS = D_ARRAY
		};

};

// Include the implementation because 
// separate templated methods won't link
#include "HostMemory.cu"

#endif // !_HOSTMEM_