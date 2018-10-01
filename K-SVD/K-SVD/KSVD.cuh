/*--------------------------------*/
/*          KSVD CLASS            */
/*--------------------------------*/
#pragma once
#ifndef _KSVD_
#define _KSVD_


// Class used for executing the
// K-SVD algorithm
class KSVD : public OMP, public DictionaryUpdate {

	private:
		// Private Functions
		bool	normcols();
		bool	errorComputation(unsigned int);
		bool	clearDict();

	public:
		// Constructor
		KSVD(DeviceMemory<datatype>*, HostMemory<datatype>*);
		// Destructor
		~KSVD();

		// Public Methods
		bool	ExecuteIterations(unsigned int);
		bool	isValid();

};

#endif // !_KSVD_