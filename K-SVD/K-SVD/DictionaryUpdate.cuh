/*-----------------------------------------------*/
/*  k-MEANS DICTIONARY UPDATE STAGE BASE CLASS   */
/*-----------------------------------------------*/
#pragma once
#ifndef _DICUPD_
#define _DICUPD_


// Base class implementing the
// Dictionary Update Stage
class DictionaryUpdate : public virtual CudaAlgorithm {

	private:
		// Private Functions
		bool	Sprow();
		bool	specialCase(unsigned int);

	protected:		
		// Default Constructor
		DictionaryUpdate();

		// Protected Members
		bool	updateDictionary(unsigned int*);

};

#endif // !_DICUPD_