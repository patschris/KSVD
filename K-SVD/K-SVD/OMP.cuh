/*----------------------------------------*/
/*  BATCH-OMP CUDA ALGORITHM BASE CLASS   */
/*----------------------------------------*/
#pragma once
#ifndef _BOMP_
#define _BOMP_


// Base class implementing the
// Batch-OMP (Orthogonal Matching Pursuit)
// algorithm
class OMP : public virtual CudaAlgorithm{

	private:
		// Private Functions
		bool	parallel_signals_operations();

	protected:
		// Default Constructor
		OMP();

		// Protected Members
		bool	BatchOMP();

};

#endif // !_BOMP_