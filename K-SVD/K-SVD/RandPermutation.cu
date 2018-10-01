/*----------------------------*/
/*   ALGORITHMIC TOOLS CLASS  */
/*   - RAND.PERM. SUBCLASS -  */
/*       IMPLEMENTATION       */
/*----------------------------*/
#include <iostream>
#include <algorithm>
#include "GlobalDeclarations.cuh"
#include "Algorithms.cuh"

using namespace std;


/*********************************
*         DECLARATIONS           *
*	      ____________           *
*********************************/
unsigned int* output_array = NULL;

/*********************************
*        CONSTANT FIELDS         *
*	     _______________         *
*********************************/
unsigned int static_values[9] = {
	1, 2, 6, 5, 0, 4, 7, 3, 8
};

/*************************************
*         MEMBER FUNCTIONS           *
*	      ________________           *
*************************************/
/*---------------------------------------*/
/* Generate a set of random permutations */
/*---------------------------------------*/
unsigned int* Algorithms::RandomPermutation::generateRandPerm(unsigned int perm_size) {
	#if STATIC_GENERATION
		return static_values;
	#else
		random_shuffle(output_array, output_array + perm_size);
		return output_array;
	#endif
}

/*--------------------------------------*/
/* Set the size of the generator buffer */
/* to be used in the calculations.      */
/*--------------------------------------*/
bool Algorithms::RandomPermutation::setGeneratorBuffer(unsigned int* b, int permsize) {
	output_array = b;
	// Initialize the given buffer space
	for (int j = 0; j < permsize; j++) {
		b[j] = j;
	}
	cudaDeviceSynchronize();
	return true;
}