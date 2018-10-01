/*------------------------------*/
/*   ALGORITHMIC TOOLS CLASS    */
/*      (REDUCTIONS etc.)       */
/*------------------------------*/
#pragma once
#ifndef _ALGOS_
#define _ALGOS_


// Collection of all abstract algorithmic
// tools we'll need in the GPU algorithms
class Algorithms {

	public:
		// Family of all reduction operations
		// based on optimized warp reductions
		class Reduction {
		
			public:
				// Reduction in 1D layout (i.e. vector) of
				// __M__A__X__I__M__U__M__ value of its
				// elements as well as its index.
				__host__ static
				bool Maximum(datatype*, int, datatype*);

				// Reduction in 1D layout (i.e. vector)
				// of __M__A__X__I__M__U__M__ squared value.
				__host__ static
				bool Maximum_squared(datatype*, unsigned int, datatype*, datatype*, unsigned int);

				// Reduction in 2D layout (i.e. matrix columns)
				// of __S__U__M__  of squared values of elements.
				// Result is then square rooted and inverted.
				__host__ static
				bool reduce2D_Sum_SQUAREDelements(datatype*, int, int, datatype*, unsigned int);

				// Reduction in 1D layout (i.e. vector)
				// of __S__U__M__ of squared values of elements
				// (Mutliblock).
				__host__ static
				bool reduce1D_Batched_Sum_TOGETHER_collincomb(datatype*, datatype*, datatype*, int, int, datatype*,
																datatype*, datatype*, unsigned int);

				// Reduction in 2D layout (i.e. matrix rows)
				// of dot products with given matrix
				// (filtered by the specified cols)
				__host__ static
				bool reduce2D_dot_products_modified(datatype*, datatype*, datatype*, datatype*, datatype*,
													unsigned int, unsigned int, unsigned int);

				// Euclidean norm of the vector.
				__host__ static
				bool euclidean_norm(datatype*, datatype*, unsigned int, unsigned int);

				// Reduction in 2D layout (i.e. matrix columns) 
				// of linear combination of input matrix and  
				// given columns (filtered by the specified rows)
				__host__ static
				bool reduce2D_rowlincomb_plus_nrm2_plus_mul(datatype*, datatype*, datatype*, datatype*, unsigned int,
													unsigned int, datatype*, unsigned int, datatype*, datatype*,
													datatype*, unsigned int);

				// Compute the Round Mean Square Error between a
				// matrix X and its approximation matrix X~
				__host__ static
				bool reduce_RMSE(datatype*, datatype*, datatype*, unsigned int, unsigned int, unsigned int,
									unsigned int, datatype*);

				// Compute the Round Mean Square Error between a
				// matrix X and its approximation matrix X~
				// ( modified for use only in Dictionary Update's
				// special case )
				__host__ static
				bool reduce_ERROR_for_special_case(datatype*, datatype*, datatype*, unsigned int, unsigned int,
													datatype*, datatype*, datatype*, unsigned int);

				// Euclidean norm of the vector
				// ( used only in the special case)
				__host__ static
				bool SCase_EU_norm(datatype*, datatype*, unsigned int, unsigned int, datatype*, datatype*, unsigned int);
					
				// Set the size of the reduction buffer 
				// to be used in the calculations
				static bool setReductionBuffer(datatype*);

			private:
				Reduction(){}

		};

		// Family of all matrix-matrix and 
		// vector-matrix multiplication variants
		class Multiplication {

			public:
				// Matrix-Matrix Multiplication
				// using cublas<T>GEMM
				// Operation format ==> C = A'*B
				static bool AT_times_B(cublasHandle_t, datatype*, int, int, datatype*, int, int, datatype*);

				// Matrix-Matrix Multiplication
				// using cublas<T>GEMM
				// Operation format ==> C = A*B
				static bool A_times_B(cublasHandle_t, datatype*, int, int, datatype*, int, int, datatype*);

				// Matrix-Vector Multiplication
				// Operation format ==> C -= A*x
				static bool A_times_x_SUBTRACT(datatype*, int, datatype*, int, datatype*, datatype*, datatype*,
												datatype*, unsigned int);

				// Matrix-Vector Multiplication
				// Operation format ==> C = AT*x
				static bool AT_times_x(cublasHandle_t, datatype*, int, datatype*, int, datatype*);

				// Matrix-Vector Multiplication 
				// Operation format ==> C = x*A 
				static bool normal_x_times_A_modified(datatype*, datatype*, datatype*, unsigned int, datatype*,
														datatype*, datatype*, datatype*, unsigned int, datatype*,
														unsigned int, datatype*, datatype*, unsigned int);

			private:
				Multiplication() {}

		};

		// Family of all random permutation
		// generators' algorithmic utilities
		class RandomPermutation {

			public:
				// Generate a random permutation
				// of indices in the specified range.
				static unsigned int* generateRandPerm(unsigned int);

				// Set the size of the generator buffer 
				// to be used in the calculations
				static bool setGeneratorBuffer(unsigned int*, int);

			private:
				RandomPermutation() {}
				
		};

	private:
		Algorithms(){}

};

#endif // !_ALGOS_