/*------------------------------*/
/*  THRUST OPERATORS NAMESPACE  */
/*       IMPLEMENTATION         */
/*------------------------------*/

/*==========================================================================*/
/*================================= BEGIN ==================================*/
/*==========================================================================*/

/*--------------------------------*/
/* Unary function for Thrust that */
/* calculates the exponential of  */
/* each element.                  */
/*--------------------------------*/
template < typename T >
struct ThrustOperators::Exp : public thrust::unary_function<T, T>
{
	__host__ __device__ T operator()(T x)
	{
		return x*x;
	}
};

/*--------------------------------*/
/* Unary function for Thrust that */
/* calculates the inverse of the  */
/* square root of each element.   */
/*--------------------------------*/
template < typename T >
struct ThrustOperators::Inv : public thrust::unary_function<T, T>
{
	__host__ __device__ T operator()(T x)
	{
		return (T) 1.0 / sqrt(x);
	}
};

/*==========================================================================*/
/*=================================  END  ==================================*/
/*==========================================================================*/