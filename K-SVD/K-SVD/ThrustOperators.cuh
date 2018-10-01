/*------------------------------*/
/*  THRUST OPERATORS NAMESPACE  */
/*------------------------------*/
#pragma once
#ifndef _THR_
#define _THR_


// Methods used as unary functions
// in Thrust family calls
namespace ThrustOperators {

	/*----------------*/
	/* Public Methods */
	/*----------------*/
	// Unary functions
	template < typename T >
	struct Exp;
	template < typename T >
	struct Inv;

};

// Include the implementation because 
// separate templated methods won't link
#include "ThrustOperators.cu"

#endif // !_THR_