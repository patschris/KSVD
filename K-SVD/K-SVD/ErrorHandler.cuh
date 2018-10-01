/*------------------------------*/
/*   ERROR HANDLING NAMESPACE   */
/*------------------------------*/
#pragma once
#ifndef _ERROR_
#define _ERROR_


// Functions used for error handling
namespace ErrorHandler {

	// Public Methods
	const char* cublasGetErrorString(cublasStatus_t);

};

#endif // !_ERROR_