/*-------------------------*/
/*   UTILITIES NAMESPACE   */
/*-------------------------*/
#pragma once
#ifndef _UTIL_
#define _UTIL_


// Functions used for various
// secondary tasks
namespace Utilities {

	// Public Methods
	void print(datatype*, unsigned int, unsigned int, unsigned int, unsigned int);
	void printTransposed(datatype*,unsigned int,unsigned int,unsigned int,unsigned int);
	void printDevice(datatype*, unsigned int, unsigned int, unsigned int, unsigned int);
	void printDeviceTransposed(datatype*, unsigned int, unsigned int, unsigned int, unsigned int);

};

#endif // !_UTIL_