/*--------------------------------*/
/* FILE OPERATIONS RELATED CLASS  */
/*--------------------------------*/
#pragma once
#ifndef _FILEMNGR_
#define _FILEMNGR_


// MatLab Libraries for MAT-files and
// Mx matrices
#include INCLUDE_FILE(MATLAB_include,mat.h)
#include INCLUDE_FILE(MATLAB_include,matrix.h)

// Functions used for reading from/writing to
// specific files
class FileManager {

	private:
		// Private Members
		mxArray *aX, *aD;

	public:
		// Constructor
		FileManager();
		// Destructor
		~FileManager();

		// Public Methods
		bool	readXarray(datatype**);
		bool	readDarray(datatype**);
		bool	writeDarray(datatype*);
		bool	writeTemporary(datatype*,unsigned int,unsigned int,char*);

};

#endif // !_FILEMNGR_