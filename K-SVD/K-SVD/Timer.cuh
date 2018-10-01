/*--------------------------*/
/* TIMING OPERATIONS CLASS  */
/*--------------------------*/
#pragma once
#ifndef _TIMER_
#define _TIMER_


// Functions used for calculating
// elapsed time on CPU or GPU
class Timer {

	private:
		// Private Members
		cudaEvent_t GPUstart, GPUstop;
		LARGE_INTEGER CPUstart, CPUstop, frequency;

	public:
		// Constructor
		Timer();
		~Timer();

		// Public Methods
		void	tic();
		double	toc();
		double	tocmilli();
		void	GPUtic();
		float	GPUtoc();
		bool	isValid();

};

#endif // !_TIMER_