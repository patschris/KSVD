________________
 * k-SVD Notes *
________________

1. ~4.220 lines of code
2. Architecture from Kepler and above is supported.
3. CUDA architecture 3.0 or above is required because of Kepler's
	_shfl_down intrinsics for reduction.
4. Fermi architecture (compute=20,sm_21) is NOT supported.