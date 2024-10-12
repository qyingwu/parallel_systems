#ifndef ATOMIC_UTILS_H
#define ATOMIC_UTILS_H

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
// atomicAdd for double only if CUDA architecture is below 6.0
//https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

#endif // ATOMIC_UTILS_H