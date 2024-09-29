#ifndef _SPIN_BARRIER_H
#define _SPIN_BARRIER_H

#include <atomic>
#include <pthread.h>
#include <iostream>

class spin_barrier {
private:
    std::atomic<int> count;
    std::atomic<int> waiting;
    int total;

public:
    spin_barrier(int num_threads);
    void wait();
};

#ifdef __cplusplus
extern "C" {
#endif

    void init_barrier(int num_threads);
    void barrier_wait();
    void destroy_barrier();

#ifdef __cplusplus
}
#endif

#endif 
