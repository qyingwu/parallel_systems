#include <atomic>
#include <thread>
#include <stdexcept>
#include <chrono>

/************************
 * spin_barrier class    *
 ************************/

class spin_barrier {
private:
    std::atomic<int> counter;  // Shared counter for fetch-and-increment
    std::atomic<int> go_state;    // Keeps track of barrier state (0 and 1) for arrival and release.
    int total;                 // Total number of threads

public:
    spin_barrier(int num_threads) : counter(0), go_state(0), total(num_threads) {
        if (num_threads <= 0) {
            throw std::invalid_argument("Number of threads must be greater than 0");
        }
    }

    // Wait function based on a atomic counter barrier 
    void wait() {
        int local_go_state = go_state.load(std::memory_order_relaxed);  // Step 1: current state

        // Step 2: fetch-and-increment the counter to signal arrival
        //memory_order_acquire making sure all memory changes are visible to the thread before incrementing.
        int local_counter = counter.fetch_add(1, std::memory_order_acquire);

        // Step 3: If the current thread is the last to arrive
        if (local_counter + 1 == total) {
            // Step 4: Reset the counter and toggle the state
            //no enforcing any synchronization
            counter.store(0, std::memory_order_relaxed);
            go_state.fetch_add(1, std::memory_order_release); 
        } else {
            // Step 5: Sleep while waiting for the state to change
            while (go_state.load(std::memory_order_acquire) == local_go_state) {
                std::this_thread::sleep_for(std::chrono::microseconds(10)); // Sleep for 10 microseconds
            }
        }
    }
};

// Global spin barrier
spin_barrier* global_barrier = nullptr;

extern "C" {
    // Initializes the global spin barrier
    void init_barrier(int num_threads) {
        if (num_threads <= 0) {
            throw std::invalid_argument("Number of threads must be greater than 0");
        }
        global_barrier = new spin_barrier(num_threads);
    }

    // Wait at the global barrier
    void barrier_wait() {
        if (!global_barrier) {
            throw std::runtime_error("Barrier not initialized.");
        }
        global_barrier->wait();
    }

    // Destroys the global spin barrier
    void destroy_barrier() {
        if (!global_barrier) {
            throw std::runtime_error("Barrier already destroyed or not initialized.");
        }
        delete global_barrier;
        global_barrier = nullptr;
    }
}
