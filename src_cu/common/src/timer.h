#pragma once

#include <chrono>

#define USE_CUDA_TIMING 0

class Timer
{
#if USE_CUDA_TIMING
private:
    cudaEvent_t start, end;

public:
    Timer(bool ticking = true)
    {
        cudaEventCreate(&start);
        cudaEventCreate(&end);

        if (ticking)
        {
            tick();
        }
    }

    ~Timer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }

    void tick()
    {
        cudaEventRecord(start);
    }

    void tock()
    {
        cudaEventRecord(end);
    }

    float duration() const
    {
        cudaEventSynchronize(end);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, end);
        return elapsedTime * 1e3f;
    }
#else
private:
    std::chrono::high_resolution_clock::time_point begin;
    std::chrono::high_resolution_clock::time_point end;

public:
    Timer(bool ticking = true)
    {
        if (ticking)
        {
            tick();
        }
    }

    ~Timer()
    {
        tock();
    }

    void tick()
    {
        begin = std::chrono::high_resolution_clock::now();
    }

    void tock()
    {
        end = std::chrono::high_resolution_clock::now();
    }

    float duration() const
    {
        auto dur =
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
        return dur.count();
    }
#endif
};
