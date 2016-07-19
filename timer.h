#ifndef timer_h
#define timer_h

#include <chrono>

template <bool cuda>
class Timer
{
  public:
    Timer() : start_time(std::chrono::high_resolution_clock::now()) {}

    // start or restart timer
    void tick()
    {
#ifdef __CUDACC__
      if(cuda)
        cudaDeviceSynchronize();
#endif
      start_time = std::chrono::high_resolution_clock::now();
    }

    // return time since last start
    double tock()
    {
#ifdef __CUDACC__
      if(cuda)
        cudaDeviceSynchronize();
#endif
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = end - start_time;
      return diff.count();
    }

  private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
};

#endif
