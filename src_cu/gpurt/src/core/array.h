#pragma once

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <vector>

namespace rtx
{
template <typename T>
class device_vector;

template <typename T>
class host_vector : public std::vector<T>
{
public:
    host_vector<T>()
    {
    }
    host_vector<T>(size_t l)
    {
        this->resize(l);
    }
    host_vector<T>(const host_vector<T>& in)
    {
        this->operator=(in);
    }
    host_vector<T>(const device_vector<T>& in);
    host_vector<T>& operator=(const device_vector<T>& in);

    T* ptr()
    {
        return &(*this)[0];
    }
    const T* ptr() const
    {
        return &(*this)[0];
    }
};

template <typename T>
class device_vector
{
    T*     _data = nullptr;
    size_t _len  = 0;
    void   _safe_free()
    {
        if (_data)
        {
            checkCudaErrors(cudaFree(_data));
            _data = nullptr;
            _len  = 0;
        }
    }

public:
    device_vector<T>()
    {
    }
    device_vector<T>(const host_vector<T>& in)
    {
        (*this) = in;
    }
    device_vector<T>& operator=(const host_vector<T>& in)
    {
        if (in.empty()) return *this;
        resize(in.size());
        checkCudaErrors(cudaMemcpy(
            _data, in.ptr(), sizeof(T) * _len, cudaMemcpyHostToDevice));
        return *this;
    }
    T* ptr()
    {
        return _data;
    }
    const T* ptr() const
    {
        return _data;
    }
    size_t size() const
    {
        return _len;
    }
    void resize(size_t new_size)
    {
        if (new_size != _len)
        {
            _safe_free();
            if (new_size > 0)
            {
                checkCudaErrors(cudaMalloc((void**)&_data, sizeof(T) * new_size));
            }
            _len = new_size;
        }
    }
    ~device_vector()
    {
        _safe_free();
    }
};

template <typename T>
host_vector<T>& host_vector<T>::operator=(const device_vector<T>& in)
{
    this->resize(in.size());
    checkCudaErrors(cudaMemcpy(this->ptr(),
                               in.ptr(),
                               sizeof(T) * this->size(),
                               cudaMemcpyDeviceToHost));
    return *this;
}

template <typename T>
host_vector<T>::host_vector(const device_vector<T>& in)
{
    this->operator=(in);
}
}  // namespace rtx

#define RAW(x) x.ptr()
#define hvec rtx::host_vector
#define dvec rtx::device_vector

#define CHECK_CUDA_ERRORS(call)                                   \
    {                                                             \
        cudaError err = call;                                     \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr,                                       \
                    "Cuda error in file '%s' in line %i : %s.\n", \
                    __FILE__,                                     \
                    __LINE__,                                     \
                    cudaGetErrorString(err));                     \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    }
