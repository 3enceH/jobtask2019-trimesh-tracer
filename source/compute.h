#pragma once

#define __CL_ENABLE_EXCEPTIONS
#define CL_TARGET_OPENCL_VERSION 120
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <iostream>

class ComputeEnv
{
public:
    ComputeEnv(const ComputeEnv&) = delete;
    ComputeEnv(ComputeEnv&&) = delete;
    ComputeEnv();

    cl::Context& currentContext() { return m_contexts[0]; }
    cl::Device& currentDevice() { return m_devices[0]; }
    cl::CommandQueue& currentQueue() { return m_queues[0]; }
private:
    std::vector<cl::Platform> m_platforms;
    std::vector<cl::Context> m_contexts;
    std::vector<cl::Device> m_devices;
    std::vector<cl::CommandQueue> m_queues;
};

class CBuffer
{
public:
    CBuffer();
    //wraps host data if not nullptr, else allocs
    CBuffer(const cl::Context& context, size_t size, void* hostPtr = nullptr);
    CBuffer(CBuffer&& other);
    CBuffer(const CBuffer& other);
    ~CBuffer();

    operator cl::Buffer&() { return m_deviceBuffer; }
    operator void*() { return m_hostPtr; }

    void upload(const cl::CommandQueue& queue, size_t offset = 0, size_t size = 0);
    void download(const cl::CommandQueue& queue, size_t offset = 0, size_t size = 0);
private:
    bool m_owns;
    void* m_hostPtr;
    size_t m_size;
    cl::Buffer m_deviceBuffer;
};

