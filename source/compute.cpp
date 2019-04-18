#include "compute.h"

ComputeEnv::ComputeEnv()
{
    cl::Platform::get(&m_platforms);
    for (int i = 0; i < m_platforms.size(); i++)
    {
        std::cout << "Platform #" << i << " | "
            << m_platforms[i].getInfo<CL_PLATFORM_NAME>() << "| "
            << m_platforms[i].getInfo<CL_PLATFORM_VERSION>() << std::endl;

        cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(m_platforms[i])(), 0 };
        m_contexts.emplace_back(CL_DEVICE_TYPE_GPU, properties);
    }
    for (auto& ctx : m_contexts)
    {
        std::vector<cl::Device> devices = ctx.getInfo<CL_CONTEXT_DEVICES>();
        for (auto& dev : devices)
        {
            m_devices.push_back(dev);
            m_queues.emplace_back(ctx, dev);
        }
    }
    for (int i = 0; i < m_devices.size(); i++)
    {
        std::cout << "Device #" << i << " | "
            << m_devices[i].getInfo<CL_DEVICE_NAME>() << "| "
            << m_devices[i].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024e2) << "MB" << std::endl;
    }
}

CBuffer::CBuffer()
    : m_size(0)
    , m_owns(false)
    , m_hostPtr(nullptr)
{
}

CBuffer::CBuffer(const cl::Context& context, size_t size, void* hostPtr/* = nullptr*/)
    : m_size(size)
{
    if (hostPtr)
    {
        m_owns = false;
        m_hostPtr = hostPtr;
    }
    else
    {
        m_owns = true;
        m_hostPtr = new uint8_t[m_size];
    }
    m_deviceBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, m_size);
}

CBuffer::CBuffer(CBuffer&& other)
    : m_size(other.m_size)
    , m_deviceBuffer(other.m_deviceBuffer)
    , m_hostPtr(other.m_hostPtr)
{
    std::cout << "CBuffer::move" << std::endl;
    m_owns = other.m_owns ? true : false;
    other.m_owns = false;
}

CBuffer::CBuffer(const CBuffer& other)
    : m_size(other.m_size)
    , m_deviceBuffer(other.m_deviceBuffer)
    , m_hostPtr(other.m_hostPtr)
{
    std::cout << "CBuffer::copy" << std::endl;
    if (other.m_owns)
    {
        m_hostPtr = new uint8_t[m_size];
        memcpy(m_hostPtr, other.m_hostPtr, m_size);
    }
    m_owns = other.m_owns;
}

CBuffer::~CBuffer()
{
    if (m_owns && m_hostPtr)
    {
        delete[] m_hostPtr;
        m_hostPtr = nullptr;
    }
}

void CBuffer::upload(const cl::CommandQueue& queue, size_t offset/* = 0*/, size_t size/* = 0*/)
{
    if (size == 0) size = m_size;
    queue.enqueueWriteBuffer(m_deviceBuffer, cl_bool(false), offset, size, m_hostPtr);
}
void CBuffer::download(const cl::CommandQueue& queue, size_t offset/* = 0*/, size_t size/* = 0*/)
{
    if (size == 0) size = m_size;
    queue.enqueueReadBuffer(m_deviceBuffer, cl_bool(false), offset, size, m_hostPtr);
}
