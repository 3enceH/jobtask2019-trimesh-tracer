#include "compute.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cctype>

ComputeEnv::ComputeEnv()
{
    cl_int status(CL_SUCCESS);
    check(cl::Platform::get(&m_platforms));
    for (int i = 0; i < m_platforms.size(); i++)
    {
        std::string name = m_platforms[i].getInfo<CL_PLATFORM_NAME>(&status);
        std::string version = m_platforms[i].getInfo<CL_PLATFORM_VERSION>(&status);
        std::cout << "Platform #" << i << " | " << name << " | "  << version << std::endl;
        check(status);

        cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(m_platforms[i])(), 0 };
        cl::Context ctx(CL_DEVICE_TYPE_GPU, properties, 0, 0, &status);
        check(status);

        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::tolower(c); });
        if (name.find("nvidia") != name.npos || name.find("amd") != name.npos) //restriction
        {
            m_contexts.push_back(ctx);
        }
    }
    for (auto& ctx : m_contexts)
    {
        std::vector<cl::Device> devices = ctx.getInfo<CL_CONTEXT_DEVICES>(&status);
        check(status);
        for (auto& dev : devices)
        {
            m_devices.push_back(dev);
            m_queues.emplace_back(ctx, dev, 0, &status);
            check(status);
        }
    }
    for (int i = 0; i < m_devices.size(); i++)
    {
        std::cout << "Device #" << i << " | "
            << m_devices[i].getInfo<CL_DEVICE_NAME>(&status) << "| "
            << m_devices[i].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>(&status) / (1024e2) << "MB" << std::endl;
        check(status);
    }
}

void ComputeEnv::loadProgram(const std::string& path, std::vector<std::string> includeDirs, std::vector<std::string> definitions)
{
    cl_int status(CL_SUCCESS);
    std::ifstream file(path);
    std::string source((std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());
    cl::Program program(source, &status);
    check(status);
    m_programs[path] = program;
    std::stringstream options;
    for (auto& dir : includeDirs)
    {
        options << "-I" << dir << " ";
    }
    for (auto& def : definitions)
    {
        options << "-D" << def << " ";
    }
    status = program.build(options.str().c_str());
    if (status == CL_BUILD_PROGRAM_FAILURE)
    {
        cl_build_status build = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(currentDevice(), &status);
        check(status);
        if (build != CL_BUILD_SUCCESS)
        {
            std::cout << build << std::endl;
            std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(currentDevice(), &status);
            check(status);
            std::cout << log << std::endl;
        }
    }
    else
        check(status);

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
    cl_int status(CL_SUCCESS);
    m_deviceBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, m_size, &status);
    check(status);
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
    check(queue.enqueueWriteBuffer(m_deviceBuffer, cl_bool(false), offset, size, m_hostPtr));
}
void CBuffer::download(const cl::CommandQueue& queue, size_t offset/* = 0*/, size_t size/* = 0*/)
{
    if (size == 0) size = m_size;
    check(queue.enqueueReadBuffer(m_deviceBuffer, cl_bool(false), offset, size, m_hostPtr));
}
