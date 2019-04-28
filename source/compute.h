#pragma once

#define CL_TARGET_OPENCL_VERSION 120
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <map>
#include <sstream>

#define ERROR_CASE(code) case code: return #code;

inline const char* getErrorString(cl_int status)
{
    switch (status)
    {
        ERROR_CASE(CL_SUCCESS                                  )
        ERROR_CASE(CL_DEVICE_NOT_FOUND                         )
        ERROR_CASE(CL_DEVICE_NOT_AVAILABLE                     )
        ERROR_CASE(CL_COMPILER_NOT_AVAILABLE                   )
        ERROR_CASE(CL_MEM_OBJECT_ALLOCATION_FAILURE            )
        ERROR_CASE(CL_OUT_OF_RESOURCES                         )
        ERROR_CASE(CL_OUT_OF_HOST_MEMORY                       )
        ERROR_CASE(CL_PROFILING_INFO_NOT_AVAILABLE             )
        ERROR_CASE(CL_MEM_COPY_OVERLAP                         )
        ERROR_CASE(CL_IMAGE_FORMAT_MISMATCH                    )
        ERROR_CASE(CL_IMAGE_FORMAT_NOT_SUPPORTED               )
        ERROR_CASE(CL_BUILD_PROGRAM_FAILURE                    )
        ERROR_CASE(CL_MAP_FAILURE                              )
#ifdef CL_VERSION_1_1                                  
        ERROR_CASE(CL_MISALIGNED_SUB_BUFFER_OFFSET             )
        ERROR_CASE(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
#endif                                                 
#ifdef CL_VERSION_1_2                                  
        ERROR_CASE(CL_COMPILE_PROGRAM_FAILURE                  )
        ERROR_CASE(CL_LINKER_NOT_AVAILABLE                     )
        ERROR_CASE(CL_LINK_PROGRAM_FAILURE                     )
        ERROR_CASE(CL_DEVICE_PARTITION_FAILED                  )
        ERROR_CASE(CL_KERNEL_ARG_INFO_NOT_AVAILABLE            )
#endif                                                 
                                                       
        ERROR_CASE(CL_INVALID_VALUE                            )
        ERROR_CASE(CL_INVALID_DEVICE_TYPE                      )
        ERROR_CASE(CL_INVALID_PLATFORM                         )
        ERROR_CASE(CL_INVALID_DEVICE                           )
        ERROR_CASE(CL_INVALID_CONTEXT                          )
        ERROR_CASE(CL_INVALID_QUEUE_PROPERTIES                 )
        ERROR_CASE(CL_INVALID_COMMAND_QUEUE                    )
        ERROR_CASE(CL_INVALID_HOST_PTR                         )
        ERROR_CASE(CL_INVALID_MEM_OBJECT                       )
        ERROR_CASE(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR          )
        ERROR_CASE(CL_INVALID_IMAGE_SIZE                       )
        ERROR_CASE(CL_INVALID_SAMPLER                          )
        ERROR_CASE(CL_INVALID_BINARY                           )
        ERROR_CASE(CL_INVALID_BUILD_OPTIONS                    )
        ERROR_CASE(CL_INVALID_PROGRAM                          )
        ERROR_CASE(CL_INVALID_PROGRAM_EXECUTABLE               )
        ERROR_CASE(CL_INVALID_KERNEL_NAME                      )
        ERROR_CASE(CL_INVALID_KERNEL_DEFINITION                )
        ERROR_CASE(CL_INVALID_KERNEL                           )
        ERROR_CASE(CL_INVALID_ARG_INDEX                        )
        ERROR_CASE(CL_INVALID_ARG_VALUE                        )
        ERROR_CASE(CL_INVALID_ARG_SIZE                         )
        ERROR_CASE(CL_INVALID_KERNEL_ARGS                      )
        ERROR_CASE(CL_INVALID_WORK_DIMENSION                   )
        ERROR_CASE(CL_INVALID_WORK_GROUP_SIZE                  )
        ERROR_CASE(CL_INVALID_WORK_ITEM_SIZE                   )
        ERROR_CASE(CL_INVALID_GLOBAL_OFFSET                    )
        ERROR_CASE(CL_INVALID_EVENT_WAIT_LIST                  )
        ERROR_CASE(CL_INVALID_EVENT                            )
        ERROR_CASE(CL_INVALID_OPERATION                        )
        ERROR_CASE(CL_INVALID_GL_OBJECT                        )
        ERROR_CASE(CL_INVALID_BUFFER_SIZE                      )
        ERROR_CASE(CL_INVALID_MIP_LEVEL                        )
        ERROR_CASE(CL_INVALID_GLOBAL_WORK_SIZE                 )
#ifdef CL_VERSION_1_1                                  
        ERROR_CASE(CL_INVALID_PROPERTY                         )
#endif                                                 
#ifdef CL_VERSION_1_2                                  
        ERROR_CASE(CL_INVALID_IMAGE_DESCRIPTOR                 )
        ERROR_CASE(CL_INVALID_COMPILER_OPTIONS                 )
        ERROR_CASE(CL_INVALID_LINKER_OPTIONS                   )
        ERROR_CASE(CL_INVALID_DEVICE_PARTITION_COUNT           )
#endif                                                 
#ifdef CL_VERSION_2_0                                  
        ERROR_CASE(CL_INVALID_PIPE_SIZE                        )
        ERROR_CASE(CL_INVALID_DEVICE_QUEUE                     )
#endif                                                 
#ifdef CL_VERSION_2_2                                  
        ERROR_CASE(CL_INVALID_SPEC_ID                          )
        ERROR_CASE(CL_MAX_SIZE_RESTRICTION_EXCEEDED            )
#endif
    default:
        return "Unknown";
    }
}
#define check(status) if((status) != CL_SUCCESS) \
{ \
    std::stringstream ss; \
    ss << getErrorString((status)) << " file " << __FILE__ << " line " << __LINE__; \
    throw std::runtime_error(ss.str()); \
}

class ComputeEnv
{
public:
    ComputeEnv(const ComputeEnv&) = delete;
    ComputeEnv(ComputeEnv&&) = delete;
    ComputeEnv();

    void loadProgram(const std::string& path, std::vector<std::string> includeDirs = std::vector<std::string>(), std::vector<std::string> definitions = std::vector<std::string>());

    cl::Context& currentContext() { return m_contexts[0]; }
    cl::Device& currentDevice() { return m_devices[0]; }
    cl::CommandQueue& currentQueue() { return m_queues[0]; }
private:
    std::vector<cl::Platform> m_platforms;
    std::vector<cl::Context> m_contexts;
    std::vector<cl::Device> m_devices;
    std::vector<cl::CommandQueue> m_queues;
    std::map<std::string, cl::Program> m_programs;
    std::vector<cl::Kernel> m_kernels;
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

