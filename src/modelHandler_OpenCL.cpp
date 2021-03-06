#define CLLIB_EXTERN

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <libgen.h>
#include <sys/stat.h>
#endif
#include <vector>
#include <stdio.h>
#include <string>
#include <string.h>
#include "filters.hpp"
#include "sec.hpp"
#include "CLlib.h"
#include "params.h"

static const char prog[] = 
#include "modelHandler_OpenCL.cl.h"
	;

#define S_(a) #a
#define S(a) S_(a)

namespace w2xc {


#ifdef _WIN32
static HMODULE handle;
#else
static void *handle;
#endif

static int
cllib_init(void)
{
#ifdef _WIN32
	handle = LoadLibrary("OpenCL.dll");
#else
        handle = dlopen("libOpenCL.so.1", RTLD_LAZY);

#define GetProcAddress dlsym

#endif

        if (!handle) {
                return -1;
        }

#define LOAD(name)                              \
        p_##name = (__decltype(p_##name)) GetProcAddress(handle, #name); \
        if (p_##name == NULL) {                 \
                return -1;                      \
        }

        LOAD(clGetDeviceInfo);
        LOAD(clGetPlatformIDs);
        LOAD(clGetDeviceIDs);
        LOAD(clGetPlatformInfo);
        LOAD(clCreateProgramWithSource);
        LOAD(clCreateProgramWithBinary);
        LOAD(clBuildProgram);
        LOAD(clGetProgramBuildInfo);
        LOAD(clGetProgramInfo);
        LOAD(clReleaseProgram);
        LOAD(clCreateKernel);
        LOAD(clCreateBuffer);
        LOAD(clEnqueueWriteBuffer);
        LOAD(clFlush);
        LOAD(clReleaseMemObject);
        LOAD(clEnqueueReadBuffer);
        LOAD(clFinish);
        LOAD(clEnqueueNDRangeKernel);
        LOAD(clReleaseKernel);
        LOAD(clSetKernelArg);
        LOAD(clCreateCommandQueue);
        LOAD(clCreateContext);
        LOAD(clReleaseCommandQueue);
        LOAD(clReleaseContext);
        LOAD(clWaitForEvents);
        LOAD(clReleaseEvent);

        return 0;
}

bool
initOpenCL(ComputeEnv *env)
{
        int r = cllib_init();
        if (r < 0) {
                return false;
        }

        cl_uint num_plt;
        cl_platform_id plts[16];
        clGetPlatformIDs(16, plts, &num_plt);
        bool found = false;
        cl_int err;

        cl_platform_id platform;
        cl_context context;
        cl_device_id dev;
        cl_command_queue queue;
        cl_kernel ker_filter, ker_filter_in1_out32, ker_filter_in128_out1;
        cl_program program;

        for (unsigned int i=0; i<num_plt; i++) {
                size_t sz;
                cl_uint num_dev;

                clGetPlatformInfo(plts[i], CL_PLATFORM_NAME, 0, nullptr, &sz);
                std::vector<char> name(sz);
                clGetPlatformInfo(plts[i], CL_PLATFORM_NAME, sz, &name[0], &sz);

                bool is_amd = strstr(&name[0], "AMD") != NULL;
                //bool is_intel = strstr(&name[0], "Intel") != NULL;
                //bool is_nvidia = strstr(&name[0], "NVIDIA") != NULL;

                if (!is_amd) {
                        continue;
                }

                clGetDeviceIDs(plts[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_dev);
                if (num_dev == 0) {
                        continue;
                }

                std::vector<cl_device_id> devs(num_dev);
                clGetDeviceIDs(plts[i], CL_DEVICE_TYPE_GPU, num_dev, &devs[0], &num_dev);

                platform = plts[i];
                dev = devs[0];

                cl_context_properties props[] =
                        {CL_CONTEXT_PLATFORM, (cl_context_properties)(plts[i]), 0};
                cl_context ctxt = clCreateContext(props, 1, &devs[0], NULL, NULL, &err);
                if (err != CL_SUCCESS) {
                        continue;
                }

                context = ctxt;

                found = true;
                break;
        }

        if (!found) {
                return false;
        }

        size_t dev_name_len;
        clGetDeviceInfo(dev, CL_DEVICE_NAME, 0, nullptr, &dev_name_len);
        std::vector<char> dev_name(dev_name_len+1);
        clGetDeviceInfo(dev, CL_DEVICE_NAME, dev_name_len, &dev_name[0], &dev_name_len);

        printf("use GPU: %s\n",
               &dev_name[0]);

        bool bin_avaiable = false;

#if defined __linux || _WIN32

#ifdef __linux
        ssize_t path_len = 4;
        char *self_path = (char*)malloc(path_len+1);
        while (1) {
                ssize_t r = readlink("/proc/self/exe", self_path, path_len);
                if (r < path_len) {
                        self_path[r] = '\0';
                        break;
                }

                path_len *= 2;
                self_path = (char*)realloc(self_path, path_len+1);
        }

        struct stat self_st;
        stat(self_path, &self_st);
        self_path = dirname(self_path);
#else
        ssize_t path_len = 4;
        char *self_path = (char*)malloc(path_len+1);
	DWORD len;
        while (1) {
		len = GetModuleFileName(NULL, self_path, path_len);
		if (len > 0 && len != path_len) {
			break;
		}

                path_len *= 2;
                self_path = (char*)realloc(self_path, path_len+1);
        }
	WIN32_FIND_DATA self_st;
	HANDLE finder = FindFirstFile(self_path, &self_st);
	FindClose(finder);

	for (int si=len-1; si>=0; si--) {
		if (self_path[si] == '\\') {
			self_path[si] = '\0';
			break;
		}
	}
#endif

        std::string bin_path = std::string(self_path) + "/" + &dev_name[0] + ".bin";

        FILE *binfp = fopen(bin_path.c_str(), "rb");
        if (binfp) {
#ifdef __linux
                struct stat bin_st;
                stat(bin_path.c_str(), &bin_st);

                bool old = false;
                if (bin_st.st_mtim.tv_sec < self_st.st_mtim.tv_sec) {
                        old = true;
                }

                if (bin_st.st_mtim.tv_sec == self_st.st_mtim.tv_sec) {
                        if (bin_st.st_mtim.tv_nsec < self_st.st_mtim.tv_nsec) {
                                old = true;
                        }
                }
		size_t bin_sz = bin_st.st_size;
#else
                WIN32_FIND_DATA bin_st;
		HANDLE finder = FindFirstFile(bin_path.c_str(), &bin_st);
		FindClose(finder);

		bool old = false;
		uint64_t self_time = (((uint64_t)self_st.ftLastWriteTime.dwHighDateTime)<<32) |
			((uint64_t)self_st.ftLastWriteTime.dwLowDateTime);
		uint64_t bin_time = (((uint64_t)bin_st.ftLastWriteTime.dwHighDateTime)<<32) |
			((uint64_t)bin_st.ftLastWriteTime.dwLowDateTime);

		if (bin_time < self_time) {
			old = true;
		}

		size_t bin_sz = bin_st.nFileSizeLow;
#endif

                if (!old) {
                        unsigned char *bin = (unsigned char*)malloc(bin_sz);

                        size_t rem = bin_sz;
                        unsigned char *p = bin;
                        while (rem) {
                                size_t rsz = fread(p, 1, rem, binfp);
                                if (rsz <= 0) {
                                        break;
                                }

                                rem -= rsz;
                                p += rsz;
                        }

                        if (rem == 0) {
                                cl_int err;
                                program = clCreateProgramWithBinary(context, 1, &dev, &bin_sz,
                                                                    (const unsigned char**)&bin, NULL, &err);

                                if (err == CL_SUCCESS) {
                                        bin_avaiable = true;
                                }
                        }

                        free(bin);
                }

                fclose(binfp);
        }
#endif

        if (! bin_avaiable) {
                const char *source[1] = {prog};
                size_t src_len[1] = {sizeof(prog)-1};

                program = clCreateProgramWithSource(context, 1, source, src_len, &err);
                if (err != CL_SUCCESS) {
                        clReleaseContext(context);
                        return false;
                }

        }

#if defined __linux || defined _WIN32
        free(self_path);
#endif

        err = clBuildProgram(program, 1, &dev, "" , nullptr, nullptr);
        if (err != CL_SUCCESS) {
                size_t log_len;
                clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_len);

                std::vector<char> log(log_len+1);
                clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_len, &log[0], &log_len);
                log[log_len] = '\0';

                puts(&log[0]);

                clReleaseProgram(program);
                clReleaseContext(context);
                return false;
        }



#if defined __linux || _WIN32
        if (!bin_avaiable) {
                size_t binsz;
                size_t ret_len;
                clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(binsz), &binsz, &ret_len);

                char *buffer = new char [binsz];
                char *ptrs[1];
                ptrs[0] = buffer;

                clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(ptrs), ptrs, &ret_len);

                FILE *fp = fopen(bin_path.c_str(), "wb");

                size_t rem = binsz;
                char *p = buffer;

                while (rem) {
                        size_t wsz = fwrite(p, 1, rem, fp);
                        if (wsz <= 0) {
                                fclose(fp);
                                unlink(bin_path.c_str());
                                fp=NULL;
                                break;
                        }
                        rem -= wsz;
                        p += wsz;
                }

                if (fp) {
                        fclose(fp);
                }

                delete [] buffer;
        }
#endif



        ker_filter = clCreateKernel(program, "filter", &err);
        if (err != CL_SUCCESS) {
                clReleaseProgram(program);
                clReleaseContext(context);
                return false;
        }

        ker_filter_in1_out32 = clCreateKernel(program, "filter_in1_out32", &err);
        if (err != CL_SUCCESS) {
                clReleaseProgram(program);
                clReleaseContext(context);
                clReleaseKernel(ker_filter);
                return false;
        }

        ker_filter_in128_out1 = clCreateKernel(program, "filter_in128_out1", &err);
        if (err != CL_SUCCESS) {
                clReleaseProgram(program);
                clReleaseContext(context);
                clReleaseKernel(ker_filter);
                clReleaseKernel(ker_filter_in1_out32);
                return false;
        }

        queue = clCreateCommandQueue(context, dev, 0, &err);
        if (err != CL_SUCCESS) {
                clReleaseProgram(program);
                clReleaseContext(context);
                clReleaseKernel(ker_filter);
                clReleaseKernel(ker_filter_in1_out32);
                return false;
        }

        env->num_cl_dev = 1;
        env->cl_dev_list = new OpenCLDev[1];

        env->cl_dev_list[0].platform = platform;
        env->cl_dev_list[0].context = context;
        env->cl_dev_list[0].devid = dev;
        env->cl_dev_list[0].queue = queue;
        env->cl_dev_list[0].ker_filter = ker_filter;
        env->cl_dev_list[0].ker_filter_in1_out32 = ker_filter_in1_out32;
        env->cl_dev_list[0].ker_filter_in128_out1 = ker_filter_in128_out1;

        return true;
}


void
filter_OpenCL_impl(ComputeEnv *env,
                   Buffer *packed_input_buf,
                   Buffer *packed_output_buf,
                   int nInputPlanes,
                   int nOutputPlanes,
                   const float *fbiases,
                   const float *weight,
                   int w,
                   int h,
                   int nJob)
{
        cl_int err;

        OpenCLDev *dev = &env->cl_dev_list[0];
        cl_context context = dev->context;

        cl_mem cl_packed_input = packed_input_buf->get_read_ptr_cl(env, 0);
        cl_mem cl_packed_output = packed_output_buf->get_write_ptr_cl(env, 0);

        cl_mem cl_fbiases = clCreateBuffer(context,
                                           CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                           sizeof(float) * nOutputPlanes,
                                           (void*)fbiases, &err
                );
        enum filter_type {
                FILTER_GENERIC,
                FILTER_IN1,
                FILTER_OUT1
        } type = FILTER_GENERIC;

        cl_kernel ker = dev->ker_filter;

        if (nInputPlanes == 1 && nOutputPlanes == 32) {
                type = FILTER_IN1;
                ker = dev->ker_filter_in1_out32;
        } else if (nOutputPlanes == 1 && nInputPlanes == 128) {
                type = FILTER_OUT1;
                ker = dev->ker_filter_in128_out1;
        }


        size_t weight_size;

        if (type == FILTER_GENERIC) {
                weight_size = sizeof(float) * GPU_VEC_WIDTH * nInputPlanes * 9;
        } else {
                weight_size = sizeof(float) * nOutputPlanes * nInputPlanes * 9;
        }

        cl_mem cl_weight = clCreateBuffer(context,
                                          CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                          weight_size,
                                          (void*)weight, &err
                );

        int ai = 0;

        clSetKernelArg(ker, ai++, sizeof(cl_mem), &cl_packed_input);
        clSetKernelArg(ker, ai++, sizeof(cl_int), &nInputPlanes);
        clSetKernelArg(ker, ai++, sizeof(cl_mem), &cl_packed_output);
        clSetKernelArg(ker, ai++, sizeof(cl_int), &nOutputPlanes);
        clSetKernelArg(ker, ai++, sizeof(cl_mem), &cl_fbiases);
        clSetKernelArg(ker, ai++, sizeof(cl_int), &h);
        clSetKernelArg(ker, ai++, sizeof(cl_int), &w);
        clSetKernelArg(ker, ai++, sizeof(cl_mem), &cl_weight);

        size_t local_size = 0;
        //local_size += sizeof(float) * 256;
        //local_size += sizeof(float) * GPU_VEC_WIDTH;

        if (type == FILTER_GENERIC) {
                local_size += sizeof(float) * nInputPlanes * (GPU_BLOCK_SIZE+2) * 3;
                clSetKernelArg(ker, ai++, local_size, nullptr);
        }

        cl_event event;

        size_t gws[3] = {1, 1, 1};
        size_t lws[3] = {1, 1, 1};
        if (type == FILTER_GENERIC) {
                gws[0] = h * nOutputPlanes;
                lws[0] = nOutputPlanes;
        } else if (type == FILTER_IN1) {
                gws[0] = h * 256;
                lws[0] = 256;
        } else if (type == FILTER_OUT1) {
                gws[0] = h*128;
                lws[0] = 128;
        }

        err = clEnqueueNDRangeKernel(dev->queue,
                                     ker,
                                     3,
                                     nullptr, gws, lws,
                                     0, nullptr, &event);
        if (err != CL_SUCCESS) {
                printf("enqueue ndrange error : %d\n", err);
                exit(1);
        }

        err = clWaitForEvents(1, &event);
        if (err != CL_SUCCESS) {
                printf("wait ndrange error : %d\n", err);
                exit(1);
        }

        if (err != CL_SUCCESS) {
                printf("read buffer error : %d\n", err);
                exit(1);
        }

        clReleaseMemObject(cl_fbiases);
        clReleaseMemObject(cl_weight);
        clReleaseEvent(event);
}

}
