/**********
Copyright (c) 2019, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

#include "xcl2.hpp"
#include "scamp.h"
#include <algorithm>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>


using namespace std;

//Number of HBM Banks required
#define MAX_HBM_BANKCOUNT 30
#define BANK_NAME(n) n | XCL_MEM_TOPOLOGY
const int bank[MAX_HBM_BANKCOUNT] = {
    BANK_NAME(0),  BANK_NAME(1),  BANK_NAME(2),  BANK_NAME(3),  BANK_NAME(4),
    BANK_NAME(5),  BANK_NAME(6),  BANK_NAME(7),  BANK_NAME(8),  BANK_NAME(9),
    BANK_NAME(10), BANK_NAME(11), BANK_NAME(12), BANK_NAME(13), BANK_NAME(14),
    BANK_NAME(15), BANK_NAME(16), BANK_NAME(17), BANK_NAME(18), BANK_NAME(19),
    BANK_NAME(20), BANK_NAME(21), BANK_NAME(22), BANK_NAME(23), BANK_NAME(24),
    BANK_NAME(25), BANK_NAME(26), BANK_NAME(27), BANK_NAME(28), BANK_NAME(29)};

void preprocess(std::vector<DTYPE, aligned_allocator<DTYPE>> &tSeries,
				std::vector<DTYPE, aligned_allocator<DTYPE>> &means,
				std::vector<DTYPE, aligned_allocator<DTYPE>> &norms,
				std::vector<DTYPE, aligned_allocator<DTYPE>> &df,
				std::vector<DTYPE, aligned_allocator<DTYPE>> &dg,
				ITYPE profileLength, ITYPE windowSize)
{
	std::vector<DTYPE, aligned_allocator<DTYPE>> prefix_sum		(tSeries.size());
	std::vector<DTYPE, aligned_allocator<DTYPE>> prefix_sum_sq	(tSeries.size());

	  // Calculates prefix sum and square sum vectors
	  prefix_sum [0] = tSeries[0];
	  prefix_sum_sq [0] = tSeries[0] * tSeries[0];
	  for (ITYPE i = 1; i < tSeries.size(); ++i)
	  {
	    prefix_sum[i]    = tSeries[i] + prefix_sum[i - 1];
	    prefix_sum_sq[i] = tSeries[i] * tSeries[i] + prefix_sum_sq[i - 1];
	  }

	  // Prefix sum value is used to calculate mean value of a given window, taking last value
	  // of the window minus the first one and dividing by window size
	  means[0] = prefix_sum[windowSize - 1] / static_cast<DTYPE> (windowSize);
	  for (ITYPE i = 1; i < profileLength; ++i)
	  {
	    means[i] = (prefix_sum[i + windowSize - 1] - prefix_sum[i - 1]) / static_cast<DTYPE> (windowSize);
	  }

	  DTYPE sum = 0;
	  for (ITYPE i = 0; i < windowSize; ++i)
	  {
	    DTYPE val = tSeries[i] - means[0];
	    sum += val * val;
	  }
	  norms[0] = sum;

	  // Calculates L2-norms (euclidean norm, euclidean distance)
	  for (ITYPE i = 1; i < profileLength; ++i)
	  {
	    norms[i] = norms[i - 1] + ((tSeries[i - 1] - means[i - 1]) + (tSeries[i + windowSize - 1] - means[i])) *
	            (tSeries[i + windowSize - 1] - tSeries[i - 1]);
	  }
	  for (ITYPE i = 0; i < profileLength; ++i)
	  {
	    norms[i] = 1.0 / sqrt(norms[i]);
	  }

	  // Calculates df and dg vectors
	  for (ITYPE i = 0; i < profileLength - 1; ++i) {
	    df[i] = (tSeries[i + windowSize] - tSeries[i]) / 2.0;
	    dg[i] = (tSeries[i + windowSize] - means[i + 1]) + (tSeries[i] - means[i]);
	  }
}


void scamp_host(std::vector<DTYPE, aligned_allocator<DTYPE>> &tSeries,
				std::vector<ITYPE, aligned_allocator<ITYPE>> &diagonals,
				std::vector<DTYPE, aligned_allocator<DTYPE>> &means,
				std::vector<DTYPE, aligned_allocator<DTYPE>> &norms,
				std::vector<DTYPE, aligned_allocator<DTYPE>> &df,
				std::vector<DTYPE, aligned_allocator<DTYPE>> &dg,
				std::vector<DTYPE, aligned_allocator<DTYPE>> &profile,
				std::vector<ITYPE, aligned_allocator<ITYPE>> &profileIndex,
				ITYPE profileLength, ITYPE exclusionZone, ITYPE windowSize)
{

	    DTYPE covariance, correlation;

	    // Go through diagonals
	   for (ITYPE diag = exclusionZone + 1; diag < profileLength; diag++)
	    {
	      covariance = 0;
	      for (ITYPE i = 0; i < windowSize; i++)
	      {
	        covariance += ((tSeries[diag + i] - means[diag]) * (tSeries[i] - means[0]));
	      }

	      int i = 0;
	      int j = diag;

	      correlation = covariance * norms[i] * norms[j];

	      if (correlation > profile[i])
	      {
	        profile[i] = correlation;
	        profileIndex[i] = j;
	      }
	      if (correlation > profile[j])
	      {
	        profile[j] = correlation;
	        profileIndex[j] = i;
	      }

/*	      i = 1;

	      for (ITYPE j = diag + 1; j < profileLength; j++)
	      {
	        //covariance += (df[i - 1] * dg[j - 1] + df[j - 1] * dg[i - 1]);
	        correlation = covariance * norms[i] * norms[j];

	        if (correlation > profile[i])
	        {
	          profile[i] = correlation;
	          profileIndex[i] = j;
	        }

	        if (correlation > profile[j])
	        {
	          profile[j] = correlation;
	          profileIndex[j] = i;
	        }
	        i++;
	      }*/
	    }
}


// Function for verifying results
bool verify(std::vector<DTYPE, aligned_allocator<DTYPE>> &source_sw_profile,
            std::vector<DTYPE, aligned_allocator<DTYPE>> &source_hw_profile,
			std::vector<ITYPE, aligned_allocator<ITYPE>> &source_sw_profileIdxs,
			std::vector<ITYPE, aligned_allocator<ITYPE>> &source_hw_profileIdxs,
            unsigned int size) {
    bool check = true;
    for (size_t i = 0; i < size; i++) {
        if (source_hw_profile[i] != source_sw_profile[i]) {
            std::cout << "Error: Profile result mismatch" << std::endl;
            std::cout << "i = " << i << " CPU result = " << source_sw_profile[i]
                      << " Device result = " << source_hw_profile[i]
                      << std::endl;
            check = false;
            break;
        }
        /*if (source_hw_profileIdxs[i] != source_sw_profileIdxs[i]) {
            std::cout << "Error: Profile Idxs result mismatch" << std::endl;
            std::cout << "i = " << i << " CPU result = " << source_sw_profileIdxs[i]
                      << " Device result = " << source_hw_profileIdxs[i]
                      << std::endl;
            check = false;
            break;
        }*/
    }
    return check;
}

double run_krnl(cl::Context &context,
                cl::CommandQueue &q,
                cl::Kernel &kernel,
                std::vector<DTYPE, aligned_allocator<DTYPE>> &source_tSeries,
				std::vector<ITYPE, aligned_allocator<ITYPE>> &source_diagonals,
                std::vector<DTYPE, aligned_allocator<DTYPE>> &source_means,
				std::vector<DTYPE, aligned_allocator<DTYPE>> &source_norms,
				std::vector<DTYPE, aligned_allocator<DTYPE>> &source_df,
				std::vector<DTYPE, aligned_allocator<DTYPE>> &source_dg,
                std::vector<DTYPE, aligned_allocator<DTYPE>> &source_hw_profile,
				std::vector<ITYPE, aligned_allocator<ITYPE>> &source_hw_profileIndex,
                int *bank_assign,
                ITYPE size,
				ITYPE profileLength, ITYPE numDiagonals, ITYPE windowSize) {
    cl_int err;

    // For Allocating Buffer to specific Global Memory Bank, user has to use cl_mem_ext_ptr_t
    // and provide the Banks

    cl_mem_ext_ptr_t inBufExt1, inBufExt2, inBufExt3, inBufExt4, inBufExt5, inBufExt6, inBufExt7;
    cl_mem_ext_ptr_t inoutBufExt1, inoutBufExt2;

    inBufExt1.obj = source_tSeries.data();
    inBufExt1.param = 0;
    inBufExt1.flags = bank[0];

    inBufExt2.obj = source_diagonals.data();
    inBufExt2.param = 0;
    inBufExt2.flags = bank[1];

    inBufExt3.obj = source_means.data();
    inBufExt3.param = 0;
    inBufExt3.flags = bank[2];

    inBufExt4.obj = source_norms.data();
    inBufExt4.param = 0;
    inBufExt4.flags = bank[3];

    inBufExt5.obj = source_df.data();
    inBufExt5.param = 0;
    inBufExt5.flags = bank[4];

    inBufExt6.obj = source_dg.data();
    inBufExt6.param = 0;
    inBufExt6.flags = bank[5];

    inBufExt7.obj = source_tSeries.data();
    inBufExt7.param = 0;
    inBufExt7.flags = bank[8];

    inoutBufExt1.obj = source_hw_profile.data();
    inoutBufExt1.param = 0;
    inoutBufExt1.flags = bank[6];

    inoutBufExt2.obj = source_hw_profileIndex.data();
    inoutBufExt2.param = 0;
    inoutBufExt2.flags = bank[7];

    // Creating Buffers
    cout << "[HOST] Creating buffers...";
    OCL_CHECK(err, cl::Buffer buffer_input1(context,
    										CL_MEM_READ_ONLY |
                                            CL_MEM_EXT_PTR_XILINX |
                                            CL_MEM_USE_HOST_PTR,
										    sizeof(DTYPE) * size,
										   	&inBufExt1,
										   	&err));
    OCL_CHECK(err, cl::Buffer buffer_input2(context,
                                        	CL_MEM_READ_ONLY |
                                            CL_MEM_EXT_PTR_XILINX |
                                            CL_MEM_USE_HOST_PTR,
											sizeof(ITYPE) * size,
											&inBufExt2,
											&err));
    OCL_CHECK(err, cl::Buffer buffer_input3(context,
    										CL_MEM_READ_ONLY |
											CL_MEM_EXT_PTR_XILINX |
											CL_MEM_USE_HOST_PTR,
											sizeof(DTYPE) * size,
											&inBufExt3,
											&err));
    OCL_CHECK(err, cl::Buffer buffer_input4(context,
    										CL_MEM_READ_ONLY |
											CL_MEM_EXT_PTR_XILINX |
											CL_MEM_USE_HOST_PTR,
											sizeof(DTYPE) * size,
											&inBufExt4,
											&err));
    OCL_CHECK(err, cl::Buffer buffer_input5(context,
    										CL_MEM_READ_ONLY |
											CL_MEM_EXT_PTR_XILINX |
											CL_MEM_USE_HOST_PTR,
											sizeof(DTYPE) * size,
											&inBufExt5,
											&err));
    OCL_CHECK(err, cl::Buffer buffer_input6(context,
    										CL_MEM_READ_ONLY |
											CL_MEM_EXT_PTR_XILINX |
											CL_MEM_USE_HOST_PTR,
											sizeof(DTYPE) * size,
											&inBufExt6,
											&err));
    OCL_CHECK(err, cl::Buffer buffer_input7(context,
    										CL_MEM_READ_ONLY |
											CL_MEM_EXT_PTR_XILINX |
											CL_MEM_USE_HOST_PTR,
											sizeof(DTYPE) * size,
											&inBufExt7,
											&err));
    OCL_CHECK(err, cl::Buffer buffer_inout1(context,
                                       	   	CL_MEM_READ_WRITE |
											CL_MEM_EXT_PTR_XILINX |
											CL_MEM_USE_HOST_PTR,
											sizeof(DTYPE) * size,
											&inoutBufExt1,
											&err));
    OCL_CHECK(err, cl::Buffer buffer_inout2(context,
                                       	   	CL_MEM_READ_WRITE |
											CL_MEM_EXT_PTR_XILINX |
											CL_MEM_USE_HOST_PTR,
											sizeof(ITYPE) * size,
											&inoutBufExt2,
											&err));
    cout << "OK." << endl;
    //Setting the kernel Arguments
    OCL_CHECK(err, err = (kernel).setArg(0,  buffer_input1));
    OCL_CHECK(err, err = (kernel).setArg(1,  buffer_input7));
    OCL_CHECK(err, err = (kernel).setArg(2,  buffer_input2));
    OCL_CHECK(err, err = (kernel).setArg(3,  buffer_input3));
    OCL_CHECK(err, err = (kernel).setArg(4,  buffer_input4));
    OCL_CHECK(err, err = (kernel).setArg(5,  buffer_input5));
    OCL_CHECK(err, err = (kernel).setArg(6,  buffer_input6));
    OCL_CHECK(err, err = (kernel).setArg(7,  buffer_inout1));
    OCL_CHECK(err, err = (kernel).setArg(8,  buffer_inout2));
    OCL_CHECK(err, err = (kernel).setArg(9,  profileLength));
    OCL_CHECK(err, err = (kernel).setArg(10,  numDiagonals));
    OCL_CHECK(err, err = (kernel).setArg(11, windowSize));

    // Copy input data to Device Global Memory
    cout << "[HOST] Copying data to device...";
    OCL_CHECK(err,
              err = q.enqueueMigrateMemObjects({buffer_input1, buffer_input2,
    											buffer_input3, buffer_input4,
												buffer_input5, buffer_input6,buffer_input7,
												buffer_inout1, buffer_inout2},
                                                0 /* 0 means from host*/));
    q.finish();
    cout << "OK." << endl;

    cout << "[FPGA] Running SCAMP kernel..." << endl;
    std::chrono::duration<double> kernel_time(0);

    auto kernel_start = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = q.enqueueTask(kernel));
    q.finish();
    auto kernel_end = std::chrono::high_resolution_clock::now();

    kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);

    // Copy Result from Device Global Memory to Host Local Memory
    cout << "[HOST] Copying data from device...";
    OCL_CHECK(err,
             err = q.enqueueMigrateMemObjects({buffer_inout1, buffer_inout2},
                                               CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();
    cout << "OK." << endl;
    return kernel_time.count();
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <XCLBIN> \n", argv[0]);
        return -1;
    }
    cl_int err;
    cl::Context context;
    cl::CommandQueue q;
    cl::Kernel kernel_scamp;
    std::string binaryFile = argv[1];

    // The get_xil_devices will return vector of Xilinx Devices
    auto devices = xcl::get_xil_devices();

    // read_binary_file() command will find the OpenCL binary file created using the
    // V++ compiler load into OpenCL Binary and return pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);

    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    int valid_device = 0;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context({device}, NULL, NULL, NULL, &err));
        OCL_CHECK(err,
                  q = cl::CommandQueue(
                      context, {device}, CL_QUEUE_PROFILING_ENABLE, &err));

        std::cout << "Trying to program device[" << i
                  << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, NULL, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i
                      << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err,
                      kernel_scamp = cl::Kernel(program, "krnl_scamp", &err));
            valid_device++;
            break; // we break because we found a valid device
        }
    }
    if (valid_device == 0) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    unsigned int dataSize = 32768;
    if (xcl::is_emulation()) {
        dataSize = 1024;
        std::cout << "Original Dataset is reduced for faster execution on "
                     "emulation flow. Data size="
                  << dataSize << std::endl;
    }


    std::vector<DTYPE, aligned_allocator<DTYPE>> source_tSeries       	(dataSize);
    std::vector<DTYPE, aligned_allocator<DTYPE>> source_means         	(dataSize);
    std::vector<DTYPE, aligned_allocator<DTYPE>> source_norms        	(dataSize);
    std::vector<DTYPE, aligned_allocator<DTYPE>> source_df	        	(dataSize);
    std::vector<DTYPE, aligned_allocator<DTYPE>> source_dg	        	(dataSize);
    std::vector<DTYPE, aligned_allocator<DTYPE>> source_hw_profile    	(dataSize);
    std::vector<ITYPE, aligned_allocator<ITYPE>> source_hw_profileIndex	(dataSize);
    std::vector<DTYPE, aligned_allocator<DTYPE>> source_sw_profile    	(dataSize);
    std::vector<ITYPE, aligned_allocator<ITYPE>> source_sw_profileIndex	(dataSize);
    std::vector<ITYPE, aligned_allocator<ITYPE>> source_diagonals	 	(dataSize);

    // Prepare data
    ITYPE timeSeriesLength = dataSize;
	ITYPE windowSize = 256;
	ITYPE exclusionZone = windowSize / 4;
	ITYPE profileLength = timeSeriesLength - windowSize + 1;;


	int min = 5;
	int max = 100;

	for(ITYPE i = 0; i < source_tSeries.size(); i++)
	{
		source_tSeries[i] = min + (rand() % static_cast<int>(min - max + 1));
		//std::cout << source_tSeries[i] << std::endl;
	}

    preprocess(source_tSeries, source_means, source_norms, source_df,source_dg,profileLength, windowSize);

    source_diagonals.clear();
    for (ITYPE i = exclusionZone+1; i < profileLength; i++)
      source_diagonals.push_back(i);

//    std::random_shuffle(diagonals.begin(), diagonals.end());

    //end of data preparation

    // Initialize profiles
    for (ITYPE i = 0; i < dataSize; i++) {
        source_sw_profile[i] = -std::numeric_limits<DTYPE>::infinity();
        source_sw_profileIndex[i] = 0;
        source_hw_profile[i] = -std::numeric_limits<DTYPE>::infinity();
        source_hw_profileIndex[i] = 0;
    }

    //Compute host

    std::cout << "[HOST] Computing SCAMP..." << std::endl;
    std::chrono::duration<double> host_time(0);
    auto host_start = std::chrono::high_resolution_clock::now();

    scamp_host(source_tSeries,source_diagonals,source_means, source_norms,
    		source_df, source_dg, source_sw_profile, source_sw_profileIndex,
			profileLength, exclusionZone, windowSize);

    auto host_end = std::chrono::high_resolution_clock::now();
    host_time = std::chrono::duration<double>(host_end - host_start);

    std::cout << "[HOST] DONE. Execution time: " << host_time.count() << " seconds." << std::endl;

    double kernel_time_in_sec = 0;
    bool match = true;
    const int numBuf = 9; // Since four buffers are being used
    int bank_assign[numBuf];

    for (int j = 0; j < numBuf; j++) {
        bank_assign[j] = bank[j];
    }

    kernel_time_in_sec = run_krnl(context,
                                  q,
                                  kernel_scamp,
                                  source_tSeries,
								  source_diagonals,
                                  source_means,
								  source_norms,
								  source_df,
								  source_dg,
                                  source_hw_profile,
								  source_hw_profileIndex,
                                  bank_assign,
                                  dataSize,
								  profileLength,
								  source_diagonals.size(),
								  windowSize);

    match = verify(source_sw_profile, source_hw_profile, source_sw_profileIndex, source_hw_profileIndex ,dataSize);

    std::cout << "[FPGA] DONE. Execution time: " << kernel_time_in_sec << " seconds." << std::endl;

    std::cout << (match ? "TEST PASSED" : "TEST FAILED") << std::endl;
    return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}
