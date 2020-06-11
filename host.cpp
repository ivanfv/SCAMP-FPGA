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
	      ITYPE i = 0;
	      for (ITYPE w = 0; w < windowSize; w++)
	      {
	        covariance += ((tSeries[diag + w] - means[diag]) * (tSeries[w] - means[0]));
	      }
	        correlation = covariance *norms[0] * norms[diag];


	        if (correlation > profile[i])
	        {
	          profile[i] = correlation;
	          profileIndex[i] = diag;
	        }

	        if (correlation > profile[diag])
	        {
	          profile[diag] = correlation;
	          profileIndex[diag] = i;
	        }


	      i++;
	      for (ITYPE j = diag + 1; j < profileLength; j++)
	      {
	    	covariance += (df[i-1] * dg[j-1]) + (df[j-1] * dg[i-1]);
	        correlation = covariance *norms[i] * norms[j];

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
	      }
	    }
}


// Function for verifying results
bool verify(std::vector<DTYPE, aligned_allocator<DTYPE>> &source_sw_profile,
            std::vector<DTYPE, aligned_allocator<DTYPE>> &source_hw_profile,
			std::vector<ITYPE, aligned_allocator<ITYPE>> &source_sw_profileIdxs,
			std::vector<ITYPE, aligned_allocator<ITYPE>> &source_hw_profileIdxs,
            unsigned int size) {
    bool check = true;
    unsigned counter = 0;
    for (size_t i = 0; i < size; i++) {
        if (source_hw_profile[i] != source_sw_profile[i] && counter < 32) {
            std::cout << "Error: Profile result mismatch" << std::endl;
            std::cout << "i = " << i << " CPU result = " << source_sw_profile[i]
                      << " Device result = " << source_hw_profile[i]
                      << std::endl;
            check = false;
            counter++;
        }
       if (source_hw_profileIdxs[i] != source_sw_profileIdxs[i]) {
            std::cout << "Error: Profile Idxs result mismatch" << std::endl;
            std::cout << "i = " << i << " CPU result = " << source_sw_profileIdxs[i]
                      << " Device result = " << source_hw_profileIdxs[i]
                      << std::endl;
            check = false;
            //break;
        }
    }
    return check;
}

double run_krnls(cl::Context &context,
                cl::CommandQueue &q,
                std::vector<cl::Kernel> &kernels,
                std::vector<DTYPE, aligned_allocator<DTYPE>> &source_tSeries,
                std::vector<DTYPE, aligned_allocator<DTYPE>> &source_means,
				std::vector<DTYPE, aligned_allocator<DTYPE>> &source_norms,
				std::vector<DTYPE, aligned_allocator<DTYPE>> &source_df,
				std::vector<DTYPE, aligned_allocator<DTYPE>> &source_dg,
                std::vector<DTYPE, aligned_allocator<DTYPE>> &source_hw_profile,
				std::vector<ITYPE, aligned_allocator<ITYPE>> &source_hw_profileIndex,
                const int *bank_assign,
                ITYPE size,
				ITYPE profileLength, ITYPE exclusionZone, ITYPE windowSize) {
    cl_int err;

    // Temporal profile and profileIndex
    std::vector<DTYPE, aligned_allocator<DTYPE>> profile_tmp_0    	(size + VDATA_SIZE);
    std::vector<DTYPE, aligned_allocator<DTYPE>> profile_tmp_1    	(size + VDATA_SIZE);
    std::vector<DTYPE, aligned_allocator<DTYPE>> profile_tmp_2    	(size + VDATA_SIZE);
    std::vector<DTYPE, aligned_allocator<DTYPE>> profile_tmp_3    	(size + VDATA_SIZE);
    std::vector<ITYPE, aligned_allocator<ITYPE>> profileIndex_tmp_0	(size + VDATA_SIZE);
    std::vector<ITYPE, aligned_allocator<ITYPE>> profileIndex_tmp_1	(size + VDATA_SIZE);
    std::vector<ITYPE, aligned_allocator<ITYPE>> profileIndex_tmp_2	(size + VDATA_SIZE);
    std::vector<ITYPE, aligned_allocator<ITYPE>> profileIndex_tmp_3	(size + VDATA_SIZE);

    // Temporal profiles initialization
    for(ITYPE i = 0; i < size; i++)
    {
    	profile_tmp_0[i] = -std::numeric_limits<DTYPE>::infinity();
    	profile_tmp_1[i] = -std::numeric_limits<DTYPE>::infinity();
    	profile_tmp_2[i] = -std::numeric_limits<DTYPE>::infinity();
    	profile_tmp_3[i] = -std::numeric_limits<DTYPE>::infinity();

    	profileIndex_tmp_0[i] = 0;
    	profileIndex_tmp_1[i] = 0;
    	profileIndex_tmp_2[i] = 0;
    	profileIndex_tmp_3[i] = 0;
    }



    // Static Scheduling
    ITYPE startDiags[NUM_KERNELS];
    ITYPE endDiags[NUM_KERNELS];

   // startDiags[0] = exclusionZone + 1;
   // endDiags[0]   = profileLength / 3;

   // startDiags[1] = (profileLength / 3) + 1;
   // endDiags[1]   = profileLength;

    startDiags[0] = exclusionZone + 1;
     endDiags[0]   = profileLength / 9;

    startDiags[1] = (profileLength / 9) + 1;
     endDiags[1]   = (profileLength / 3);

     startDiags[2] = profileLength / 3 + 1;
      endDiags[2]   = profileLength * 5 / 9;

     startDiags[3] = (profileLength * 5 / 9) + 1;
      endDiags[3]   = profileLength;




    cout << "[HOST] Creating buffers...";
    // For Allocating Buffer to specific Global Memory Bank, user has to use cl_mem_ext_ptr_t
    // and provide the Banks
    cl_mem_ext_ptr_t inBuffersExt[22];
    cl_mem_ext_ptr_t inOutBuffersExt[8];

    // Read-only buffers parameters
    unsigned index = 0;

    // In buffers HBM[0]
    for(unsigned i = 0; i < 11; i++)
    {
    		inBuffersExt[index].param = 0;
    		inBuffersExt[index].flags = bank[i];
    		index++;
    }

    // In buffers HBM[1]
    for(unsigned i = 15; i < 26; i++)
    {
    		inBuffersExt[index].param = 0;
    		inBuffersExt[index].flags = bank[i];
    		index++;
    }


    inBuffersExt[0].obj  = source_tSeries.data();
    inBuffersExt[1].obj  = source_means.data();
    inBuffersExt[2].obj  = source_df.data();
    inBuffersExt[3].obj  = source_dg.data();
    inBuffersExt[4].obj  = source_norms.data();
    inBuffersExt[5].obj  = source_df.data();
    inBuffersExt[6].obj  = source_dg.data();
    inBuffersExt[7].obj  = source_norms.data();
    inBuffersExt[8].obj  = source_df.data();
    inBuffersExt[9].obj  = source_dg.data();
    inBuffersExt[10].obj = source_norms.data();
    inBuffersExt[11].obj = source_tSeries.data();
    inBuffersExt[12].obj = source_means.data();
    inBuffersExt[13].obj = source_df.data();
    inBuffersExt[14].obj = source_dg.data();
    inBuffersExt[15].obj = source_norms.data();
    inBuffersExt[16].obj = source_df.data();
    inBuffersExt[17].obj = source_dg.data();
    inBuffersExt[18].obj = source_norms.data();
    inBuffersExt[19].obj = source_df.data();
    inBuffersExt[20].obj = source_dg.data();
    inBuffersExt[21].obj = source_norms.data();

    // Read-Write buffers parameters
    index = 0;

    // Inout buffers HBM[0]
    for(unsigned i = 11; i < 15; i++)
    {
    		inOutBuffersExt[index].param = 0;
    		inOutBuffersExt[index].flags = bank[i];
			index++;
    }

    // Inout buffers HBM[1]
    for(unsigned i = 26; i < 30; i++)
    {
    		inOutBuffersExt[index].param = 0;
    		inOutBuffersExt[index].flags = bank[i];
			index++;
    }

    inOutBuffersExt[0].obj = profile_tmp_0.data();
	inOutBuffersExt[1].obj = profileIndex_tmp_0.data();
	inOutBuffersExt[2].obj = profile_tmp_1.data();
	inOutBuffersExt[3].obj = profileIndex_tmp_1.data();
    inOutBuffersExt[4].obj = profile_tmp_2.data();
	inOutBuffersExt[5].obj = profileIndex_tmp_2.data();
	inOutBuffersExt[6].obj = profile_tmp_3.data();
	inOutBuffersExt[7].obj = profileIndex_tmp_3.data();


	// Read-only buffers create
	cl::Buffer buffers_input[22];




	// 5 buffers for PU 0 and 1
	for(unsigned i = 0; i < 8; i++)
	{
		buffers_input[i] = cl::Buffer (context, CL_MEM_READ_ONLY |
									   CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
									   sizeof(DTYPE) * (size + VDATA_SIZE), &inBuffersExt[i], &err);
	}

	for(unsigned i = 11; i < 19; i++)
	{
		buffers_input[i] = cl::Buffer (context, CL_MEM_READ_ONLY |
									   CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
									   sizeof(DTYPE) * (size + VDATA_SIZE), &inBuffersExt[i], &err);
	}


	// Read-Write buffers create
	cl::Buffer buffers_inout[8];

	// 2 buffers for PU 0 and 1
	for(unsigned i = 0; i < 8; i++)
	{
			buffers_inout[i] = cl::Buffer (context,
               	   									 CL_MEM_READ_WRITE |
													 CL_MEM_EXT_PTR_XILINX |
													 CL_MEM_USE_HOST_PTR,
													 sizeof(DTYPE) * (size + VDATA_SIZE),
													 &inOutBuffersExt[i],
													 &err);
	}


    cout << "OK." << endl;
    //Setting the kernel Arguments

    // Kernel 0
    OCL_CHECK(err, err = (kernels[0]).setArg(0, buffers_input[0])); // tSeries_i
    OCL_CHECK(err, err = (kernels[0]).setArg(1, buffers_input[1])); // means
    OCL_CHECK(err, err = (kernels[0]).setArg(2, buffers_input[2])); // df
    OCL_CHECK(err, err = (kernels[0]).setArg(3, buffers_input[3])); // dg
    OCL_CHECK(err, err = (kernels[0]).setArg(4, buffers_input[4])); // norms
    OCL_CHECK(err, err = (kernels[0]).setArg(5, buffers_inout[0])); // profile
    OCL_CHECK(err, err = (kernels[0]).setArg(6, buffers_inout[1])); // profileIndex
    OCL_CHECK(err, err = (kernels[0]).setArg(7, profileLength));	// profileLength
    OCL_CHECK(err, err = (kernels[0]).setArg(8, startDiags[0])); 	// startDiag
    OCL_CHECK(err, err = (kernels[0]).setArg(9, endDiags[0]));	    // endDiag
    OCL_CHECK(err, err = (kernels[0]).setArg(10, windowSize));		// windowSize

    // Kernel 1
    OCL_CHECK(err, err = (kernels[1]).setArg(0, buffers_input[0])); // tSeries_i
    OCL_CHECK(err, err = (kernels[1]).setArg(1, buffers_input[1])); // means
    OCL_CHECK(err, err = (kernels[1]).setArg(2, buffers_input[5])); // df
    OCL_CHECK(err, err = (kernels[1]).setArg(3, buffers_input[6])); // dg
    OCL_CHECK(err, err = (kernels[1]).setArg(4, buffers_input[7])); // norms
    OCL_CHECK(err, err = (kernels[1]).setArg(5, buffers_inout[2])); // profile
    OCL_CHECK(err, err = (kernels[1]).setArg(6, buffers_inout[3])); // profileIndex
    OCL_CHECK(err, err = (kernels[1]).setArg(7, profileLength));	// profileLength
    OCL_CHECK(err, err = (kernels[1]).setArg(8, startDiags[1])); 	// startDiag
    OCL_CHECK(err, err = (kernels[1]).setArg(9, endDiags[1]));	    // endDiag
    OCL_CHECK(err, err = (kernels[1]).setArg(10, windowSize));		// windowSize

    // Kernel 2
    OCL_CHECK(err, err = (kernels[2]).setArg(0, buffers_input[11])); // tSeries_i
    OCL_CHECK(err, err = (kernels[2]).setArg(1, buffers_input[12])); // means
    OCL_CHECK(err, err = (kernels[2]).setArg(2, buffers_input[13])); // df
    OCL_CHECK(err, err = (kernels[2]).setArg(3, buffers_input[14])); // dg
    OCL_CHECK(err, err = (kernels[2]).setArg(4, buffers_input[15])); // norms
    OCL_CHECK(err, err = (kernels[2]).setArg(5, buffers_inout[4])); // profile
    OCL_CHECK(err, err = (kernels[2]).setArg(6, buffers_inout[5])); // profileIndex
    OCL_CHECK(err, err = (kernels[2]).setArg(7, profileLength));	// profileLength
    OCL_CHECK(err, err = (kernels[2]).setArg(8, startDiags[2])); 	// startDiag
    OCL_CHECK(err, err = (kernels[2]).setArg(9, endDiags[2]));	    // endDiag
    OCL_CHECK(err, err = (kernels[2]).setArg(10, windowSize));		// windowSize

    // Kernel 3
    OCL_CHECK(err, err = (kernels[3]).setArg(0, buffers_input[11])); // tSeries_i
    OCL_CHECK(err, err = (kernels[3]).setArg(1, buffers_input[12])); // means
    OCL_CHECK(err, err = (kernels[3]).setArg(2, buffers_input[16])); // df
    OCL_CHECK(err, err = (kernels[3]).setArg(3, buffers_input[17])); // dg
    OCL_CHECK(err, err = (kernels[3]).setArg(4, buffers_input[18])); // norms
    OCL_CHECK(err, err = (kernels[3]).setArg(5, buffers_inout[6])); // profile
    OCL_CHECK(err, err = (kernels[3]).setArg(6, buffers_inout[7])); // profileIndex
    OCL_CHECK(err, err = (kernels[3]).setArg(7, profileLength));	// profileLength
    OCL_CHECK(err, err = (kernels[3]).setArg(8, startDiags[3])); 	// startDiag
    OCL_CHECK(err, err = (kernels[3]).setArg(9, endDiags[3]));	    // endDiag
    OCL_CHECK(err, err = (kernels[3]).setArg(10, windowSize));		// windowSize

    // Copy input data to Device Global Memory
    cout << "[HOST] Copying data to device...";
	for(unsigned i = 0; i < 8; i++)
	{
	    OCL_CHECK(err,
	              err = q.enqueueMigrateMemObjects({buffers_input[i]},
	                                                0 /* 0 means from host*/));
	}

	for(unsigned i = 11; i < 19; i++)
	{
	    OCL_CHECK(err,
	              err = q.enqueueMigrateMemObjects({buffers_input[i]},
	                                                0 /* 0 means from host*/));
	}

	for(unsigned i = 0; i < 8; i++)
	{
	    OCL_CHECK(err,
	              err = q.enqueueMigrateMemObjects({buffers_inout[i]},
	                                                0 /* 0 means from host*/));
	}

    q.finish();
    cout << "OK." << endl;

    cout << "[FPGA] Running SCAMP kernel..." << endl;
    std::chrono::duration<double> kernel_time(0);

    auto kernel_start = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = q.enqueueTask(kernels[0]));
    OCL_CHECK(err, err = q.enqueueTask(kernels[1]));
    OCL_CHECK(err, err = q.enqueueTask(kernels[2]));
    OCL_CHECK(err, err = q.enqueueTask(kernels[3]));
    q.finish();
    auto kernel_end = std::chrono::high_resolution_clock::now();

    kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);

    // Copy Result from Device Global Memory to Host Local Memory
    cout << "[HOST] Copying data from device...";
	for(unsigned i = 0; i < 8; i++)
	{
	    OCL_CHECK(err,
	              err = q.enqueueMigrateMemObjects({buffers_inout[i]},
	            		  	  	  	  	  	  CL_MIGRATE_MEM_OBJECT_HOST));
	}
    /*OCL_CHECK(err,
             err = q.enqueueMigrateMemObjects({buffer_inout1, buffer_inout2},
                                               CL_MIGRATE_MEM_OBJECT_HOST));*/
    q.finish();


    // Profile reduction
    for(ITYPE i = 0; i < profileLength; i++)
    {
    	// Prof i
    	if (profile_tmp_0[i] > source_hw_profile[i])
    	{
    		source_hw_profile[i] = profile_tmp_0[i];
    		source_hw_profileIndex[i] = profileIndex_tmp_0[i];
    	}

    	if (profile_tmp_1[i] > source_hw_profile[i])
    	{
    		source_hw_profile[i] = profile_tmp_1[i];
    		source_hw_profileIndex[i] = profileIndex_tmp_1[i];
    	}

    	//prof J
    	if (profile_tmp_2[i] > source_hw_profile[i])
    	{
    		source_hw_profile[i] = profile_tmp_2[i];
    		source_hw_profileIndex[i] = profileIndex_tmp_2[i];
    	}
    	if (profile_tmp_3[i] > source_hw_profile[i])
    	{
    		source_hw_profile[i] = profile_tmp_3[i];
    		source_hw_profileIndex[i] = profileIndex_tmp_3[i];
    	}
    }
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
    std::vector<cl::Kernel> kernels(NUM_KERNELS);
    std::string binaryFile = argv[1];
    std::string kernel_name = "krnl_scamp";

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
                      context, {device}, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));

        std::cout << "Trying to program device[" << i
                  << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, NULL, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i
                      << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            for(int k = 0; k < NUM_KERNELS; k++)
            {
            	std::string s = kernel_name+":{"+kernel_name + "_" + to_string(k+1)+"}";
            	OCL_CHECK(err,
            			kernels[k] = cl::Kernel(program, s.c_str(), &err));
            }
            valid_device++;
            break; // we break because we found a valid device
        }
    }
    if (valid_device == 0) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    unsigned int dataSize = 131072;
    if (xcl::is_emulation()) {
        dataSize = 2048;
        std::cout << "Original Dataset is reduced for faster execution on "
                     "emulation flow. Data size="
                  << dataSize << std::endl;
    }


    std::vector<DTYPE, aligned_allocator<DTYPE>> source_tSeries       	(dataSize + 512);
    std::vector<DTYPE, aligned_allocator<DTYPE>> source_means         	(dataSize + 512);
    std::vector<DTYPE, aligned_allocator<DTYPE>> source_norms        	(dataSize + 512);
    std::vector<DTYPE, aligned_allocator<DTYPE>> source_df	        	(dataSize + 512);
    std::vector<DTYPE, aligned_allocator<DTYPE>> source_dg	        	(dataSize + 512);
    std::vector<DTYPE, aligned_allocator<DTYPE>> source_hw_profile    	(dataSize + 512);
    std::vector<ITYPE, aligned_allocator<ITYPE>> source_hw_profileIndex	(dataSize + 512);
    std::vector<DTYPE, aligned_allocator<DTYPE>> source_sw_profile    	(dataSize);
    std::vector<ITYPE, aligned_allocator<ITYPE>> source_sw_profileIndex	(dataSize);

    // Prepare data
    ITYPE timeSeriesLength = dataSize;
	ITYPE windowSize = 1024;
	ITYPE exclusionZone = windowSize / 4;
	ITYPE profileLength = timeSeriesLength - windowSize + 1;

	int min = 5;
	int max = 100;

	// Generate a random time series
	for(ITYPE i = 0; i < source_tSeries.size(); i++)
	{
		source_tSeries[i] = min + (rand() % static_cast<int>(min - max + 1));
	}

    preprocess(source_tSeries, source_means, source_norms, source_df,source_dg,profileLength, windowSize);

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

    scamp_host(source_tSeries,source_means, source_norms,
    		source_df, source_dg, source_sw_profile, source_sw_profileIndex,
			profileLength, exclusionZone, windowSize);
    // cambiar a steady clock
    auto host_end = std::chrono::high_resolution_clock::now();
    host_time = std::chrono::duration<double>(host_end - host_start);

    std::cout << "[HOST] DONE. Execution time: " << host_time.count() << " seconds." << std::endl;

    double kernel_time_in_sec = 0;
    bool match = false;



    kernel_time_in_sec = run_krnls(context,
                                  q,
                                  kernels,
                                  source_tSeries,
                                  source_means,
								  source_norms,
								  source_df,
								  source_dg,
                                  source_hw_profile,
								  source_hw_profileIndex,
                                  bank,
                                  dataSize,
								  profileLength,
								  exclusionZone,
								  windowSize);

    match = verify(source_sw_profile, source_hw_profile, source_sw_profileIndex, source_hw_profileIndex ,dataSize);
//match = true;

    std::cout << "[FPGA] DONE. Execution time: " << kernel_time_in_sec << " seconds." << std::endl;

    std::cout << (match ? "TEST PASSED" : "TEST FAILED") << std::endl;
    return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}
