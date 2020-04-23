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

/*******************************************************************************
Description: 
    This is a SCAMP implemented using C++ HLS.
    
*******************************************************************************/

#include "scamp.h"
#include <cstring>
#include "ap_int.h"

#define VDATA_SIZE       512
#define IDATA_SIZE       16
#define ARRAY_PARTITION  32
#define LOOP_UNROLLING   32

#define MEM_WIDTH 512
#define FLOAT_BITS 32

const unsigned int vector_data_size = VDATA_SIZE;
const unsigned int array_partition 	= ARRAY_PARTITION;
const unsigned int loop_unrolling 	= LOOP_UNROLLING;

/*typedef struct v_datatype {
    ap_int<32> data[16];
} v_dt;*/

typedef struct v_intdatatype {
    ITYPE data[IDATA_SIZE];
} v_intdt;

typedef union
{
	unsigned int raw_val;
	float float_val;
} raw_float;

inline float get_float(ap_int<MEM_WIDTH> ap, int index)
{
	raw_float tmp;
	tmp.raw_val = ap.range((index + 1) * FLOAT_BITS, index * FLOAT_BITS);
	return tmp.float_val;
}


extern "C" {
void krnl_scamp(const DTYPE *tSeries_i, 	// Time Series input 1
				const DTYPE *tSeries_j, 	// Time Series input 2
				const DTYPE *means,			// Means input
				const DTYPE *norms_i,		// Norms input 1
				const DTYPE *norms_j,		// Norms input 2
				const DTYPE *df_i,			// df input 1
				const ap_int<512> * df_j,	// df input 2
				const DTYPE *dg_i,			// dg input 1
				const DTYPE *dg_j,			// dg input 2
				DTYPE *profile_i,			// Profile input 1
				DTYPE *profile_j,			// Profile input 2
				ITYPE *profileIndex_i,		// ProfileIndex input 1
				ITYPE *profileIndex_j,		// ProfileIndex input 2
				const ITYPE profileLength,	// profileLength
				const ITYPE numDiagonals,	// numDiagonals
				const ITYPE windowSize		// windowSize
) {
	/* --------------------- INTERFACES CONFIGURATION -------------------------- */
	#pragma HLS INTERFACE m_axi port = tSeries_i      offset = slave bundle = gmem0 max_read_burst_length=256
	#pragma HLS INTERFACE m_axi port = tSeries_j      offset = slave bundle = gmem1	max_read_burst_length=256
	#pragma HLS INTERFACE m_axi port = means          offset = slave bundle = gmem0 max_read_burst_length=256
	#pragma HLS INTERFACE m_axi port = norms_i        offset = slave bundle = gmem0 max_read_burst_length=256
	#pragma HLS INTERFACE m_axi port = norms_j        offset = slave bundle = gmem1 max_read_burst_length=256
	#pragma HLS INTERFACE m_axi port = df_i           offset = slave bundle = gmem0 max_read_burst_length=256
	#pragma HLS INTERFACE m_axi port = df_j           offset = slave bundle = gmem1 max_read_burst_length=256
	#pragma HLS INTERFACE m_axi port = dg_i           offset = slave bundle = gmem0 max_read_burst_length=256
	#pragma HLS INTERFACE m_axi port = dg_j           offset = slave bundle = gmem1 max_read_burst_length=256
	#pragma HLS INTERFACE m_axi port = profile_i      offset = slave bundle = gmem0 max_read_burst_length=256
	#pragma HLS INTERFACE m_axi port = profile_j      offset = slave bundle = gmem1 max_read_burst_length=256
	#pragma HLS INTERFACE m_axi port = profileIndex_i offset = slave bundle = gmem0 max_read_burst_length=256
	#pragma HLS INTERFACE m_axi port = profileIndex_j offset = slave bundle = gmem1 max_read_burst_length=256

	#pragma HLS INTERFACE s_axilite port = tSeries_i      bundle = control
	#pragma HLS INTERFACE s_axilite port = tSeries_j      bundle = control
	#pragma HLS INTERFACE s_axilite port = means          bundle = control
	#pragma HLS INTERFACE s_axilite port = norms_i        bundle = control
	#pragma HLS INTERFACE s_axilite port = norms_j        bundle = control
	#pragma HLS INTERFACE s_axilite port = df_i           bundle = control
	#pragma HLS INTERFACE s_axilite port = df_j           bundle = control
	#pragma HLS INTERFACE s_axilite port = dg_i           bundle = control
	#pragma HLS INTERFACE s_axilite port = dg_j           bundle = control
	#pragma HLS INTERFACE s_axilite port = profile_i      bundle = control
	#pragma HLS INTERFACE s_axilite port = profile_j      bundle = control
	#pragma HLS INTERFACE s_axilite port = profileIndex_i bundle = control
	#pragma HLS INTERFACE s_axilite port = profileIndex_j bundle = control
	#pragma HLS INTERFACE s_axilite port = profileLength  bundle = control
	#pragma HLS INTERFACE s_axilite port = numDiagonals   bundle = control
	#pragma HLS INTERFACE s_axilite port = windowSize     bundle = control
	#pragma HLS INTERFACE s_axilite port = return         bundle = control
	/* -------------------------------------------------------------------------- */
	#pragma HLS DATA_PACK variable = df_j
	//#pragma HLS DATA_PACK variable = dg_j
	//#pragma HLS DATA_PACK variable = means
	//#pragma HLS DATA_PACK variable = norms
	//#pragma HLS DATA_PACK variable = df
	//#pragma HLS DATA_PACK variable = dg
	//#pragma HLS DATA_PACK variable = profile
	//#pragma HLS DATA_PACK variable = profileIndex

	// Diagonal position coordinates
	ITYPE i, j;

	// Time series temporal buffers
	DTYPE tmp_tSeries_i[VDATA_SIZE];
	DTYPE tmp_tSeries_j[VDATA_SIZE];

	// Means temporal buffers
	DTYPE means_0;
	DTYPE tmp_means_j[VDATA_SIZE];

	// Norms temporal buffers
	DTYPE tmp_norms_i[VDATA_SIZE];
	DTYPE tmp_norms_j[VDATA_SIZE];

	// df temporal buffers
	DTYPE tmp_df_i[VDATA_SIZE];
	DTYPE tmp_df_j[VDATA_SIZE];

	ap_int<512> tmp_df_j_test[32];


	// dg temporal buffers
	DTYPE tmp_dg_i[VDATA_SIZE];
	DTYPE tmp_dg_j[VDATA_SIZE];

	// profile temporal buffers
	DTYPE tmp_profile_i[VDATA_SIZE];
	DTYPE tmp_profile_j[VDATA_SIZE];

	// profileIndex temporal buffers
	ITYPE tmp_profileIndex_i[VDATA_SIZE];
	ITYPE tmp_profileIndex_j[VDATA_SIZE];

	// Covariances temporal buffer
	DTYPE tmp_covariances [VDATA_SIZE];

	// Correlations temporal buffer
	DTYPE tmp_correlations[VDATA_SIZE];

	// Auxiliary variables
	unsigned i_read_counter;

	/* ------------------------ ARRAY PARTITION POLICY ----------------------------- */
	#pragma HLS array_partition variable=tmp_tSeries_i    cyclic factor=array_partition
	#pragma HLS array_partition variable=tmp_tSeries_j    cyclic factor=array_partition
	#pragma HLS array_partition variable=tmp_means_j      cyclic factor=array_partition
	#pragma HLS array_partition variable=tmp_df_i 	      cyclic factor=array_partition
	#pragma HLS array_partition variable=tmp_df_j         cyclic factor=array_partition
	#pragma HLS array_partition variable=tmp_dg_i         cyclic factor=array_partition
	#pragma HLS array_partition variable=tmp_dg_j         cyclic factor=array_partition
	#pragma HLS array_partition variable=tmp_norms_i      cyclic factor=array_partition
	#pragma HLS array_partition variable=tmp_norms_j      cyclic factor=array_partition
	#pragma HLS array_partition variable=tmp_covariances  cyclic factor=array_partition
	#pragma HLS array_partition variable=tmp_correlations cyclic factor=array_partition
	#pragma HLS array_partition variable=tmp_profile_i 	  cyclic factor=array_partition
	#pragma HLS array_partition variable=tmp_profile_j 	  cyclic factor=array_partition
	/* ----------------------------------------------------------------------------- */

	// Initialize means value for i=0 (same for all diagonals)
	means_0 = means[0];

	// Main loop: go through diagonals
	main_loop: for(ITYPE diag = windowSize/4 + 1; diag < profileLength; diag+=VDATA_SIZE)
	{
		// Initialize diagonal coordinates
		i = 0;
		j = diag;

		// Initialization of covariances to 0
		for (int k = 0; k < VDATA_SIZE; k++)
		{
			#pragma HLS PIPELINE II=1
			tmp_covariances[k] = 0;
		}

		// Burst read of means for current set of diagonals
		for (int k = 0; k < VDATA_SIZE; k++)
		{
			#pragma HLS PIPELINE II=1
			tmp_means_j[k] = means[j + k];
		}

		// i-buffer counter initialization (forces first time filling)
		i_read_counter = VDATA_SIZE - 1;

		// Calculate first covariances for this set of diagonals
		first_cov:for (ITYPE w = 0; w < windowSize; w++)
		{
			i_read_counter++;

			// if i_read_counter reached VDATA_SIZE update buffer
			if(i_read_counter == VDATA_SIZE)
			{
				i_read_counter = 0;
				memcpy(tmp_tSeries_i, &tSeries_j[w], sizeof(DTYPE) * VDATA_SIZE);
			}

			// Burst read of tSeries_j buffer
			for (int k = 0; k < VDATA_SIZE; k++)
			{
				#pragma HLS PIPELINE II=1
				tmp_tSeries_j[k] = tSeries_i[j + w + k];
			}


			// Update of covariances
			first_cov_calc:for (int k = 0; k < VDATA_SIZE; k++)
			{
				#pragma HLS unroll factor=loop_unrolling
				tmp_covariances[k] += ((tmp_tSeries_j[k] - tmp_means_j[k]) * (tmp_tSeries_i[i_read_counter] - means_0));
			}
	  }


	  i_read_counter = VDATA_SIZE - 1;

	  diag:for (j = diag; j < profileLength; j++)
	  {

		  i_read_counter++;

		  if(i_read_counter == VDATA_SIZE)
		  {
			  i_read_counter = 0;

			  read_norms_profile_i:for (int k = 0; k < VDATA_SIZE; k++)
			  {
				  #pragma HLS PIPELINE II=1
				  tmp_norms_i[k]   = norms_j[i + k];
				  tmp_profile_i[k] = profile_i[i + k];
			  }

			  read_df_dg_i:for (int k = 0; k < VDATA_SIZE; k++)
			  {
				  #pragma HLS PIPELINE II=1
				  tmp_df_i[k] = df_i[i + k];
				  tmp_dg_i[k] = dg_j[i + k];
			  }

		 }

		  read_norms_profile_j:for (int k = 0; k < VDATA_SIZE; k++)
		  {
			  #pragma HLS PIPELINE II=1
			  tmp_norms_j[k]   = norms_i[j + k];
			  tmp_profile_j[k] = profile_j[j + k];
		  }

		  read_df_dg_j:for (int k = 0; k < VDATA_SIZE; k++)
		  {
			  #pragma HLS PIPELINE II=1
			  tmp_df_j[k] = df_i[j + k];
			  tmp_dg_j[k] = dg_j[j + k];
		  }

		memcpy(tmp_df_j_test, &df_j[j], sizeof(DTYPE) * VDATA_SIZE);



		 correlations:for (int k = 0; k < VDATA_SIZE; k++)
		 {
			 #pragma HLS unroll factor=loop_unrolling
			 #pragma HLS pipeline II=1
			 tmp_correlations[k] = tmp_covariances[k] * tmp_norms_i[i_read_counter] * get_float(tmp_df_j_test, 4)/*tmp_norms_j[k]*/;
		 }


		covariances_update:for(int k = 0; k < VDATA_SIZE; k++)
		{
			#pragma HLS unroll factor=loop_unrolling
			#pragma HLS pipeline II=1
			tmp_covariances[k] += (tmp_df_i[i_read_counter] * tmp_dg_j[k]) + (tmp_df_j[k] * tmp_dg_i[i_read_counter]);
		}


		calculate_i_updates:for (int k = 0; k < VDATA_SIZE; k++)
		{
				#pragma HLS pipeline II=1
				if (tmp_correlations[k] > tmp_profile_i[i_read_counter])
				{
					tmp_profile_i[i_read_counter]      = tmp_correlations[k];
					tmp_profileIndex_i[i_read_counter] = j + k;
				}
		}


		calculate_j_updates:for (int k = 0; k < VDATA_SIZE; k++)
		{
				#pragma HLS unroll factor=loop_unrolling
				#pragma HLS pipeline II=1
				if (tmp_correlations[k] > tmp_profile_j[k])
				{
					tmp_profile_j[k] = tmp_correlations[k];
					tmp_profileIndex_j[k] = i;
				}
		}

		write_back_profile_j:for (int k = 0; k < VDATA_SIZE; k++)
		{
					#pragma HLS PIPELINE II=1
					profile_j[j + k] = tmp_profile_j[k];
		}

		profile_i[i] = tmp_profile_i[i_read_counter];
		i++;
	  }
	}
}
}
