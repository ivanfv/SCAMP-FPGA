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
#define ARRAY_PARTITION  16
#define LOOP_UNROLLING   16

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
	DTYPE data_512 [16];
	DTYPE data_32;

}t_copy;

typedef struct access_type {
    ap_int<32> data[16];
} acces_type;

typedef union
{
	unsigned int raw_val;
	float float_val;
} raw_float;

inline float get_float(ap_int<MEM_WIDTH> ap, int index)
{
	raw_float tmp;
	tmp.raw_val = ap.range((index + 1) * FLOAT_BITS - 1, index * FLOAT_BITS);
	return tmp.float_val;
}

inline void store_float(ap_int<MEM_WIDTH> &ap, int index, DTYPE val)
{
	raw_float tmp;
	tmp.float_val = val;
	ap.range((index + 1) * FLOAT_BITS - 1, index * FLOAT_BITS) = tmp.raw_val;

}


void update_offset(ap_int<MEM_WIDTH> * profile_j_buf, DTYPE * profile_j, unsigned offset)
{
	unsigned index = 0;
	update_profile_l1:for(int j = offset; j < 16; j++)
	{
		#pragma HLS PIPELINE II=1
		store_float(profile_j_buf[0], j, profile_j[index]);
		index++;
	}

	update_profile_l2:for (int i = 1; i < 32; i++)
	{
		#pragma HLS PIPELINE II=1

		for(int j = 0; j < 16; j++)
		{
			#pragma HLS unroll factor=16
			store_float(profile_j_buf[i], j, profile_j[index]);
			index++;
		}
	}

	update_profile_l3:for(int j = 0; j < offset; j++)
	{
		#pragma HLS PIPELINE II=1
		store_float(profile_j_buf[32], j, profile_j[index]);
		index++;
	}
}



void adjust_offset(ap_int<MEM_WIDTH> * in, DTYPE * out, unsigned offset)
{
	unsigned index = 0;
	adjust_offset_l1:for(int j = offset; j < 16; j++)
	{
		#pragma HLS PIPELINE II=1
		out[index] = get_float(in[0], j);
		index++;
	}

	adjust_offset_l2:for (int i = 1; i < 32; i++)
	{
		#pragma HLS PIPELINE II=1

		for(int j = 0; j < 16; j++)
		{
			#pragma HLS unroll factor=16
			out[index] = get_float(in[i], j);
			index++;
		}
	}

	adjust_offset_l3:for(int j = 0; j < offset; j++)
	{
		#pragma HLS PIPELINE II=1
		out[index] = get_float(in[32], j);
		index++;
	}
}

/*inline float get_float(acces_type ap, int idx)
{
	raw_float tmp;
	tmp.raw_val = ap.data[idx];
	return tmp.float_val;
}*/

extern "C" {
void krnl_scamp(const ap_int<512> *tSeries_i,
				const ap_int<512> *tSeries_j,
				const ap_int<512> *means,			// Means input
				//const DTYPE *norms_i,		// Norms input 1
				const ap_int<512> *norms_i,		// Norms input 1
				const DTYPE *norms_j,		// Norms input 2
				const ap_int<512> *df_i,			// df input 1
				const DTYPE * df_j,	// df input 2
				const ap_int<512> *dg_i,			// dg input 1
				const DTYPE  * dg_j,	// df input 2
				ap_int<512> *profile_i,			// Profile input 1
				//DTYPE *profile_i,			// Profile input 1
				DTYPE *profile_j,			// Profile input 2
				//ap_int<512> *profile_j,			// Profile input 2
				ITYPE *profileIndex_i,		// ProfileIndex input 1
				ITYPE *profileIndex_j,		// ProfileIndex input 2
				const ITYPE profileLength,	// profileLength
				const ITYPE numDiagonals,	// numDiagonals
				const ITYPE windowSize		// windowSize
) {
	/* --------------------- INTERFACES CONFIGURATION -------------------------- */
	#pragma HLS INTERFACE m_axi port = tSeries_i      offset = slave bundle = gmem1 max_read_burst_length=256
	#pragma HLS INTERFACE m_axi port = tSeries_j      offset = slave bundle = gmem1 max_read_burst_length=256
	#pragma HLS INTERFACE m_axi port = means          offset = slave bundle = gmem1 max_read_burst_length=256
	#pragma HLS INTERFACE m_axi port = norms_i        offset = slave bundle = gmem1 max_read_burst_length=256
	#pragma HLS INTERFACE m_axi port = norms_j        offset = slave bundle = gmem0 max_read_burst_length=256
	#pragma HLS INTERFACE m_axi port = df_i           offset = slave bundle = gmem1 max_read_burst_length=256
	#pragma HLS INTERFACE m_axi port = df_j           offset = slave bundle = gmem0 max_read_burst_length=256
	#pragma HLS INTERFACE m_axi port = dg_i           offset = slave bundle = gmem1 max_read_burst_length=256
	#pragma HLS INTERFACE m_axi port = dg_j           offset = slave bundle = gmem0 max_read_burst_length=256
	#pragma HLS INTERFACE m_axi port = profile_i      offset = slave bundle = gmem1 max_read_burst_length=256
	#pragma HLS INTERFACE m_axi port = profile_j      offset = slave bundle = gmem0 max_read_burst_length=256
	#pragma HLS INTERFACE m_axi port = profileIndex_i offset = slave bundle = gmem0 max_read_burst_length=256
	#pragma HLS INTERFACE m_axi port = profileIndex_j offset = slave bundle = gmem0 max_read_burst_length=256

	#pragma HLS INTERFACE s_axilite port = tSeries_i        bundle = control
	#pragma HLS INTERFACE s_axilite port = tSeries_j        bundle = control
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

	// Diagonal position coordinates
	ITYPE i, j;

	// General purpose buffers
	ap_int<512> buff_A[33];
	ap_int<512> buff_B[33];
	ap_int<512> buff_C[33];
	ap_int<512> buff_D[33];

	// Time series temporal buffers
	ap_int<512> tmp_tSeries_i[32];
	DTYPE       tmp_tSeries_j[VDATA_SIZE];

	// Means temporal buffers
	DTYPE means_0;
	DTYPE tmp_means_j[VDATA_SIZE];

	// Norms temporal buffers
	ap_int<512> tmp_norms_i[VDATA_SIZE];
	DTYPE       tmp_norms_j[VDATA_SIZE];

	// df temporal buffers
	ap_int<512> tmp_df_i[VDATA_SIZE];
	DTYPE       tmp_df_j[VDATA_SIZE];

	// dg temporal buffers
	ap_int<512> tmp_dg_i[VDATA_SIZE];
	DTYPE       tmp_dg_j[VDATA_SIZE];

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

	// imax temporal buffer
	DTYPE tmp_i_max[4];
	ITYPE tmp_i_index_max[4];

	// Auxiliary variables
	unsigned i_read_counter;
	unsigned y_offset;
	unsigned index;

	/* ------------------------ ARRAY PARTITION POLICY ----------------------------- */
	#pragma HLS array_partition variable=tmp_tSeries_j    cyclic factor=array_partition
	#pragma HLS array_partition variable=tmp_means_j      cyclic factor=array_partition
	#pragma HLS array_partition variable=tmp_df_j         cyclic factor=array_partition
	#pragma HLS array_partition variable=tmp_dg_j         cyclic factor=array_partition
	#pragma HLS array_partition variable=tmp_norms_j      cyclic factor=array_partition
	#pragma HLS array_partition variable=tmp_covariances  cyclic factor=array_partition
	#pragma HLS array_partition variable=tmp_correlations cyclic factor=array_partition
	#pragma HLS array_partition variable=tmp_profile_j 	  cyclic factor=array_partition
	#pragma HLS array_partition variable=tmp_i_max 	      complete
	#pragma HLS array_partition variable=tmp_i_index_max  complete

	/* ----------------------------------------------------------------------------- */

	// Initialize means value for i=0 (same for all diagonals)
	means_0 = get_float(means[0], 0);

	// Main loop: go through diagonals
	main_loop: for(ITYPE diag = windowSize/4 + 1; diag < profileLength; diag+=VDATA_SIZE)
	{
		// Initialize diagonal coordinates
		i = 0;
		j = diag;

		// Initialization of covariances to 0
		for (int k = 0; k < VDATA_SIZE/2; k++)
		{
			#pragma HLS PIPELINE II=1
			tmp_covariances[k] = 0;
			tmp_covariances[k + VDATA_SIZE/2] = 0;
		}

		// Burst read of means for current set of diagonals
		index = 0;
		y_offset = j % 16;

		memcpy(buff_A, &means[j/16], 2112);

		adjust_offset(buff_A,  tmp_means_j, y_offset);


		// i-buffer counter initialization (forces first time filling)
		i_read_counter = VDATA_SIZE - 1;

		memcpy(buff_A, &(tSeries_i[(j) / 16]), 2112);

		// Calculate first covariances for this set of diagonals
		first_cov_main:for (ITYPE w = 0; w < windowSize; w++)
		{
			i_read_counter++;
			y_offset = (j + w) % 16;
			index = 0;

			// if i_read_counter reached VDATA_SIZE update buffer
			if(i_read_counter == VDATA_SIZE)
			{
				i_read_counter = 0;
				memcpy(tmp_tSeries_i, &tSeries_i[w / 16], sizeof(DTYPE) * VDATA_SIZE);
			}

			if(y_offset == 0)
				memcpy(buff_A, &(tSeries_i[(j + w) / 16]), 2112);

			adjust_offset(buff_A,  tmp_tSeries_j, y_offset);

			DTYPE tSeries_i_value = get_float(tmp_tSeries_i[i_read_counter / 16], i_read_counter % 16);

			first_cov_calc:for (int k = 0; k < VDATA_SIZE; k++)
			{
				#pragma HLS PIPELINE II=1
				#pragma HLS unroll factor=16
				tmp_covariances[k] += (tmp_tSeries_j[k] - tmp_means_j[k]) * (tSeries_i_value - means_0);
			}
	  }

	  i_read_counter = VDATA_SIZE - 1;


	  // Initialize buffers
	  memcpy(buff_A, &df_i[j / 16],    2112);
	  memcpy(buff_B, &dg_i[j / 16],    2112);
	  memcpy(buff_C, &norms_i[j / 16], 2112);
	//  memcpy(buff_D, &profile_j[j / 16], 2112);


	  diag:for (j = diag; j < profileLength; j++)
	  {
		  i_read_counter++;

		  if(i_read_counter == VDATA_SIZE)
		  {
			  i_read_counter = 0;

			  memcpy(tmp_norms_i, &norms_i[i / 16], sizeof(DTYPE) * VDATA_SIZE);
			  memcpy(tmp_df_i,       &df_i[i / 16], sizeof(DTYPE) * VDATA_SIZE);
			  memcpy(tmp_dg_i,       &dg_i[i / 16], sizeof(DTYPE) * VDATA_SIZE);
		 }

		  // Calculate j offset
		  y_offset = j % 16;

		  // Read norms if necessary
		  if(y_offset == 0)
		  {
			  memcpy(buff_C, &norms_i[j / 16], 2112);
		  }

		  adjust_offset(buff_C,  tmp_norms_j, y_offset);

		 // Obtain norm value for i index
		 DTYPE norms_i_value = get_float(tmp_norms_i[i_read_counter / 16], i_read_counter % 16);

		 // Calculate correlations
		 correlations:for (int k = 0; k < VDATA_SIZE; k++)
		 {
			 #pragma HLS unroll factor=loop_unrolling
			 #pragma HLS pipeline II=1
			 tmp_correlations[k] = tmp_covariances[k] * norms_i_value * tmp_norms_j[k];
		 }


		 // Obtain df and dg values for i index
		DTYPE df_i_value = get_float(tmp_df_i[i_read_counter / 16], i_read_counter % 16);
		DTYPE dg_i_value = get_float(tmp_dg_i[i_read_counter / 16], i_read_counter % 16);

		// Read df and dg if necessary
		if(y_offset == 0)
		{
			memcpy(buff_A, &df_i[j / 16], 2112);
			memcpy(buff_B, &dg_i[j / 16], 2112);
		}
		adjust_offset(buff_A,  tmp_df_j, y_offset);
		adjust_offset(buff_B,  tmp_dg_j, y_offset);


		// Update covariances
		covariances_update:for(int k = 0; k < VDATA_SIZE; k++)
		{
			#pragma HLS unroll factor=loop_unrolling
			#pragma HLS pipeline II=1
			tmp_covariances[k] += (df_i_value * tmp_dg_j[k]) + (tmp_df_j[k] * dg_i_value);
		}


		 if(j + VDATA_SIZE > profileLength)
		 {
			 unsigned num_preserve = profileLength  - j;

			 for(int l = num_preserve; l < VDATA_SIZE; l++)
			 {
				#pragma HLS pipeline II=1

				 tmp_correlations[l] = -1;
			 }

		 }

		tmp_i_max[0] = tmp_correlations[0];
		 tmp_i_max[1] = tmp_correlations[0];
		 tmp_i_max[2] = tmp_correlations[0];
		 tmp_i_max[3] = tmp_correlations[0];

		 calculate_i_updates:for (int k = 0; k < (VDATA_SIZE / 4); k++)
		{
			#pragma HLS pipeline II=1

			if(tmp_correlations[k] > tmp_i_max[0]){
				tmp_i_index_max[0] = j + k;
				tmp_i_max[0] = tmp_correlations[k];
			}


			if(tmp_correlations[(k + VDATA_SIZE / 4)] > tmp_i_max[1]){
				tmp_i_max[1] = tmp_correlations[(k + VDATA_SIZE / 4)];
				tmp_i_index_max[1] = j + (k + VDATA_SIZE / 4);
			}


			if(tmp_correlations[(k + VDATA_SIZE / 2)] > tmp_i_max[2]){
				tmp_i_max[2] = tmp_correlations[(k + VDATA_SIZE / 2)];
				tmp_i_index_max[2] = j + (k + VDATA_SIZE / 2);
			}

			if(tmp_correlations[(k + 3*VDATA_SIZE / 4)] > tmp_i_max[3]){
				tmp_i_max[3] = tmp_correlations[(k + 3*VDATA_SIZE / 4)];
				tmp_i_index_max[3] = j + (k + 3*VDATA_SIZE / 4);
			}

		}

		if(tmp_i_max[1] > tmp_i_max[0])
		{
			tmp_i_max[0] = tmp_i_max[1];
			tmp_i_index_max[0] = tmp_i_index_max[1];
		}

		if(tmp_i_max[2] > tmp_i_max[0])
		{
			tmp_i_max[0] = tmp_i_max[2];
			tmp_i_index_max[0] = tmp_i_index_max[2];
		}

		if(tmp_i_max[3] > tmp_i_max[0])
		{
			tmp_i_max[0] = tmp_i_max[3];
			tmp_i_index_max[0] = tmp_i_index_max[3];
		}



		 /*calculate_i_updates:for (int k = 0; k < VDATA_SIZE; k++)
		 {
		 			#pragma HLS pipeline II=1

		 			if(tmp_correlations[k] > tmp_i_max[0]){
		 				tmp_i_index_max[0] = j + k;
		 				tmp_i_max[0] = tmp_correlations[k];
		 			}

		 		}*/




		//if(y_offset == 0)
		//memcpy(buff_D, &profile_j[j / 16], 2112);

		// Read profile
		//memcpy(buff_D, &profile_j[j / 16], 2112);

		//adjust_offset(buff_D, tmp_profile_j, y_offset);


		/*calculate_j_updates:for (int k = 0; k < VDATA_SIZE; k++)
		{
				#pragma HLS unroll factor=loop_unrolling
				#pragma HLS pipeline II=1
				if (tmp_correlations[k] > tmp_profile_j[k])
				{
					tmp_profile_j[k] = tmp_correlations[k];
					tmp_profileIndex_j[k] = i;
				}
		}*/



		calculate_j_updates:for (int k = 0; k < VDATA_SIZE; k++)
		{
				#pragma HLS unroll factor=loop_unrolling
				#pragma HLS pipeline II=1
				if (tmp_correlations[k] > profile_j[j+k])
				{
					profile_j[j+k] = tmp_correlations[k];
					tmp_profileIndex_j[k] = i;
				}
		}


		///if(tmp_i_max[0] > profile_j[i])
		//	{
		//	 profile_j[i] = tmp_i_max[0];
				//profile_j[i / 16].range((i_offset + 1) * FLOAT_BITS - 1, i_offset * FLOAT_BITS) = tmp_float.raw_val;
				//profileIndex_i[i] = tmp_i_index_max[0];
		//	}


		//update_offset(buff_D, tmp_profile_j, y_offset);

		//memcpy(&profile_j[j / 16], buff_D, 2112);

		/*write_back_profile_j:for (int k = 0; k < VDATA_SIZE; k++)
		{
					#pragma HLS PIPELINE II=1
					profile_j[j + k] = tmp_profile_j[k];
		}*/

		unsigned i_offset = i % 16;

		ap_int<512> tmp_profile_i = profile_i[i / 16];

		raw_float tmp_float;

		if(tmp_i_max[0] > get_float(tmp_profile_i, i_offset))
		{
			tmp_float.float_val = tmp_i_max[0];
			tmp_profile_i.range((i_offset + 1) * FLOAT_BITS - 1, i_offset * FLOAT_BITS) = tmp_float.raw_val;
			profileIndex_i[i] = tmp_i_index_max[0];

			profile_i[i / 16] = tmp_profile_i;

		}


		/*ap_int<512> tmp_profile_i;

		memcpy(&tmp_profile_i, &profile_i[i / 16], 64);

		unsigned i_offset = i % 16;

		raw_float tmp_float;

		if(tmp_i_max[0] > get_float(tmp_profile_j, i_offset))
		{
			tmp_float.float_val = tmp_i_max[0];
			tmp_profile_i.range((i_offset + 1) * FLOAT_BITS - 1, i_offset * FLOAT_BITS) = tmp_float.raw_val;
			profileIndex_i[i] = tmp_i_index_max[0];
		}

		// Store max i
		memcpy(&profile_i[i / 16], &tmp_profile_i, 64);
*/
		i++;
	  }
	}
}
}
