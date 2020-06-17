/**********
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
**********/

/*******************************************************************************
Description: 
    This is the SCAMP kernel implementation using C++ HLS
*******************************************************************************/

#include "scamp.h"
#include <cstring>
#include "ap_int.h"
#include <iostream>

const unsigned int vector_data_size  = VDATA_SIZE;
const unsigned int array_partition 	 = ARRAY_PARTITION;
const unsigned int loop_unrolling 	 = LOOP_UNROLLING;
const unsigned int wide_buff_size    = (VDATA_SIZE / (MEM_WIDTH / FLOAT_BITS));
const unsigned int general_buff_size = (VDATA_SIZE / (MEM_WIDTH / FLOAT_BITS)) + 1;

typedef union
{
	unsigned int raw_val;
	float float_val;
} raw_float;

inline float get_float(ap_int<MEM_WIDTH> ap, int index)
{
#pragma HLS inline
	raw_float tmp;
	tmp.raw_val = ap.range((index + 1) * FLOAT_BITS - 1, index * FLOAT_BITS);
	return tmp.float_val;
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

extern "C" {
void krnl_scamp(const ap_int<MEM_WIDTH> *tSeries, // tSeries input
				const ap_int<MEM_WIDTH> *means,	  // Means input
				const ap_int<MEM_WIDTH> *df,	  // df input
				const ap_int<MEM_WIDTH> *dg,	  // dg input
				const ap_int<MEM_WIDTH> *norms,	  // Norms input
				DTYPE *profile,					  // Profile input
				ITYPE *profileIndex,		      // ProfileIndex input
				const ITYPE profileLength,	      // profileLength
				const ITYPE startDiag,	          // First Diagonal
				const ITYPE endDiag,	          // Last Diagonal
				const ITYPE windowSize		      // windowSize
) {
	/* -------------------------------- INTERFACES CONFIGURATION --------------------------------------- */
	#pragma HLS INTERFACE m_axi port = tSeries      offset = slave bundle = gmem0 max_read_burst_length=64
	#pragma HLS INTERFACE m_axi port = means        offset = slave bundle = gmem0 max_read_burst_length=64
	#pragma HLS INTERFACE m_axi port = df           offset = slave bundle = gmem0 max_read_burst_length=64
	#pragma HLS INTERFACE m_axi port = dg           offset = slave bundle = gmem1 max_read_burst_length=64
	#pragma HLS INTERFACE m_axi port = norms        offset = slave bundle = gmem0 max_read_burst_length=64
	#pragma HLS INTERFACE m_axi port = profile      offset = slave bundle = gmem2 max_read_burst_length=64
	#pragma HLS INTERFACE m_axi port = profileIndex offset = slave bundle = gmem2 max_read_burst_length=64

	#pragma HLS INTERFACE s_axilite port = tSeries       bundle = control
	#pragma HLS INTERFACE s_axilite port = means         bundle = control
	#pragma HLS INTERFACE s_axilite port = df            bundle = control
	#pragma HLS INTERFACE s_axilite port = dg            bundle = control
	#pragma HLS INTERFACE s_axilite port = norms         bundle = control
	#pragma HLS INTERFACE s_axilite port = profile       bundle = control
	#pragma HLS INTERFACE s_axilite port = profileIndex  bundle = control
	#pragma HLS INTERFACE s_axilite port = profileLength bundle = control
	#pragma HLS INTERFACE s_axilite port = startDiag     bundle = control
	#pragma HLS INTERFACE s_axilite port = endDiag       bundle = control
	#pragma HLS INTERFACE s_axilite port = windowSize    bundle = control
	#pragma HLS INTERFACE s_axilite port = return        bundle = control
	/* ------------------------------------------------------------------------------------------------- */

	// General purpose buffers
	ap_int<MEM_WIDTH> buff_A[general_buff_size];
	ap_int<MEM_WIDTH> buff_B[general_buff_size];
	ap_int<MEM_WIDTH> buff_C[general_buff_size];
	ap_int<MEM_WIDTH> buff_D[general_buff_size];

	// Time series temporal buffers
	ap_int<MEM_WIDTH> tmp_tSeries_i[wide_buff_size];
	DTYPE             tmp_tSeries_j[VDATA_SIZE];

	// Means temporal buffers
	DTYPE means_0;
	DTYPE tmp_means_j[VDATA_SIZE];

	// df temporal buffers
	ap_int<MEM_WIDTH> tmp_df_i[VDATA_SIZE];
	DTYPE tmp_df_j            [VDATA_SIZE];

	// dg temporal buffers
	ap_int<MEM_WIDTH> tmp_dg_i[VDATA_SIZE];
	DTYPE tmp_dg_j            [VDATA_SIZE];

	// profile temporal buffers
	DTYPE tmp_profile_j[VDATA_SIZE];

	// profileIndex temporal buffers
	ITYPE tmp_profileIndex_i[VDATA_SIZE];
	ITYPE tmp_profileIndex_j[VDATA_SIZE];

	// Covariances temporal buffer
	DTYPE tmp_covariances [VDATA_SIZE];

	// Correlations temporal buffer
	DTYPE tmp_correlations[VDATA_SIZE];

	// imax temporal buffer
	DTYPE tmp_i_max      [loop_unrolling];
	ITYPE tmp_i_index_max[loop_unrolling];

	// Auxiliary variables
	ITYPE i, j;
	unsigned i_read_counter;
	unsigned j_offset;
	unsigned index;

	/* ------------------------ ARRAY PARTITION POLICY ----------------------------- */
	#pragma HLS array_partition variable=tmp_tSeries_j      cyclic factor=array_partition
	#pragma HLS array_partition variable=tmp_means_j        cyclic factor=array_partition
	#pragma HLS array_partition variable=tmp_df_j           cyclic factor=array_partition
	#pragma HLS array_partition variable=tmp_dg_j           cyclic factor=array_partition
	#pragma HLS array_partition variable=tmp_covariances    cyclic factor=array_partition
	#pragma HLS array_partition variable=tmp_correlations   cyclic factor=array_partition
	#pragma HLS array_partition variable=tmp_profile_j 	    cyclic factor=array_partition
	#pragma HLS array_partition variable=tmp_profileIndex_j cyclic factor=array_partition
	#pragma HLS array_partition variable=tmp_i_max 	        complete
	#pragma HLS array_partition variable=tmp_i_index_max    complete
	/* ----------------------------------------------------------------------------- */

	// Initialize means value for i=0 (same for all diagonals)
	means_0 = get_float(means[0], 0);

	// Main loop: go through diagonals
	main_loop: for(ITYPE diag = startDiag; diag < endDiag; diag+=VDATA_SIZE)
	{
		// Initialize diagonal coordinates
		i = 0;
		j = diag;

		// Initialization of covariances to 0
		for (int k = 0; k < VDATA_SIZE/2; k++)
		{
			#pragma HLS PIPELINE II=1
			tmp_covariances[k] = 0;
			tmp_covariances[k + VDATA_SIZE / 2] = 0;
		}

		// Burst read of means for current set of diagonals
		index = 0;
		j_offset = j % 16;

		memcpy(buff_A, &means[j / 16], 2112);

		adjust_offset(buff_A,  tmp_means_j, j_offset);

		// i-buffer counter initialization (forces first time filling)
		i_read_counter = VDATA_SIZE - 1;

		// Copy first time series data
		memcpy(buff_A, &(tSeries[j / 16]), 2112);

		// Calculate first covariances for this set of diagonals
		first_cov_main:for (ITYPE w = 0; w < windowSize; w++)
		{
			i_read_counter++;
			j_offset = (j + w) % 16;
			index = 0;

			// if i_read_counter reached VDATA_SIZE update buffer
			if(i_read_counter == VDATA_SIZE)
			{
				i_read_counter = 0;
				memcpy(tmp_tSeries_i, &tSeries[w / 16], sizeof(DTYPE) * VDATA_SIZE);
			}

			if(j_offset == 0)
				memcpy(buff_A, &(tSeries[(j + w) / 16]), 2112);

			adjust_offset(buff_A,  tmp_tSeries_j, j_offset);

			DTYPE tSeries_i_value = get_float(tmp_tSeries_i[i_read_counter / 16], i_read_counter % 16);

			first_cov_calc:for (int k = 0; k < VDATA_SIZE; k++)
			{
				#pragma HLS PIPELINE II=1
				#pragma HLS unroll factor=loop_unrolling
				tmp_covariances[k] += (tmp_tSeries_j[k] - tmp_means_j[k]) * (tSeries_i_value - means_0);
			}
		}

		i_read_counter = VDATA_SIZE - 1;

		// Initialize buffers
		memcpy(buff_A, &df   [j / 16], 2112);
		memcpy(buff_B, &dg   [j / 16], 2112);
		memcpy(buff_C, &norms[j / 16], 2112);


		for(int k = 0; k < VDATA_SIZE; k++)
		{
			#pragma HLS pipeline II=1
			tmp_profile_j[k] = profile[j + k];
 		}

		for(int k = 0; k < VDATA_SIZE; k++)
		{
			#pragma HLS pipeline II=1
			tmp_profileIndex_j[k] = profileIndex[j + k];
 		}

		diag:for (j = diag; j < profileLength; j++)
		{
			i_read_counter++;

			if(i_read_counter == VDATA_SIZE)
			{
				i_read_counter = 0;

				memcpy(tmp_tSeries_i, &norms[i / 16], sizeof(DTYPE) * VDATA_SIZE);

				{
					#pragma HLS loop_merge
					memcpy(tmp_df_i,      &df[i / 16],    sizeof(DTYPE) * VDATA_SIZE);
					memcpy(tmp_dg_i,      &dg[i / 16],    sizeof(DTYPE) * VDATA_SIZE);
				}
			}

			// Calculate j offset
			j_offset = j % 16;

			// Read norms if necessary
			if(j_offset == 0)
			{
			  memcpy(buff_C, &norms[j / 16], 2112);
			}

			adjust_offset(buff_C,  tmp_means_j, j_offset);

			// Obtain norm value for i index
			DTYPE norms_i_value = get_float(tmp_tSeries_i[i_read_counter / 16], i_read_counter % 16);

			// Calculate correlations
			correlations:for (int k = 0; k < VDATA_SIZE; k++)
			{
				#pragma HLS unroll factor=loop_unrolling
				#pragma HLS pipeline II=1
				tmp_correlations[k] = tmp_covariances[k] * norms_i_value * tmp_means_j[k];
			}

			// Obtain df and dg values for i index
			DTYPE df_i_value = get_float(tmp_df_i[i_read_counter / 16], i_read_counter % 16);
			DTYPE dg_i_value = get_float(tmp_dg_i[i_read_counter / 16], i_read_counter % 16);

			// Read df and dg if necessary
			if(j_offset == 0)
			{
				#pragma HLS loop_merge
				memcpy(buff_A, &df[j / 16], 2112);
				memcpy(buff_B, &dg[j / 16], 2112);
			}

			// Adjust offset of df and dg
			adjust_offset(buff_A,  tmp_df_j, j_offset);
			adjust_offset(buff_B,  tmp_dg_j, j_offset);

			// Update covariances
			covariances_update:for(int k = 0; k < VDATA_SIZE; k++)
			{
				#pragma HLS unroll factor=loop_unrolling
				#pragma HLS pipeline II=1
				tmp_covariances[k] += (df_i_value * tmp_dg_j[k]) + (tmp_df_j[k] * dg_i_value);
			}

			// Discard out-of-bound values
			if(j + VDATA_SIZE > profileLength)
			{
				unsigned num_preserve = profileLength - j;
				discard_loop:for(int k = num_preserve; k < VDATA_SIZE; k++)
				{
					#pragma HLS pipeline II=1
					tmp_correlations[k] = -1;
				}
			}

			// Init max i values to -1
			init_i_updates:for(int k = 0; k < 16; k++)
			{
				#pragma HLS pipeline II=1
				tmp_i_max[k] = -1;
			}

			// Calculate both i and j updates
			calculate_updates:for (int k = 0; k < VDATA_SIZE; k += loop_unrolling)
			{
				#pragma HLS pipeline II=1

				for(int kk = 0; kk < loop_unrolling; kk++)
				{
					#pragma HLS unroll
					if(tmp_correlations[k + kk] > tmp_i_max[kk]){
						tmp_i_max[kk] = tmp_correlations[k + kk];
						tmp_i_index_max[kk] = j + k + kk;
					}

					if (tmp_correlations[k + kk] > tmp_profile_j[k+kk])
					{
						tmp_profile_j[k+kk] = tmp_correlations[k+kk];
						tmp_profileIndex_j[k+kk] = i;
					}
				}
			}

			// Update profile j and profileIndex j
			profile[j]      = tmp_profile_j[0];
			profileIndex[j] = tmp_profileIndex_j[0];

			// shift temp profile j one to the left
			shift_profile_j:for(int i =0; i < VDATA_SIZE - 1; i++)
			{
				#pragma HLS unroll
				tmp_profile_j[i] = tmp_profile_j[i+1];
			}

			// shift temp profileIndex j one to the left
			shift_profileIndex_j:for(int i =0; i < VDATA_SIZE - 1; i++)
			{
				#pragma HLS unroll
				tmp_profileIndex_j[i] = tmp_profileIndex_j[i+1];

			}

			// Get new element for temp profile j and temp profileIndex j
			tmp_profile_j[VDATA_SIZE - 1]      = profile[j + VDATA_SIZE];
			tmp_profileIndex_j[VDATA_SIZE - 1] = profileIndex[j + VDATA_SIZE];


			calculate_i_updates:for(int k = 0; k<loop_unrolling; k++)
			{
				#pragma HLS pipeline II=1
				if(tmp_i_max[k] > tmp_i_max[0])
				{
					tmp_i_max[0]       = tmp_i_max[k];
					tmp_i_index_max[0] = tmp_i_index_max[k];
				}

			}

			// Update profile i if necessary
			if(tmp_i_max[0] > profile[i])
			{
				profile[i] = tmp_i_max[0];
				profileIndex[i] = tmp_i_index_max[0];
			}
			i++;
		}
	}
}
}
