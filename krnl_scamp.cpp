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
    This is a simple vector addition example using C++ HLS.
    
*******************************************************************************/

#include "scamp.h"
#include <limits>

#define VDATA_SIZE 32
#define IDATA_SIZE 16

//TRIPCOUNT indentifier
const unsigned int c_dt_size = VDATA_SIZE;

typedef struct v_datatype {
    DTYPE data[VDATA_SIZE];
} v_dt;

typedef struct v_intdatatype {
    ITYPE data[IDATA_SIZE];
} v_intdt;

extern "C" {


void update_covariances(const DTYPE * df, const DTYPE * dg,
		const DTYPE df_i, const DTYPE dg_i, DTYPE * tmp_covariances,
		const ITYPE j)
{
	DTYPE tmp_df_j[VDATA_SIZE];
	DTYPE tmp_dg_j[VDATA_SIZE];

	#pragma HLS array_partition variable=tmp_df_j complete
	#pragma HLS array_partition variable=tmp_dg_j complete


	//#pragma HLS dataflow
	read_df_dg_data:for(int k = 0; k < VDATA_SIZE; k++)
	{
		#pragma HLS PIPELINE II=1
		tmp_df_j[k]    = df[j + k];
		tmp_dg_j[k]    = dg[j + k];
	}

	covariances_update:for(int k = 0; k < VDATA_SIZE; k++)
	{
		#pragma HLS unroll factor=32
		tmp_covariances[k] += (df_i * tmp_dg_j[j + k]) + (tmp_df_j[j + k] * dg_i) ;
	}
}

void update_profile(const DTYPE * tmp_correlations, DTYPE * tmp_profile_j, DTYPE * profile_j, ITYPE j)
{

	//DTYPE tmp_profile_j[VDATA_SIZE];
    #pragma HLS array_partition variable=tmp_profile_j cyclic factor=16

	calculate_updates:for (int k = 0; k < VDATA_SIZE; k++)
	{
			#pragma HLS unroll factor=16
			/*if (tmp_correlations[k] > tmp_profile_i[0])
			{
				tmp_profile_i[0] = tmp_correlations[k];
			}*/

			if (tmp_correlations[k] > tmp_profile_j[k])
			{
				tmp_profile_j[k] = tmp_correlations[k];
			}
		}

		write_back:for (int k = 0; k < VDATA_SIZE; k++)
		{
			#pragma HLS PIPELINE II=1
			//profile_i[i] = tmp_profile_i[0];
			profile_j[j + k] = tmp_profile_j[k];
		}

}

/*void do_diagonal(const DTYPE * norms_0, DTYPE * tmp_covariances, const DTYPE * tmp_norms_i,
		)
{
	DTYPE tmp_correlations[VDATA_SIZE];
	DTYPE tmp_norms_0[VDATA_SIZE];
	DTYPE tmp_profile_0[VDATA_SIZE];

	#pragma HLS array_partition variable=tmp_correlations cyclic factor=16
	#pragma HLS array_partition variable=tmp_norms_0 cyclic factor=16
	#pragma HLS array_partition variable=tmp_profile_0 cyclic factor=16


	#pragma HLS dataflow

	read_norms_data:for(int k = 0; k < VDATA_SIZE; k++)
	{
			#pragma HLS PIPELINE II=1
			tmp_norms_0[k] = norms_0[j + k];
	}

    correlations:for (int k = 0; k < VDATA_SIZE; k++)
	{
		#pragma HLS unroll factor=16
		tmp_correlations[k] = tmp_covariances[k] * tmp_norms_i[i_read_counter] * tmp_norms_0[k];
	}

    update_covariances(df_i, dg_j, tmp_df_i[i_read_counter], tmp_dg_i[i_read_counter], tmp_covariances, j);

	read_profile_data:for(int k = 0; k < VDATA_SIZE; k++)
	{
			#pragma HLS PIPELINE II=1
			tmp_profile_0[k] = profile_0[j + k];
	}

    update_profile(tmp_correlations, tmp_profile_0, profile_0, j);


}*/

void krnl_scamp(const DTYPE *tSeries_i, 	// Time Series input 1
				const DTYPE *tSeries_j, 	// Time Series input 1
				const DTYPE *means,
				const DTYPE *norms_i,
				const DTYPE *norms_j,
				const DTYPE *df_i,
				const DTYPE *df_j,
				const DTYPE *dg_i,
				const DTYPE *dg_j,
				DTYPE *profile_i,
				DTYPE *profile_j,
				ITYPE *profileIndex_i,
				ITYPE *profileIndex_j,
				const ITYPE profileLength,
				const ITYPE numDiagonals,
				const ITYPE windowSize
) {
#pragma HLS INTERFACE m_axi port = tSeries_i offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = tSeries_j offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = means offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = norms_i offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = norms_j offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = df_i offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = df_j offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = dg_i offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = dg_j offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = profile_i offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = profile_j offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = profileIndex_i offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = profileIndex_j offset = slave bundle = gmem1

#pragma HLS INTERFACE s_axilite port = tSeries_i bundle = control
#pragma HLS INTERFACE s_axilite port = tSeries_j bundle = control
#pragma HLS INTERFACE s_axilite port = means bundle = control
#pragma HLS INTERFACE s_axilite port = norms_i bundle = control
#pragma HLS INTERFACE s_axilite port = norms_j bundle = control
#pragma HLS INTERFACE s_axilite port = df_i bundle = control
#pragma HLS INTERFACE s_axilite port = df_j bundle = control
#pragma HLS INTERFACE s_axilite port = dg_i bundle = control
#pragma HLS INTERFACE s_axilite port = dg_j bundle = control
#pragma HLS INTERFACE s_axilite port = profile_i bundle = control
#pragma HLS INTERFACE s_axilite port = profile_j bundle = control
#pragma HLS INTERFACE s_axilite port = profileIndex_i bundle = control
#pragma HLS INTERFACE s_axilite port = profileIndex_j bundle = control
#pragma HLS INTERFACE s_axilite port = profileLength bundle = control
#pragma HLS INTERFACE s_axilite port = numDiagonals bundle = control
#pragma HLS INTERFACE s_axilite port = windowSize bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

//#pragma HLS array_partition variable=tSeries block factor=2
//#pragma HLS array_partition variable=norms block factor=2
//#pragma HLS DATA_PACK variable = tSeries
//#pragma HLS DATA_PACK variable = diagonals
//#pragma HLS DATA_PACK variable = means
//#pragma HLS DATA_PACK variable = norms
//#pragma HLS DATA_PACK variable = df
//#pragma HLS DATA_PACK variable = dg
//#pragma HLS DATA_PACK variable = profile
//#pragma HLS DATA_PACK variable = profileIndex


DTYPE covariance, correlation;
ITYPE i, j;

DTYPE tmp_tSeries_i[VDATA_SIZE];
DTYPE tmp_tSeries_j[VDATA_SIZE];
DTYPE tmp_means_j[VDATA_SIZE];

DTYPE tmp_norms_i[VDATA_SIZE];
DTYPE tmp_norms_j[VDATA_SIZE];

DTYPE tmp_df_i[VDATA_SIZE];
//DTYPE tmp_df_j[VDATA_SIZE];

DTYPE tmp_dg_i[VDATA_SIZE];
//DTYPE tmp_dg_j[VDATA_SIZE];


DTYPE tmp_covariances[VDATA_SIZE];
DTYPE tmp_correlations[VDATA_SIZE];
DTYPE tmp_profile_i[VDATA_SIZE];
DTYPE tmp_profile_j[VDATA_SIZE];


#pragma HLS array_partition variable=tmp_tSeries_i cyclic factor=16
#pragma HLS array_partition variable=tmp_tSeries_j cyclic factor=16

#pragma HLS array_partition variable=tmp_df_i cyclic factor=16
//#pragma HLS array_partition variable=tmp_df_j cyclic factor=16
#pragma HLS array_partition variable=tmp_dg_i cyclic factor=16
//#pragma HLS array_partition variable=tmp_dg_j cyclic factor=16

#pragma HLS array_partition variable=tmp_means_j cyclic factor=16

#pragma HLS array_partition variable=tmp_norms_i cyclic factor=16
#pragma HLS array_partition variable=tmp_norms_j cyclic factor=16

#pragma HLS array_partition variable=tmp_covariances cyclic factor=16

#pragma HLS array_partition variable=tmp_correlations cyclic factor=16
#pragma HLS array_partition variable=tmp_profile_i cyclic factor=16
#pragma HLS array_partition variable=tmp_profile_j cyclic factor=16


DTYPE means_0 = means[0];

// Go through diagonals
main_loop: for(ITYPE diag = windowSize/4 + 1; diag < profileLength; diag+=VDATA_SIZE)
{
  covariance = 0;

  i = 0;
  j = diag;


  // Copy mean of this diag to local variable
 // DTYPE tmp_means_j[VDATA_SIZE]; = means[j];

  for (int k = 0; k < VDATA_SIZE; k++)
  {
		#pragma HLS PIPELINE II=1
    	tmp_covariances[k] = 0;
  }
  for (int k = 0; k < VDATA_SIZE; k++)
  {
		#pragma HLS PIPELINE II=1
	  	tmp_means_j[k] = means[j+k];
  }

  first_cov: for (ITYPE w = 0; w < windowSize; w++)
  {
	  	first_cov_l1:for (int k = 0; k < VDATA_SIZE; k++)
	  	{
	  		#pragma HLS PIPELINE II=1
		  	tmp_tSeries_i[k] = tSeries_i[w];
	  		tmp_tSeries_j[k] = tSeries_j[j + w + k];
	  	}

	  	first_cov_l2:for (int k = 0; k < VDATA_SIZE; k++)
	  	{
			#pragma HLS unroll factor=16
	  		tmp_covariances[k] += ((tmp_tSeries_j[k] - tmp_means_j[k]) * (tmp_tSeries_i[k] - means_0));
	  	}
  }


  unsigned i_read_counter = VDATA_SIZE - 1;

  diag:for (j = diag + 1; j < profileLength; j++)
  {

	 i_read_counter++;

	 if(i_read_counter == VDATA_SIZE)
	 {
		 i_read_counter = 0;

		read_i_data_loop_1:for(int k = 0; k < VDATA_SIZE; k++)
		{
			#pragma HLS PIPELINE II=1
			tmp_norms_i[k] = norms_i[i + k];
			tmp_df_i[k]    = df_j[i + k];
		}
		read_i_data_loop_2:for(int k = 0; k < VDATA_SIZE; k++)
		{
			#pragma HLS PIPELINE II=1
			tmp_profile_i[k] = profile_i[i + k];
			tmp_dg_i[k]      = dg_j[i + k];
		}

	 }
		read_norms_data:for(int k = 0; k < VDATA_SIZE; k++)
		{
				#pragma HLS PIPELINE II=1
				tmp_norms_j[k] = norms_j[j + k];
		}

	    correlations:for (int k = 0; k < VDATA_SIZE; k++)
		{
			#pragma HLS unroll factor=16
			tmp_correlations[k] = tmp_covariances[k] * tmp_norms_i[i_read_counter] * tmp_norms_j[k];
		}

	    update_covariances(df_i, dg_j, tmp_df_i[i_read_counter], tmp_dg_i[i_read_counter], tmp_covariances, j);

		read_profile_data:for(int k = 0; k < VDATA_SIZE; k++)
		{
				#pragma HLS PIPELINE II=1
				tmp_profile_j[k] = profile_j[j + k];
		}

	    update_profile(tmp_correlations, tmp_profile_j, profile_j, j);



    	i++;
	}
  }
 }
}
