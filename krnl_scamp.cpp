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

#define VDATA_SIZE 64
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



void calculate_cov(const DTYPE * tSeries_i, const DTYPE * tSeries_j, DTYPE * covariance, DTYPE tmpMeans1, DTYPE means_0, ITYPE w, ITYPE j)
{
	DTYPE tmpVect1[VDATA_SIZE];
	DTYPE tmpVect2[VDATA_SIZE];
	DTYPE tmpVect3[VDATA_SIZE];
	DTYPE tmpVect4[VDATA_SIZE];

	DTYPE tmp_cov = *covariance;

	#pragma HLS dataflow

	cc_loop1:for (int k = 0; k < VDATA_SIZE; k++)
	{
		#pragma HLS PIPELINE II=1
		tmpVect1[k] = tSeries_j[j + w + k];
		//tmpVect2[k] = tSeries[w + k];
	}

	cc_loop2:for (int k = 0; k < VDATA_SIZE; k++)
	{
		#pragma HLS PIPELINE II=1
		tmpVect2[k] = tSeries_i[w + k];
		tmpVect3[k] = tmpVect1[k];
	}


	cc_loop3:for (int k = 0; k < VDATA_SIZE; k+=4)
	{
		//#pragma HLS PIPELINE II=1
		//covariance += ((tSeries[diag + w].data[0] - means[diag].data[0]) * (tSeries[w].data[0] - means[0].data[0]));
		tmpVect4[k] = ((tmpVect3[k] - tmpMeans1) * (tmpVect2[k] - means_0));
		tmpVect4[k+1] = ((tmpVect3[k+1] - tmpMeans1) * (tmpVect2[k+1] - means_0));
		tmpVect4[k+2] = ((tmpVect3[k+2] - tmpMeans1) * (tmpVect2[k+2] - means_0));
		tmpVect4[k+3] = ((tmpVect3[k+3] - tmpMeans1) * (tmpVect2[k+3] - means_0));
	}

	cc_loop4:for (int k = 0; k < VDATA_SIZE; k++)
	{
		//#pragma HLS PIPELINE II=1
		//covariance += ((tSeries[diag + w].data[0] - means[diag].data[0]) * (tSeries[w].data[0] - means[0].data[0]));
		tmp_cov += tmpVect4[k];
	}

	*covariance = tmp_cov;
}




void krnl_scamp(const DTYPE *tSeries_i, 	// Time Series input 1
				const DTYPE *tSeries_j, 	// Time Series input 1
		       const ITYPE *diagonals,
               const DTYPE *means,
			   const DTYPE *norms,
			   const DTYPE *df,
			   const DTYPE *dg,
               DTYPE *profile,
			   ITYPE *profileIndex,
			   const ITYPE profileLength,
			   const ITYPE numDiagonals,
			   const ITYPE windowSize
) {
#pragma HLS INTERFACE m_axi port = tSeries_i offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = tSeries_j offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = diagonals offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = means offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = norms offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = df offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = dg offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = profile offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = profileIndex offset = slave bundle = gmem

#pragma HLS INTERFACE s_axilite port = tSeries_i bundle = control
#pragma HLS INTERFACE s_axilite port = tSeries_j bundle = control
#pragma HLS INTERFACE s_axilite port = means bundle = control
#pragma HLS INTERFACE s_axilite port = norms bundle = control
#pragma HLS INTERFACE s_axilite port = df bundle = control
#pragma HLS INTERFACE s_axilite port = dg bundle = control
#pragma HLS INTERFACE s_axilite port = diagonals bundle = control
#pragma HLS INTERFACE s_axilite port = profile bundle = control
#pragma HLS INTERFACE s_axilite port = profileIndex bundle = control
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



DTYPE means_0 = means[0];

// Go through diagonals
main_loop: for (ITYPE diag = windowSize/4 + 1; diag < profileLength; diag++)
{
  covariance = 0;

  i = 0;
  j = diag;

  DTYPE tmpMeans1 = means[j];


  dotp: for (ITYPE w = 0; w < windowSize; w+=VDATA_SIZE)
  {
	  calculate_cov(tSeries_i, tSeries_j, &covariance,  tmpMeans1,  means_0,  w,  j);
  }

  correlation = covariance * norms[i] * norms[j];

  if (correlation > profile[0])
  {
	profile[0] = correlation;
	profileIndex[0] = j;
  }
  if (correlation > profile[j])
  {
	profile[j] = correlation;
	profileIndex[j] = i;
  }
/*
  i++;

  j=diag+1;
  diag:for (j = diag + 1; j < profileLength; j+= VDATA_SIZE)
  {
	norms_i_r:for (int k = 0; k < VDATA_SIZE; k++)
	{
		#pragma HLS PIPELINE II=1
		tmpVect1[k] = norms[i + k];
	}
	norms_j_r:for (int k = 0; k < VDATA_SIZE; k++)
	{
		#pragma HLS PIPELINE II=1
		tmpVect2[k] = norms[j + k];
	}

	covariance:for (int k = 0; k < VDATA_SIZE; k++)
	{
		#pragma HLS unroll factor=64
		tmpVect3[k] = covariance * tmpVect1[k] * tmpVect2[k];
	}

	profile_i_r:for (int k = 0; k < VDATA_SIZE; k++)
	{
		#pragma HLS PIPELINE II=1
		tmpVect1[k] = profile[i + k];
	}

	profile_j_r:for (int k = 0; k < VDATA_SIZE; k++)
	{
		#pragma HLS PIPELINE II=1
		tmpVect2[k] = profile[j + k];
	}


	update:for (int k = 0; k < VDATA_SIZE; k++)
	{
		#pragma HLS unroll factor=64
		if (tmpVect3[k] > tmpVect1[k])
		{
		  tmpVect1[k] = tmpVect3[k];
		  //profileIndex[i + k] = j + k;
		}

		if (tmpVect3[k] > tmpVect2[k])
		{
			tmpVect2[k] = tmpVect3[k];
		  //profileIndex[j + k] = i + k;
		}
	}

	profile_i_w:for (int k = 0; k < VDATA_SIZE; k++)
	{
		#pragma HLS PIPELINE II=1
		profile[i + k] = tmpVect1[k];
	}

	profile_j_w:for (int k = 0; k < VDATA_SIZE; k++)
	{
		#pragma HLS PIPELINE II=1
		profile[j + k] = tmpVect2[k];
	}

	i+=VDATA_SIZE;

	 }*/
  /*i = 1;


  for (j = diag + 1; j < profileLength; j++)
  {
	covariance += (df[i - 1].data[0] * dg[j - 1].data[0] + df[j - 1].data[0] * dg[i - 1].data[0]);
	correlation = covariance * norms[i].data[0] * norms[j].data[0];

	if (correlation > profile[i].data[0])
	{
	  profile[i].data[0] = correlation;
	  profileIndex[i].data[0] = j;
	}

	if (correlation > profile[j].data[0])
	{
	  profile[j].data[0] = correlation;
	  profileIndex[j].data[0] = i;
	}
	i++;
  }*/
}
}
}
