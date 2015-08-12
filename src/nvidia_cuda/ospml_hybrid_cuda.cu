/*
    Copyright 2014-2015 Dake Feng, Peri LLC, dakefeng@gmail.com
    This file is part of TomograPeri.
    TomograPeri is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    TomograPeri is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with TomograPeri.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
 * V.1.1.1 8_3_2015
 */
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include "utils_cuda.cuh"


void ospml_hybrid_forEachSlice( CCudaData *pData, int s)
{
	float mov;
    int subset_ind1, subset_ind2, os_total=pData->host_num_block;

	preprocessing_cuda(pData->host_ngridx, pData->host_ngridy, pData->host_dz, pData->host_center[s],
        &mov, pData->dev_gridx, pData->dev_gridy); // Outputs: mov, gridx, gridy

    subset_ind1 = pData->host_dx/pData->host_num_block;
    subset_ind2 = subset_ind1;
	if (pData->host_dx%pData->host_num_block!=0)
		os_total++;

    // For each ordered-subset num_subset
    for (int os=0; os<pData->host_num_block+1; os++) 
    {
        if (os == pData->host_num_block) 
        {
            subset_ind2 = pData->host_dx%pData->host_num_block;
        }

		pData->cleanEFGS();
		if(subset_ind2!=0)
		{
			ospml_forEachProjectionPixelCUDA(pData,s,mov,subset_ind1,subset_ind2,os);
		}
		hybrid_weightProjection(pData,s);

		updateReconCUDA(pData, s);
	}
}

void ospml_hybrid_forEachIter( CCudaData *pData)
{
	pData->cleanSimdata();

	for (int s=0; s<pData->host_dy; s++)
		ospml_hybrid_forEachSlice(pData ,s);
}


void
ospml_hybrid_cuda(
    float *data, const int dx, const int dy, const int dz, float *center, float *theta,
    float *recon, const int ngridx, const int ngridy, const int num_iter, float *reg_pars, int num_block,
    float *ind_block)
{
	CCudaData *pData=new CCudaData(data, dx, dy, dz, center, theta,
									recon, ngridx, ngridy, num_iter, reg_pars, num_block, ind_block);

	// For each slice
    for (int i=0; i<num_iter; i++)
        ospml_hybrid_forEachIter(pData);

	// Retrieve recon result
	pData->retrieveRecon();

	delete pData;
}
