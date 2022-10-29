#include"ray_sampler_header.h"
#include <thrust/scan.h>

template <typename TYPE>
__global__ void unpack_data(
	const uint32_t n_rays,
	int n_samples,					
	TYPE *__restrict__ output, 					
	const Vector3f *__restrict__ simpleinfo,
	uint32_t *__restrict__ numsteps_compacted_in
	)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_rays)
	{
		return;
	}
	uint32_t numsteps = numsteps_compacted_in[i * 2 + 0];
	uint32_t base = numsteps_compacted_in[i * 2 + 1];
	if (numsteps == 0)
	{
		return;
	}
	output += i * n_samples * 3;

	uint32_t compacted_numsteps = 0;
	for (; compacted_numsteps < numsteps; ++compacted_numsteps)
	{

		vector_t<TYPE, 3> local_output;
		Vector3f local_simple = simpleinfo[base + compacted_numsteps];
		local_output[0] = local_simple[0];
		local_output[1] = local_simple[1];
		local_output[2] = local_simple[2];
		*(vector_t<TYPE, 3> *)output = local_output;

		output += 3;
	}
}

template <typename TYPE>
__global__ void unpack_info_to_mask(
	const uint32_t n_rays,						//batch total rays number
	int n_samples,	
	TYPE *__restrict__ dL_dws, 	
	uint32_t *__restrict__ numsteps_compacted_in,
	TYPE *__restrict__ dL_loss
	)
{

	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_rays)
	{
		return;
	}

	uint32_t numsteps = numsteps_compacted_in[i * 2 + 0];
	uint32_t base = numsteps_compacted_in[i * 2 + 1];

	if (numsteps == 0)
	{
		return;
	}

	uint32_t compacted_numsteps = 0;
	for (; compacted_numsteps < numsteps; ++compacted_numsteps)
	{

		uint32_t sidx = base + compacted_numsteps;
		dL_dws[sidx] = dL_loss[i * n_samples + compacted_numsteps];
	}

}
