#include"ray_sampler_header.h"


template <typename TYPE>
__global__ void prefix_sums_kernel(
	const uint32_t n_rays,
    const Vector3f *__restrict__ sampleinfo,
    uint32_t *__restrict__ numsteps_compacted_in,
    TYPE* __restrict__ ws_inclusive_scan,
    TYPE* __restrict__ ws_exclusive_scan,
    TYPE* __restrict__ wts_inclusive_scan,
    TYPE* __restrict__ wts_exclusive_scan
){
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

	uint32_t sidx = 0;

	float scan_ws = 0.f;
	float scan_wts = 0.f;
	float t = 0.f;
	float ws = 0.f;

	uint32_t compacted_numsteps = 0;
	for (; compacted_numsteps < numsteps; ++compacted_numsteps)
	{
		sidx = base + compacted_numsteps;
		Vector3f local_simple = sampleinfo[sidx];
		ws = local_simple[0];
		// mid t
		// t = (local_simple[1] + local_simple[2]) / 2;
		// t start
		t = local_simple[1];

		// [a0, a1, a2, a3, ...] -> [0, a0, a0+a1, a0+a1+a2, ...]
		ws_exclusive_scan[sidx] = scan_ws;
		wts_exclusive_scan[sidx] = scan_wts;

		scan_ws += ws;
		scan_wts += ws*t;
		// [a0, a1, a2, a3, ...] -> [a0, a0+a1, a0+a1+a2, a0+a1+a2+a3, ...]
		ws_inclusive_scan[sidx] = scan_ws;
		wts_inclusive_scan[sidx] = scan_wts;
		
	}


}

template <typename TYPE>
__global__ void distortion_loss_fw_kernel(
	const uint32_t n_rays,
    const TYPE *__restrict__ _loss,
    uint32_t *__restrict__ numsteps_compacted_in,
    TYPE* __restrict__ loss
){
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

	float loss_sum = 0.f;

	uint32_t compacted_numsteps = 0;
	for (; compacted_numsteps < numsteps; ++compacted_numsteps)
	{
		loss_sum += _loss[base + compacted_numsteps];
		
	}

	loss[i] = loss_sum;


}

template <typename TYPE>
__global__ void distortion_loss_bw_kernel(
	const uint32_t n_rays,
    const TYPE *__restrict__ dL_dloss,
	TYPE *__restrict__ dwloss,
    const float* __restrict__ ws_inclusive_scan,
    const float* __restrict__ wts_inclusive_scan,
	const Vector3f *__restrict__ sampleinfo,
    uint32_t *__restrict__ numsteps_compacted_in
){
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

	float ws_sum = ws_inclusive_scan[base+numsteps-1];
	float wts_sum = wts_inclusive_scan[base+numsteps-1];

	float dL_dloss_local = dL_dloss[i];

	uint32_t sidx = 0;
	float ws = 0.f;
	float t = 0.f;
	float deltas = 0.f;

	uint32_t compacted_numsteps = 0;
	for (; compacted_numsteps < numsteps; ++compacted_numsteps)
	{
		sidx = base + compacted_numsteps;
		Vector3f local_simple = sampleinfo[sidx];
		ws = local_simple[0];
		// mid t
		// t = (local_simple[1] + local_simple[2]) / 2.f;
		// t start
		t = local_simple[1];
		deltas = local_simple[2] - local_simple[1];

		dwloss[sidx] = dL_dloss_local *2.f* (
				(sidx == base? 0.f: 
				(t*ws_inclusive_scan[sidx-1] - wts_inclusive_scan[sidx-1])
			) + (wts_sum-wts_inclusive_scan[sidx] - t*(ws_sum-ws_inclusive_scan[sidx])));

		dwloss[sidx] += dL_dloss_local * 2.f/3.f*ws*deltas;	
	}

}
