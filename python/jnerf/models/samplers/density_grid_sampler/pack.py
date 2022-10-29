import os
import copy
import pathlib
import jittor as jt
import jnerf
from jittor import Function, exp, log
import numpy as np
from jnerf.ops.code_ops.global_vars import global_headers,proj_options
jt.flags.use_cuda = 1

class Pack(Function):
    def __init__(self, density_grad_header, using_fp16=False):
        self.density_grad_header = density_grad_header
        if using_fp16:
            self.grad_type = 'float16'
        else:
            self.grad_type = 'float32'

    def execute(self, rays_numsteps_compacted, ws, t1, t2):
        self.rays_numsteps_compacted = rays_numsteps_compacted.detach()
        # self.ws = ws.detach()
        # self.t1 = t1.detach()
        # self.t2 = t2.detach()
        samplesinfo = jt.stack([ws[: None], t1[: None], t2[: None]], -1)
        self.num_elements = ws.shape[0]
        self.n_samples = int(rays_numsteps_compacted[:, 0].max().item())
        self.n_rays = rays_numsteps_compacted.shape[0]
        output= jt.code((self.n_rays, self.n_samples, 3), 'float32',
            inputs=[samplesinfo, rays_numsteps_compacted], 
        cuda_header=global_headers+self.density_grad_header+'#include "pack.h"', cuda_src=f"""
        #define grad_t out0_type
        @alias(simpleinfo, in0)
        @alias(output, out0)
        @alias(rays_numsteps_compacted, in1)
        cudaStream_t stream=0;

        cudaMemsetAsync(output_p, 0, output->size);
    
     
        const int n_samples={self.n_samples};
        const uint32_t n_rays=rays_numsteps_compacted_shape0;
        linear_kernel(unpack_data<grad_t>, 0, stream,
            n_rays, n_samples, (grad_t*)output_p, (Vector3f*)simpleinfo_p, (uint32_t*)rays_numsteps_compacted_p);   
""")

        output.compile_options = proj_options
        # output.sync()
        ws = ws.detach()
        t1 = t1.detach()
        t2 = t2.detach()

        return output[:, :, 0].detach(), output[:, :, 1].detach(), output[:, :, 2].detach()

    def grad(self, grad_x, grad_y, grad_z):

        # print("grad_x", grad_x)
        # grad_x = jt.zeros((self.num_elements, 3), 'float32')
        # new_grad_x = jt.zeros((self.num_elements, 3), 'float32')
        mask = jt.code((self.num_elements, ), 'float32',
                        inputs=[self.rays_numsteps_compacted, grad_x.float()], 
                        cuda_header=global_headers+self.density_grad_header+'#include "pack.h"', cuda_src=f"""
        #define grad_t out0_type
        @alias(mask,out0)
        @alias(rays_numsteps_compacted,in0)
        @alias(grad_x,in1)

        cudaStream_t stream=0;
    
        cudaMemsetAsync(out0_p, 0, out0->size);

        const int n_samples={self.n_samples};
        const uint32_t n_rays=rays_numsteps_compacted_shape0;
        linear_kernel(unpack_info_to_mask<grad_t>, 0,stream,
            n_rays, n_samples, (grad_t*)mask_p, (uint32_t*)rays_numsteps_compacted_p, (grad_t*)grad_x_p);   

""")

        mask.compile_options=proj_options
        mask.sync()
        # print(mask.sum() == self.num_elements)
        # print(grad_x[mask].shape)
        # new_grad_x = jt.zeros((self.num_elements, 3), 'float32')
        return None, mask.detach(), None, None