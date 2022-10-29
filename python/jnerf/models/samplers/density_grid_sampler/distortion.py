import os
import copy
import pathlib
import jittor as jt
import jnerf
from jittor import Function, exp, log
import numpy as np
from jnerf.ops.code_ops.global_vars import global_headers,proj_options
jt.flags.use_cuda = 1

class Distortion(Function):
    def __init__(self, density_grad_header, using_fp16=False):
        self.density_grad_header = density_grad_header
        if using_fp16:
            self.grad_type = 'float16'
        else:
            self.grad_type = 'float32'

    def execute(self, rays_numsteps_compacted, ws, t1, t2):
        self.rays_numsteps_compacted = rays_numsteps_compacted.detach()
        self.sampleinfo = jt.stack([ws[: None], t1[: None], t2[: None]], -1).detach()
        self.num_elements = self.sampleinfo.shape[0]
        self.n_rays = rays_numsteps_compacted.shape[0]

        ws_inclusive_scan = jt.empty((self.num_elements,), 'float32')
        ws_exclusive_scan = jt.empty((self.num_elements,), 'float32')
        wts_inclusive_scan = jt.empty((self.num_elements,), 'float32')
        wts_exclusive_scan = jt.empty((self.num_elements,), 'float32')

        ws_inclusive_scan, ws_exclusive_scan, wts_inclusive_scan, wts_exclusive_scan = jt.code(inputs=[self.sampleinfo, self.rays_numsteps_compacted], outputs=[ws_inclusive_scan, ws_exclusive_scan, wts_inclusive_scan, wts_exclusive_scan, ], 
        cuda_header=global_headers+self.density_grad_header+'#include "distortion.h"', cuda_src=f"""
        #define grad_t out0_type
        @alias(sampleinfo, in0)
        @alias(rays_numsteps_compacted, in1)
        @alias(ws_inclusive_scan, out0)
        @alias(ws_exclusive_scan, out1)
        @alias(wts_inclusive_scan, out2)
        @alias(wts_exclusive_scan, out3)
        cudaStream_t stream=0;

        cudaMemsetAsync(out0_p, 0, out0->size);
        cudaMemsetAsync(out1_p, 0, out0->size);
        cudaMemsetAsync(out2_p, 0, out0->size);
        cudaMemsetAsync(out3_p, 0, out0->size);
    
        const uint32_t n_rays=rays_numsteps_compacted_shape0;
        linear_kernel(prefix_sums_kernel<grad_t>, 0, stream,
            n_rays, (Vector3f*)sampleinfo_p, (uint32_t*)rays_numsteps_compacted_p, (grad_t*)ws_inclusive_scan_p, (grad_t*)ws_exclusive_scan_p, (grad_t*)wts_inclusive_scan_p, (grad_t*)wts_exclusive_scan_p);   
""")

        ws_inclusive_scan.compile_options = proj_options
        ws_inclusive_scan.sync()
        ws_exclusive_scan.sync()
        wts_inclusive_scan.sync()
        wts_exclusive_scan.sync()

        self.ws_inclusive_scan = ws_inclusive_scan.detach()
        ws_exclusive_scan = ws_exclusive_scan.detach()
        self.wts_inclusive_scan = wts_inclusive_scan.detach()
        wts_exclusive_scan = wts_exclusive_scan.detach()

        ws = self.sampleinfo[:, 0]
        # print(ws.min())
        deltas = self.sampleinfo[:, 2] - self.sampleinfo[:, 1]
        # print(deltas.min())
        # _loss_1 = 1.0/3.0*ws*ws*deltas
        _loss = 2.0*(
            (self.wts_inclusive_scan*ws_exclusive_scan)-(self.ws_inclusive_scan*wts_exclusive_scan)
        ) + 1.0/3.0*deltas*ws.pow(2)

        # print(_loss_1.min())

        loss = jt.code((self.n_rays, ), 'float32', inputs=[_loss.detach(), rays_numsteps_compacted], 
        cuda_header=global_headers+self.density_grad_header+'#include "distortion.h"', cuda_src=f"""
        #define grad_t out0_type
        @alias(_loss, in0)
        @alias(rays_numsteps_compacted, in1)
        @alias(loss, out0)
        cudaStream_t stream=0;

        cudaMemsetAsync(out0_p, 0, out0->size);
     
        const uint32_t n_rays=rays_numsteps_compacted_shape0;
        linear_kernel(distortion_loss_fw_kernel<grad_t>, 0, stream,
            n_rays, (grad_t*)_loss_p, (uint32_t*)rays_numsteps_compacted_p, (grad_t*)loss_p);   
""")

        loss.compile_options = proj_options
        loss.sync()

        return loss.detach()

    def grad(self, grad_x):
        # print("grad_x", grad_x)
        # grad_x = jt.zeros((self.num_elesments, 3), 'float32')
        dw_loss = jt.code((self.num_elements,), 'float32',
                        inputs=[grad_x, self.ws_inclusive_scan, self.wts_inclusive_scan, self.sampleinfo, self.rays_numsteps_compacted], 
                        cuda_header=global_headers+self.density_grad_header+'#include "distortion.h"', cuda_src=f"""
        #define grad_t out0_type
        @alias(dw_loss,out0)
        @alias(dL_dloss,in0)
        @alias(ws_inclusive_scan,in1)
        @alias(wts_inclusive_scan,in2)
        @alias(sampleinfo,in3)
        @alias(rays_numsteps_compacted,in4)

        cudaStream_t stream=0;
    
        cudaMemsetAsync(out0_p, 0, out0->size);

        const uint32_t n_rays=rays_numsteps_compacted_shape0;
        linear_kernel(distortion_loss_bw_kernel<grad_t>, 0,stream,
            n_rays, (grad_t*)dL_dloss_p, (grad_t*)dw_loss_p, (float*)ws_inclusive_scan_p, (float*)wts_inclusive_scan_p , (Vector3f*)sampleinfo_p, (uint32_t*)rays_numsteps_compacted_p);   

""")

        dw_loss.compile_options=proj_options
        dw_loss.sync()
        return None, dw_loss, None, None