//
// by Jan Eric Kyprianidis <www.kyprianidis.com>
// Copyright (C) 2010-2012 Computer Graphics Systems Group at the
// Hasso-Plattner-Institut, Potsdam, Germany <www.hpi3d.de>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
#include "gpu_minmax.h"
#include "gpu_util.h"
#include "gpu_color.h"
#include <cfloat>


template <unsigned nThreads, unsigned nBlocks>
__global__ void impl_minmax(float *dst, const gpu_plm2<float> src) {
    __shared__ float sdata[2 * nThreads];

    unsigned int tmin = threadIdx.x;
    unsigned int tmax = threadIdx.x + nThreads;
    float myMin = FLT_MAX;
    float myMax = -FLT_MAX;

    unsigned o = blockIdx.x * src.stride;
    while (o < src.h * src.stride) {
        unsigned int i = threadIdx.x;
        while (i < src.w) {
            volatile float v = src.ptr[o+i];
            myMin = fminf(myMin, v);
            myMax = fmaxf(myMax, v);
            i += nThreads;
        } 
        o += nBlocks * src.stride;
    }

    sdata[tmin] = myMin;
    sdata[tmax] = myMax;
    __syncthreads();

    if (nThreads >= 512) { if (tmin < 256) { sdata[tmin] = myMin = fminf(myMin, sdata[tmin + 256]); 
                                             sdata[tmax] = myMax = fmaxf(myMax, sdata[tmax + 256]); } __syncthreads(); }
    if (nThreads >= 256) { if (tmin < 128) { sdata[tmin] = myMin = fminf(myMin, sdata[tmin + 128]);
                                             sdata[tmax] = myMax = fmaxf(myMax, sdata[tmax + 128]); } __syncthreads(); }
    if (nThreads >= 128) { if (tmin <  64) { sdata[tmin] = myMin = fminf(myMin, sdata[tmin +  64]); 
                                             sdata[tmax] = myMax = fmaxf(myMax, sdata[tmax +  64]); } __syncthreads(); }
    
    if (tmin < 32) {
        volatile float* smem = sdata;
        if (nThreads >=  64) { smem[tmin] = myMin = fminf(myMin, smem[tmin + 32]); 
                               smem[tmax] = myMax = fmaxf(myMax, smem[tmax + 32]); }
        if (nThreads >=  32) { smem[tmin] = myMin = fminf(myMin, smem[tmin + 16]); 
                               smem[tmax] = myMax = fmaxf(myMax, smem[tmax + 16]); }
        if (nThreads >=  16) { smem[tmin] = myMin = fminf(myMin, smem[tmin +  8]); 
                               smem[tmax] = myMax = fmaxf(myMax, smem[tmax +  8]); }
        if (nThreads >=   8) { smem[tmin] = myMin = fminf(myMin, smem[tmin +  4]); 
                               smem[tmax] = myMax = fmaxf(myMax, smem[tmax +  4]); }
        if (nThreads >=   4) { smem[tmin] = myMin = fminf(myMin, smem[tmin +  2]); 
                               smem[tmax] = myMax = fmaxf(myMax, smem[tmax +  2]); }
        if (nThreads >=   2) { smem[tmin] = myMin = fminf(myMin, smem[tmin +  1]); 
                               smem[tmax] = myMax = fmaxf(myMax, smem[tmax +  1]); }
    }
    
    if (tmin == 0) {
        dst[blockIdx.x] = sdata[0];
        dst[blockIdx.x + nBlocks] = sdata[0 + nThreads];
    }
}


void gpu_minmax(const gpu_image<float>& src, float *pmin, float *pmax) {
    const unsigned nBlocks = 64;
    const unsigned nThreads = 128;

    static float *dst_gpu = 0;
    static float *dst_cpu = 0;
    if (!dst_cpu) {
        cudaMalloc(&dst_gpu, 2 * sizeof(float)*nBlocks);
        cudaMallocHost(&dst_cpu, 2 * sizeof(float)*nBlocks, cudaHostAllocPortable);
    }
    
    dim3 dimBlock(nThreads, 1, 1);
    dim3 dimGrid(nBlocks, 1, 1);
    impl_minmax<nThreads, nBlocks><<< dimGrid, dimBlock >>>(dst_gpu, src);
    cudaMemcpy(dst_cpu, dst_gpu, 2*sizeof(float)*nBlocks, cudaMemcpyDeviceToHost);

    if (pmin) {
        float m = dst_cpu[0];
        for (int i = 1; i < nBlocks; ++i) m = fminf(m, dst_cpu[i]);
        *pmin = m;
    }   
    if (pmax) {
        float m = dst_cpu[nBlocks];
        for (int i = 1; i < nBlocks; ++i) m = fmaxf(m, dst_cpu[nBlocks+i]);
        *pmax = m;
    }   
}


float gpu_min( const gpu_image<float>& src ) {
    float m;
    gpu_minmax(src, &m, NULL);
    return m;
}


float gpu_max( const gpu_image<float>& src ) {
    float m;
    gpu_minmax(src, NULL, &m);
    return m;
}


gpu_image<float> gpu_normalize(const gpu_image<float>& src) {
    float pmin, pmax;
    gpu_minmax(src, &pmin, &pmax);
    return gpu_adjust(src, 1.0f / (pmax - pmin), -pmin / (pmax - pmin));
}


gpu_image<float4> gpu_normalize_gray(const gpu_image<float4>& src) {
    float pmin, pmax;
    gpu_image<float> L = gpu_rgb2gray(src);
    gpu_minmax(L, &pmin, &pmax);
    float a = 1.0f / (pmax - pmin);
    float b = -pmin / (pmax - pmin);
    return gpu_adjust(src, make_float4(a,a,a,1), make_float4(b,b,b,0));
}
