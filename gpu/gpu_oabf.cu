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
#include "gpu_oabf.h"
          

static texture<float, 2, cudaReadModeElementType> texSRC1;
static texture<float4, 2, cudaReadModeElementType> texSRC4;


template<typename T> T texSRC(float x, float y);
template<> inline __device__ float texSRC(float x, float y) { return tex2D(texSRC1, x, y); }
template<> inline __device__ float4 texSRC(float x, float y) { return tex2D(texSRC4, x, y); }


template <typename T>
__global__ void imp_oabf( gpu_plm2<T> dst, const gpu_plm2<float4> lfm, 
                          float sigma_d, float sigma_r, bool tangential, float precision ) 
{
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;

    float4 l = lfm(ix, iy);
    float2 t;
    if (tangential) {
        t = make_float2(l.x, l.y);
        sigma_d *= l.z;
    } else {
        t = make_float2(l.y, -l.x);
        sigma_d *= l.w;
    }

    float twoSigmaD2 = 2.0f * sigma_d * sigma_d;
    float twoSigmaR2 = 2.0f * sigma_r * sigma_r;
    int halfWidth = int(ceilf( precision * sigma_d ));

    float2 tabs = fabs(t);
    float ds = 1.0f / ((tabs.x > tabs.y)? tabs.x : tabs.y);

    T c0 = texSRC<T>(0.5f + ix, 0.5f + iy);
    T sum = c0;

    float norm = 1;
    for (float d = ds; d <= halfWidth; d += ds) {
        float2 dt = d * t;
        T c1 = texSRC<T>(0.5f + ix + dt.x, 0.5f + iy + dt.y);
        T c2 = texSRC<T>(0.5f + ix - dt.x, 0.5f + iy - dt.y);

        T e1 = c1 - c0;
        T e2 = c2 - c0;
        
        float kd = __expf( -dot(d,d) / twoSigmaD2 );
        float kr1 = __expf( -dot(e1,e1) / twoSigmaR2 );
        float kr2 = __expf( -dot(e2,e2) / twoSigmaR2 );
        
        sum += kd * kr1 * c1;
        sum += kd * kr2 * c2;
        norm += kd * kr1 + kd * kr2;
    }
    sum /= norm;

    dst(ix, iy) = sum;
}


gpu_image<float> gpu_oabf_1d( const gpu_image<float>& src, const gpu_image<float4>& lfm, 
                              float sigma_d, float sigma_r, bool tangential, float precision ) 
{
    if (sigma_d <= 0) return src;
    gpu_image<float> dst(src.size());
    bind(&texSRC1, src);
    texSRC1.filterMode = cudaFilterModeLinear;
    imp_oabf<float><<<dst.blocks(), dst.threads()>>>(dst, lfm, sigma_d, sigma_r, tangential, precision);
    texSRC1.filterMode = cudaFilterModePoint;
    GPU_CHECK_ERROR();
    return dst;
}


gpu_image<float4> gpu_oabf_1d( const gpu_image<float4>& src, const gpu_image<float4>& lfm, 
                               float sigma_d, float sigma_r, bool tangential, float precision ) 
{
    if (sigma_d <= 0) return src;
    gpu_image<float4> dst(src.size());
    bind(&texSRC4, src);
    texSRC4.filterMode = cudaFilterModeLinear;
    imp_oabf<float4><<<dst.blocks(), dst.threads()>>>(dst, lfm, sigma_d, sigma_r, tangential, precision);
    texSRC4.filterMode = cudaFilterModePoint;
    GPU_CHECK_ERROR();
    return dst;
}


gpu_image<float> gpu_oabf( const gpu_image<float>& src, const gpu_image<float4>& lfm, 
                           float sigma_d, float sigma_r, float precision ) 
{
    gpu_image<float> img;
    img = gpu_oabf_1d(src, lfm, sigma_d, sigma_r, false, precision);
    img = gpu_oabf_1d(img, lfm, sigma_d, sigma_r, true, precision);
    return img;
}


gpu_image<float4> gpu_oabf( const gpu_image<float4>& src, const gpu_image<float4>& lfm, 
                            float sigma_d, float sigma_r, float precision ) 
{
    gpu_image<float4> img;
    img = gpu_oabf_1d(src, lfm, sigma_d, sigma_r, false, precision);
    img = gpu_oabf_1d(img, lfm, sigma_d, sigma_r, true, precision);
    return img;
}
