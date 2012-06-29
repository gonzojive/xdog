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
#include "gpu_bilateral.h"


static texture<float, 2, cudaReadModeElementType> texSRC1;
static texture<float4, 2, cudaReadModeElementType> texSRC4;

template<typename T> T texSRC(float x, float y);
template<> inline __device__ float texSRC(float x, float y) { return tex2D(texSRC1, x, y); }
template<> inline __device__ float4 texSRC(float x, float y) { return tex2D(texSRC4, x, y); }


template <typename T>
__global__ void imp_bilateral_filter( gpu_plm2<T> dst, float sigma_d, float sigma_r, float precision ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;

    float twoSigmaD2 = 2.0f * sigma_d * sigma_d;
    float twoSigmaR2 = 2.0f * sigma_r * sigma_r;
    int halfWidth = int(ceilf( precision * sigma_d ));

    T c0 = texSRC<T>(ix, iy);
    T sum = make_zero<T>();

    float norm = 0.0f;
    for ( int i = -halfWidth; i <= halfWidth; ++i ) {
        for ( int j = -halfWidth; j <= halfWidth; ++j ) {
            float d = length(make_float2(i,j));

            T c = texSRC<T>(ix + i, iy + j);
            T e = c - c0;
            
            float kd = __expf( -dot(d,d) / twoSigmaD2 );
            float kr = __expf( -dot(e,e) / twoSigmaR2 );
            
            sum += kd * kr * c;
            norm += kd * kr;
        }
    }
    sum /= norm;
    
    dst(ix, iy) = sum;
}


gpu_image<float> gpu_bilateral_filter(const gpu_image<float>& src, float sigma_d, float sigma_r, float precision ) {
    if (sigma_d <= 0) return src;
    gpu_image<float> dst(src.size());
    bind(&texSRC1, src);
    imp_bilateral_filter<float><<<dst.blocks(), dst.threads()>>>(dst, sigma_d, sigma_r, precision);
    GPU_CHECK_ERROR();
    return dst;
}


gpu_image<float4> gpu_bilateral_filter(const gpu_image<float4>& src, float sigma_d, float sigma_r, float precision ) {
    if (sigma_d < 0) return src;
    gpu_image<float4> dst(src.size());
    bind(&texSRC4, src);
    imp_bilateral_filter<float4><<<dst.blocks(), dst.threads()>>>(dst, sigma_d, sigma_r, precision);
    GPU_CHECK_ERROR();
    return dst;
}


template<typename T, int dx, int dy> 
__global__ void imp_bilateral_filter_xy( gpu_plm2<T> dst, float sigma_d, float sigma_r, float precision ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;

    float twoSigmaD2 = 2 * sigma_d * sigma_d;
    float twoSigmaR2 = 2 * sigma_r * sigma_r;
    int halfWidth = int(ceilf( precision * sigma_d ));

    T c0 = texSRC<T>(ix, iy);
    T sum = make_zero<T>();

    float norm = 0;
    for ( int i = -halfWidth; i <= halfWidth; ++i ) {
        T c = texSRC<T>(ix + dx * i, iy + dy * i);
        T e = c - c0;
        
        float kd = __expf( -i * i / twoSigmaD2 );
        float kr = __expf( -dot(e,e) / twoSigmaR2 );
        
        sum += kd * kr * c;
        norm += kd * kr;
    }
    sum /=  norm;
    
    dst(ix, iy) = sum;
}


gpu_image<float> gpu_bilateral_filter_xy( const gpu_image<float>& src, float sigma_d, float sigma_r, float precision ) {
    if (sigma_d <= 0) return src;
    gpu_image<float> dst(src.size());
    gpu_image<float> tmp(src.size());
    bind(&texSRC1, src);
    imp_bilateral_filter_xy<float,1,0><<<tmp.blocks(), tmp.threads()>>>(tmp, sigma_d, sigma_r, precision);
    GPU_CHECK_ERROR();
    bind(&texSRC1, tmp);
    imp_bilateral_filter_xy<float,0,1><<<dst.blocks(), dst.threads()>>>(dst, sigma_d, sigma_r, precision);
    GPU_CHECK_ERROR();
    return dst;
}


gpu_image<float4> gpu_bilateral_filter_xy( const gpu_image<float4>& src, float sigma_d, float sigma_r, float precision ) {
    if (sigma_d <= 0) return src;
    gpu_image<float4> dst(src.size());
    gpu_image<float4> tmp(src.size());
    bind(&texSRC4, src);
    imp_bilateral_filter_xy<float4,1,0><<<tmp.blocks(), tmp.threads()>>>(tmp, sigma_d, sigma_r, precision);
    GPU_CHECK_ERROR();
    bind(&texSRC4, tmp);
    imp_bilateral_filter_xy<float4,0,1><<<dst.blocks(), tmp.threads()>>>(dst, sigma_d, sigma_r, precision);
    GPU_CHECK_ERROR();
    return dst;
}
