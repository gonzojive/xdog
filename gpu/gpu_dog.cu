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
#include "gpu_dog.h"


static texture<float,  2, cudaReadModeElementType> texSRC1;
static texture<float4, 2, cudaReadModeElementType> texSRC4;


__global__ void imp_isotropic_dog( gpu_plm2<float> dst, float sigma, float k, float tau, float precision ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;

    float twoSigmaE2 = 2.0f * sigma * sigma;
    float twoSigmaR2 = twoSigmaE2 * k * k;
    int halfWidth = int(ceilf( precision * sigma * k ));

    float sumE = 0;
    float sumR = 0;
    float2 norm = make_float2(0);

    for ( int i = -halfWidth; i <= halfWidth; ++i ) {
        for ( int j = -halfWidth; j <= halfWidth; ++j ) {
            float d = length(make_float2(i,j));
            float kE = __expf(-d *d / twoSigmaE2);
            float kR = __expf(-d *d / twoSigmaR2);
            
            float c = tex2D(texSRC1, ix + i, iy + j);
            sumE += kE * c;
            sumR += kR * c;
            norm += make_float2(kE, kR);
        }
    }

    sumE /= norm.x;
    sumR /= norm.y;

    float H = sumE - tau * sumR;
    dst(ix, iy) = H;
}


gpu_image<float> gpu_isotropic_dog( const gpu_image<float>& src, float sigma, float k, float tau, float precision ) 
{
    gpu_image<float> dst(src.size());
    bind(&texSRC1, src);
    imp_isotropic_dog<<<dst.blocks(), dst.threads()>>>(dst, sigma, k, tau, precision);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_gradient_dog( gpu_plm2<float> dst, const gpu_plm2<float4> tfab,
                                  float sigma, float k, float tau, float precision)
{
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;

    float4 t = tfab(ix, iy);
    float2 n = make_float2(t.y, -t.x);
    float2 nabs = fabs(n);
    float ds = 1.0f / ((nabs.x > nabs.y)? nabs.x : nabs.y);

    float twoSigmaE2 = 2 * sigma * sigma;
    float twoSigmaR2 = twoSigmaE2 * k * k;
    float halfWidth = ceilf(precision * sigma * k);

    float sumE = tex2D(texSRC1, ix, iy);
    float sumR = sumE;
    float2 norm = make_float2(1, 1);

    for( float d = ds; d <= halfWidth; d += ds ) {
        float kE = __expf( -d * d / twoSigmaE2 ); 
        float kR = __expf( -d * d / twoSigmaR2 );
        
        float2 o = d*n;
        float c = tex2D( texSRC1, 0.5f + ix - o.x, 0.5f + iy - o.y) + 
                  tex2D( texSRC1, 0.5f + ix + o.x, 0.5f + iy + o.y);
        sumE += kE * c;
        sumR += kR * c;
        norm += 2 * make_float2(kE, kR);
    }
    sumE /= norm.x;
    sumR /= norm.y;

    float H = sumE - tau * sumR;
    dst(ix, iy) = H;
}


gpu_image<float> gpu_gradient_dog( const gpu_image<float>& src, const gpu_image<float4>& tfab, 
                                   float sigma, float k, float tau, float precision) 
{
    gpu_image<float> dst(src.size());
    bind(&texSRC1, src);
    texSRC1.filterMode = cudaFilterModeLinear;
    imp_gradient_dog<<<dst.blocks(), dst.threads()>>>(dst, tfab, sigma, k, tau, precision);
    texSRC1.filterMode = cudaFilterModePoint;
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_dog_threshold_tanh( const gpu_plm2<float> src, gpu_plm2<float> dst, float epsilon, float phi )
{
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;

    float H = src(ix, iy);
    float edge = ( H > epsilon )? 1 : 1 + tanh( phi * (H - epsilon));
    dst(ix, iy) = clamp(edge, 0.0f, 1.0f);
}


gpu_image<float> gpu_dog_threshold_tanh( const gpu_image<float>& src, float epsilon, float phi ) {
    if (phi <= 0) 
        return src;

    gpu_image<float> dst(src.size());
    imp_dog_threshold_tanh<<<dst.blocks(), dst.threads()>>>(src, dst, epsilon, phi);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_dog_threshold_smoothstep( const gpu_plm2<float> src, gpu_plm2<float> dst, float epsilon, float phi )
{
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;

    float H = src(ix, iy) - epsilon;
    float edge = ( H > 0 )? 1 : 2 * smoothstep(-2, 2, phi * H );
    dst(ix, iy) = clamp(edge, 0.0f, 1.0f);
}


gpu_image<float> gpu_dog_threshold_smoothstep( const gpu_image<float>& src, float epsilon, float phi ) {
    if (phi <= 0) 
        return src;

    gpu_image<float> dst(src.size());
    imp_dog_threshold_smoothstep<<<dst.blocks(), dst.threads()>>>(src, dst, epsilon, phi);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_dog_colorize( const gpu_plm2<float> src, gpu_plm2<float4> dst )
{
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h)
        return;

    float H = clamp(src(ix, iy), -1.0f, 1.0f);
    dst(ix, iy) = make_float4( (H < 0)? -H : 0, (H > 0)? H : 0, 0, 1);
}


gpu_image<float4> gpu_dog_colorize( const gpu_image<float>& src ) {
    gpu_image<float4> dst(src.size());
    imp_dog_colorize<<<dst.blocks(), dst.threads()>>>(src, dst);
    GPU_CHECK_ERROR();
    return dst;
}
