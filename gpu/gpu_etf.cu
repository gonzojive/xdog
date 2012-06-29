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
#include "gpu_etf.h"
#include "gpu_minmax.h"


texture<float,  2, cudaReadModeElementType> texSRC1;
texture<float2, 2, cudaReadModeElementType> texSRC2;


__global__ void imp_etf_sobel( gpu_plm2<float2> dst0, gpu_plm2<float> dst1 ) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if(ix >= dst0.w || iy >= dst0.h) 
        return;

    float2 g;
    g.x = (
          -0.183f * tex2D(texSRC1, ix-1, iy-1) +
          -0.634f * tex2D(texSRC1, ix-1, iy) + 
          -0.183f * tex2D(texSRC1, ix-1, iy+1) +
          +0.183f * tex2D(texSRC1, ix+1, iy-1) +
          +0.634f * tex2D(texSRC1, ix+1, iy) + 
          +0.183f * tex2D(texSRC1, ix+1, iy+1)
          ) * 0.5f;

    g.y = (
          -0.183f * tex2D(texSRC1, ix-1, iy-1) + 
          -0.634f * tex2D(texSRC1, ix,   iy-1) + 
          -0.183f * tex2D(texSRC1, ix+1, iy-1) +
          +0.183f * tex2D(texSRC1, ix-1, iy+1) +
          +0.634f * tex2D(texSRC1, ix,   iy+1) + 
          +0.183f * tex2D(texSRC1, ix+1, iy+1)
          ) * 0.5f;

    float len = length(g);
    if (len > 0)
        g /= len;

    dst0(ix, iy) = make_float2(-g.y, g.x);
    dst1(ix, iy) = len;
}


__global__ void imp_etf_smooth_full( gpu_plm2<float2> dst, float sigma, float precision, bool gaussian ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h) 
        return;

    int halfWidth = int(ceilf( precision * sigma ));

    float2 p0 = tex2D(texSRC2, ix, iy);
    float z0 = tex2D(texSRC1, ix, iy);
    float2 g = make_float2(0);

    for ( int i = -halfWidth; i <= halfWidth; ++i ) {
        for ( int j = -halfWidth; j <= halfWidth; ++j ) {
            float d = length(make_float2(i,j));
            if (d <= halfWidth) {
                float2 p = tex2D(texSRC2, ix + i, iy + j);
                float z = tex2D(texSRC1, ix + i, iy + j);
                
                float wm = 0.5f * (z - z0 + 1);
                float wd = dot(p0, p);
                float w = wm * wd;

                if (gaussian) w *= __expf( -0.5f * d *d / sigma / sigma );

                g += w * p;
            }
        }
    }
    
    float len = length(g);
    if (len > 0)
        g /= len;
    
    dst(ix, iy) = make_float2(g.x, g.y);
}


template<int dx, int dy> 
__global__ void imp_etf_smooth_xy( gpu_plm2<float2> dst, float sigma, float precision ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h) 
        return;

    int halfWidth = int(ceilf( precision * sigma ));

    float2 p0 = tex2D(texSRC2, ix, iy);
    float z0 = tex2D(texSRC1, ix, iy);
    float2 g = make_float2(0);

    for ( int i = -halfWidth; i <= halfWidth; ++i ) {
        float2 p = tex2D(texSRC2, ix + dx * i, iy + dy * i);
        float z = tex2D(texSRC1, ix + dx * i, iy + dy * i);
        
        float wm = 0.5f * (z - z0 + 1);
        float wd = dot(p0, p);
        float w = wm * wd;

        g += w * p;
    }
    
    float len = length(g);
    if (len > 0)
        g /= len;
    
    dst(ix, iy) = make_float2(g.x, g.y);
}


gpu_image<float2> gpu_etf_full( const gpu_image<float>& src, float sigma, int N, 
                                float precision, bool gaussian ) 
{
    gpu_image<float2> etf(src.size());
    gpu_image<float> mag(src.size());
    {
        bind(&texSRC1, src);
        imp_etf_sobel<<<src.blocks(), src.threads()>>>(etf, mag);
        GPU_CHECK_ERROR();

        float pmax;
        gpu_minmax(mag, 0, &pmax);
        if (pmax > 0) mag = gpu_mul(mag, 1 / pmax);
    }

    bind(&texSRC1, mag);
    for (int k = 0; k < N; ++k) {
        gpu_image<float2> tmp(src.size());
        bind(&texSRC2, etf);
        imp_etf_smooth_full<<<tmp.blocks(), tmp.threads()>>>(tmp, sigma, precision, gaussian);
        GPU_CHECK_ERROR();
        etf = tmp;
    }

    return etf;
}


gpu_image<float2> gpu_etf_xy(const gpu_image<float>& src, float sigma, int N, float precision) {
    gpu_image<float2> etf(src.size());
    gpu_image<float> mag(src.size());
    {
        bind(&texSRC1, src);
        imp_etf_sobel<<<src.blocks(), src.threads()>>>(etf, mag);
        GPU_CHECK_ERROR();

        float pmax;
        gpu_minmax(mag, 0, &pmax);
        if (pmax > 0) mag = gpu_mul(mag, 1 / pmax);
    }

    bind(&texSRC1, mag);
    for (int k = 0; k < N; ++k) {
        gpu_image<float2> tmp(src.size());
        bind(&texSRC2, etf);
        imp_etf_smooth_xy<1,0><<<tmp.blocks(), tmp.threads()>>>(tmp, sigma, precision);
        GPU_CHECK_ERROR();

        bind(&texSRC2, tmp);
        imp_etf_smooth_xy<0,1><<<etf.blocks(), etf.threads()>>>(etf, sigma, precision);
        GPU_CHECK_ERROR();
    }

    return etf;
}
