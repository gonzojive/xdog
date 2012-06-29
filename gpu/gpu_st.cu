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
#include "gpu_st.h"
#include "gpu_gauss.h"


static texture<float, 2, cudaReadModeElementType> texSRC1;
static texture<float4, 2, cudaReadModeElementType> texSRC4;


template<typename T> __device__ T texSRC(float x, float y);
template<> inline __device__ float texSRC(float x, float y) { return tex2D(texSRC1, x, y); }
template<> inline __device__ float4 texSRC(float x, float y) { return tex2D(texSRC4, x, y); }


__global__ void imp_gray_st_central_diff( gpu_plm2<float4> dst ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;

    float u = ( -tex2D(texSRC1, ix-1, iy  ) + tex2D(texSRC1, ix+1, iy  ) ) / 2;
    float v = ( -tex2D(texSRC1, ix,   iy-1) + tex2D(texSRC1, ix,   iy+1) ) / 2;
    float3 g = make_float3(u * u, v * v, u * v);

    dst(ix, iy) = make_float4( g, 1 );
}

                  
gpu_image<float4> gpu_st_central_diff( const gpu_image<float>& src ) {
    gpu_image<float4> dst(src.size());
    bind(&texSRC1, src);
    imp_gray_st_central_diff<<<src.blocks(), src.threads()>>>(dst);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_gray_st_gaussian( gpu_plm2<float4> dst, float sigma ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;

    float twoSigma2 = 2.0f * sigma * sigma;
    int halfWidth = int(ceilf( 3.0f * sigma ));

    float u = 0;
    float v = 0;

    if (halfWidth > 0) {
        float sum = 0;
        for ( int i = -halfWidth; i <= halfWidth; ++i ) {
            for ( int j = -halfWidth; j <= halfWidth; ++j ) {
                float d = length(make_float2(i,j));
                float e = __expf( -d *d / twoSigma2 );
                float c = tex2D(texSRC1, ix + i, iy + j);
                u += i * e * c;
                v += j * e * c;
                sum += e;
            }
        }

        sum *= -sigma * sigma;
        u /= sum;
        v /= sum;
    }

    dst(ix, iy) = make_float4(u*u, v*v, u*v, 1);
}


__global__ void imp_color_st_gaussian( gpu_plm2<float4> dst, float sigma ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;

    float twoSigma2 = 2.0f * sigma * sigma;
    int halfWidth = int(ceilf( 3.0f * sigma ));

    float3 u = make_float3(0);
    float3 v = make_float3(0);

    if (halfWidth > 0) {
        float sum = 0;
        for ( int i = -halfWidth; i <= halfWidth; ++i ) {
            for ( int j = -halfWidth; j <= halfWidth; ++j ) {
                float d = length(make_float2(i,j));
                float e = __expf( -d *d / twoSigma2 );
                float3 c = make_float3(tex2D(texSRC4, ix + i, iy + j));
                u += i * e * c;
                v += j * e * c;
                sum += e;
            }
        }

        sum *= -sigma * sigma;
        u /= sum;
        v /= sum;
    }

    dst(ix, iy) = make_float4(dot(u, u), dot(v, v), dot(u, v), 1);
}

                                      
gpu_image<float4> gpu_st_gaussian( const gpu_image<float>& src, float sigma ) {
    gpu_image<float4> dst(src.size());
    bind(&texSRC1, src);
    imp_gray_st_gaussian<<<src.blocks(), src.threads()>>>(dst, sigma);
    GPU_CHECK_ERROR();
    return dst;
}


gpu_image<float4> gpu_st_gaussian( const gpu_image<float4>& src, float sigma ) {
    gpu_image<float4> dst(src.size());
    bind(&texSRC4, src);
    imp_color_st_gaussian<<<src.blocks(), src.threads()>>>(dst, sigma);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_gray_st_sobel( gpu_plm2<float4> dst ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;

    float u = (
           -1 * tex2D(texSRC1, ix-1, iy-1) +
           -2 * tex2D(texSRC1, ix-1, iy  ) + 
           -1 * tex2D(texSRC1, ix-1, iy+1) +
           +1 * tex2D(texSRC1, ix+1, iy-1) +
           +2 * tex2D(texSRC1, ix+1, iy  ) + 
           +1 * tex2D(texSRC1, ix+1, iy+1)
           ) / 8;

    float v = (
           -1 * tex2D(texSRC1, ix-1, iy-1) + 
           -2 * tex2D(texSRC1, ix,   iy-1) + 
           -1 * tex2D(texSRC1, ix+1, iy-1) +
           +1 * tex2D(texSRC1, ix-1, iy+1) +
           +2 * tex2D(texSRC1, ix,   iy+1) + 
           +1 * tex2D(texSRC1, ix+1, iy+1)
           ) / 8;
    
    dst(ix, iy) = make_float4(u*u, v*v, u*v, 1);
}


__global__ void imp_color_st_sobel( gpu_plm2<float4> dst ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;

    float3 u = (
           -1 * make_float3(tex2D(texSRC4, ix-1, iy-1)) +
           -2 * make_float3(tex2D(texSRC4, ix-1, iy)) + 
           -1 * make_float3(tex2D(texSRC4, ix-1, iy+1)) +
           +1 * make_float3(tex2D(texSRC4, ix+1, iy-1)) +
           +2 * make_float3(tex2D(texSRC4, ix+1, iy)) + 
           +1 * make_float3(tex2D(texSRC4, ix+1, iy+1))
           ) / 8;

    float3 v = (
           -1 * make_float3(tex2D(texSRC4, ix-1, iy-1)) + 
           -2 * make_float3(tex2D(texSRC4, ix,   iy-1)) + 
           -1 * make_float3(tex2D(texSRC4, ix+1, iy-1)) +
           +1 * make_float3(tex2D(texSRC4, ix-1, iy+1)) +
           +2 * make_float3(tex2D(texSRC4, ix,   iy+1)) + 
           +1 * make_float3(tex2D(texSRC4, ix+1, iy+1))
           ) / 8;
    
    dst(ix, iy) = make_float4(dot(u, u), dot(v, v), dot(u, v), 1);
}


gpu_image<float4> gpu_st_sobel( const gpu_image<float>& src ) {
    gpu_image<float4> dst(src.size());
    bind(&texSRC1, src);
    imp_gray_st_sobel<<<src.blocks(), src.threads()>>>(dst);
    GPU_CHECK_ERROR();
    return dst;
}


gpu_image<float4> gpu_st_sobel( const gpu_image<float4>& src ) {
    gpu_image<float4> dst(src.size());
    bind(&texSRC4, src);
    imp_color_st_sobel<<<src.blocks(), src.threads()>>>(dst);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_gray_st_scharr( gpu_plm2<float4> dst ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;

    float u = (
          -0.183f * tex2D(texSRC1, ix-1, iy-1) +
          -0.634f * tex2D(texSRC1, ix-1, iy) + 
          -0.183f * tex2D(texSRC1, ix-1, iy+1) +
          +0.183f * tex2D(texSRC1, ix+1, iy-1) +
          +0.634f * tex2D(texSRC1, ix+1, iy) + 
          +0.183f * tex2D(texSRC1, ix+1, iy+1)
          ) * 0.5f;

    float v = (
          -0.183f * tex2D(texSRC1, ix-1, iy-1) + 
          -0.634f * tex2D(texSRC1, ix,   iy-1) + 
          -0.183f * tex2D(texSRC1, ix+1, iy-1) +
          +0.183f * tex2D(texSRC1, ix-1, iy+1) +
          +0.634f * tex2D(texSRC1, ix,   iy+1) + 
          +0.183f * tex2D(texSRC1, ix+1, iy+1)
          ) * 0.5f;
    
    float3 g = make_float3(u * u, v * v, u * v);
    dst(ix, iy) = make_float4( g, 1 );
}


__global__ void imp_color_st_scharr( gpu_plm2<float4> dst ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;

    float3 u = (
          -0.183f * make_float3(tex2D(texSRC4, ix-1, iy-1)) +
          -0.634f * make_float3(tex2D(texSRC4, ix-1, iy)) + 
          -0.183f * make_float3(tex2D(texSRC4, ix-1, iy+1)) +
          +0.183f * make_float3(tex2D(texSRC4, ix+1, iy-1)) +
          +0.634f * make_float3(tex2D(texSRC4, ix+1, iy)) + 
          +0.183f * make_float3(tex2D(texSRC4, ix+1, iy+1))
          ) * 0.5f;

    float3 v = (
          -0.183f * make_float3(tex2D(texSRC4, ix-1, iy-1)) + 
          -0.634f * make_float3(tex2D(texSRC4, ix,   iy-1)) + 
          -0.183f * make_float3(tex2D(texSRC4, ix+1, iy-1)) +
          +0.183f * make_float3(tex2D(texSRC4, ix-1, iy+1)) +
          +0.634f * make_float3(tex2D(texSRC4, ix,   iy+1)) + 
          +0.183f * make_float3(tex2D(texSRC4, ix+1, iy+1))
          ) * 0.5f;

    dst(ix, iy) = make_float4( make_float3(dot(u,u), dot(v,v), dot(u,v)), 1 );
}


gpu_image<float4> gpu_st_scharr( const gpu_image<float>& src ) {
    gpu_image<float4> dst(src.size());
    bind(&texSRC1, src);
    imp_gray_st_scharr<<<src.blocks(), src.threads()>>>(dst);
    GPU_CHECK_ERROR();
    return dst;
}


gpu_image<float4> gpu_st_scharr( const gpu_image<float4>& src ) {
    gpu_image<float4> dst(src.size());
    bind(&texSRC4, src);
    imp_color_st_scharr<<<src.blocks(), src.threads()>>>(dst);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_st_from_gradient( gpu_plm2<float4> dst, const gpu_plm2<float2> src ) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;

    float2 g = src(ix, iy);
    dst(ix,iy) = make_float4(g.x * g.x, g.y * g.y, g.x * g.y, 1);
}


gpu_image<float4> gpu_st_from_gradient( const gpu_image<float2>& src ) {
    gpu_image<float4> dst(src.size());
    imp_st_from_gradient<<<src.blocks(), src.threads()>>>(dst, src);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_st_from_tangent( gpu_plm2<float4> dst, const gpu_plm2<float2> src ) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;

    float2 t = src(ix, iy);
    float2 g = make_float2(t.y, -t.x);
    dst(ix,iy) = make_float4(g.x * g.x, g.y * g.y, g.x * g.y, 1);
}


gpu_image<float4> gpu_st_from_tangent( const gpu_image<float2>& src ) {
    gpu_image<float4> dst(src.size());
    imp_st_from_tangent<<<src.blocks(), src.threads()>>>(dst, src);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_st_tfm( gpu_plm2<float4> dst ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;

    float4 g = tex2D(texSRC4, ix, iy);
    dst(ix, iy) = st2tfm(g);
}


gpu_image<float4> gpu_st_tfm( const gpu_image<float4>& st ) {
    gpu_image<float4> dst( st.size() );
    bind( &texSRC4, st );
    imp_st_tfm<<<dst.blocks(), dst.threads()>>>(dst); 
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_st_lfm( gpu_plm2<float4> dst, const gpu_plm2<float4> st ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;

    float4 g = st(ix, iy);
    dst(ix, iy) = st2lfm(g);
}


__global__ void imp_st_lfm( gpu_plm2<float4> dst, const gpu_plm2<float4> st, float alpha ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;

    float4 g = st(ix, iy);
    dst(ix, iy) = st2lfm(g, alpha);
}


gpu_image<float4> gpu_st_lfm( const gpu_image<float4>& st, float alpha ) {
    gpu_image<float4> dst( st.size() );
    if (alpha <= 0) 
        imp_st_lfm<<<dst.blocks(), dst.threads()>>>(dst, st); 
    else
        imp_st_lfm<<<dst.blocks(), dst.threads()>>>(dst, st, alpha); 
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_st_angle( gpu_plm2<float> dst, const gpu_plm2<float4> src ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;
    dst(ix, iy) = st2angle(src(ix, iy));
}


gpu_image<float> gpu_st_angle( const gpu_image<float4>& st ) {
    gpu_image<float> dst( st.size() );
    imp_st_angle<<<dst.blocks(), dst.threads()>>>(dst, st); 
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_st_anisotropy( gpu_plm2<float> dst, const gpu_plm2<float4> src ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;
    dst(ix, iy) = st2A(src(ix, iy));
}


gpu_image<float> gpu_st_anisotropy( const gpu_image<float4>& st ) {
    gpu_image<float> dst( st.size() );
    imp_st_anisotropy<<<dst.blocks(), dst.threads()>>>(dst, st); 
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_st_threshold_mag( gpu_plm2<float4> dst, float threshold2 ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h) 
        return;

    float3 g = make_float3(tex2D(texSRC4, ix, iy));
    float mag = g.x * g.x + g.y * g.y + 2 * g.z * g.z;
    if (mag < threshold2) {
        mag = 0;
    }
    dst(ix, iy) = make_float4(g, mag);
}

                                      
gpu_image<float4> gpu_st_threshold_mag( const gpu_image<float4>& st, float threshold ) {
    gpu_image<float4> dst( st.size() );
    bind( &texSRC4, st );
    imp_st_threshold_mag<<<dst.blocks(), dst.threads()>>>(dst, threshold*threshold); 
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_st_normalize( const gpu_plm2<float4> src, gpu_plm2<float4> dst ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h) 
        return;

    float4 g = src(ix, iy);
    float mag = sqrtf(fmaxf(0, g.x * g.x + g.y * g.y + 2 * g.z * g.z));
    if (mag > 0) 
        dst(ix, iy) = make_float4(g.x / mag, g.y / mag, g.z / mag, 1);
    else
        dst(ix, iy) = make_float4(0);
}


gpu_image<float4> gpu_st_normalize( const gpu_image<float4>& st ) {
    gpu_image<float4> dst(st.size());
    imp_st_normalize<<<dst.blocks(), dst.threads()>>>( st, dst );
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_st_flatten( const gpu_plm2<float4> src, gpu_plm2<float4> dst ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h) 
        return;

    float4 g = src(ix, iy);
    float phi = 0.5f * atan2(2 * g.z, g.x - g.y);
    float a = 0.5f * (g.y + g.x); 
    float b = 0.5f * sqrtf(fmaxf(0.0, g.y*g.y - 2*g.x*g.y + g.x*g.x + 4*g.z*g.z));
    float2 l = make_float2(a + b, a - b);

    float c = cosf(phi);
    float s = sinf(phi);

    dst(ix, iy) = make_float4(
        l.x*c*c,
        l.x*s*s,
        l.x*c*s,
        1
    );
}


gpu_image<float4> gpu_st_flatten( const gpu_image<float4>& st ) {
    gpu_image<float4> dst(st.size());
    imp_st_flatten<<<dst.blocks(), dst.threads()>>>( st, dst );
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_st_rotate( const gpu_plm2<float4> src, gpu_plm2<float4> dst, const float s, const float c ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h) 
        return;

    float4 g = src(ix, iy);
    dst(ix, iy) = make_float4(
        c*c*g.x + 2*s*c*g.z + s*s*g.y,
        s*s*g.x - 2*s*c*g.z + c*c*g.y,
        (c*c - s*s)*g.z + c*s*(g.y - g.x),
        1
    );
}


gpu_image<float4> gpu_st_rotate( const gpu_image<float4>& st, float angle ) {
    gpu_image<float4> dst(st.size());
    float phi = CUDART_PI_F * angle / 180.0f;
    float s = sin(phi);
    float c = cos(phi);
    imp_st_rotate<<<dst.blocks(), dst.threads()>>>( st, dst, s, c );
    GPU_CHECK_ERROR();
    return dst;
}


