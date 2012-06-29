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
#include "gpu_grad.h"


static texture<float, 2, cudaReadModeElementType> texSRC1;
static texture<float4, 2, cudaReadModeElementType> texSRC4;


__global__ void imp_grad_central_diff( gpu_plm2<float2> dst, bool normalize ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;

    float gx = ( -tex2D(texSRC1, ix-1, iy  ) + tex2D(texSRC1, ix+1, iy  ) ) / 2;
    float gy = ( -tex2D(texSRC1, ix,   iy-1) + tex2D(texSRC1, ix,   iy+1) ) / 2;

    if (normalize) {
        float n = sqrtf(gx*gx + gy*gy);
        dst(ix,iy) = (n > 0)? make_float2(gx / n, gy / n) : make_float2(0);
    } else {
        dst(ix,iy) = make_float2(gx, gy);
    }
}

                  
gpu_image<float2> gpu_grad_central_diff( const gpu_image<float>& src, bool normalize ) {
    gpu_image<float2> dst(src.size());
    bind(&texSRC1, src);
    imp_grad_central_diff<<<src.blocks(), src.threads()>>>(dst, normalize);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_grad_scharr_for_axis( gpu_plm2<float> dst, int axis ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;

    if (axis == 0) {
        float u = (
              -0.183f * tex2D(texSRC1, ix-1, iy-1) +
              -0.634f * tex2D(texSRC1, ix-1, iy) + 
              -0.183f * tex2D(texSRC1, ix-1, iy+1) +
              +0.183f * tex2D(texSRC1, ix+1, iy-1) +
              +0.634f * tex2D(texSRC1, ix+1, iy) + 
              +0.183f * tex2D(texSRC1, ix+1, iy+1)
              ) * 0.5f;
        dst(ix, iy) = u;
    } else {
        float v = (
              -0.183f * tex2D(texSRC1, ix-1, iy-1) + 
              -0.634f * tex2D(texSRC1, ix,   iy-1) + 
              -0.183f * tex2D(texSRC1, ix+1, iy-1) +
              +0.183f * tex2D(texSRC1, ix-1, iy+1) +
              +0.634f * tex2D(texSRC1, ix,   iy+1) + 
              +0.183f * tex2D(texSRC1, ix+1, iy+1)
              ) * 0.5f;
        dst(ix, iy) = v;
    }
}


gpu_image<float> gpu_grad_scharr_for_axis( const gpu_image<float>& src, int axis ) {
    gpu_image<float> dst(src.size());
    bind(&texSRC1, src);
    imp_grad_scharr_for_axis<<<src.blocks(), src.threads()>>>(dst, axis);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_grad_scharr( gpu_plm2<float2> dst, bool normalize ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;

    float gx = (
          -0.183f * tex2D(texSRC1, ix-1, iy-1) +
          -0.634f * tex2D(texSRC1, ix-1, iy) + 
          -0.183f * tex2D(texSRC1, ix-1, iy+1) +
          +0.183f * tex2D(texSRC1, ix+1, iy-1) +
          +0.634f * tex2D(texSRC1, ix+1, iy) + 
          +0.183f * tex2D(texSRC1, ix+1, iy+1)
          ) * 0.5f;

    float gy = (
          -0.183f * tex2D(texSRC1, ix-1, iy-1) + 
          -0.634f * tex2D(texSRC1, ix,   iy-1) + 
          -0.183f * tex2D(texSRC1, ix+1, iy-1) +
          +0.183f * tex2D(texSRC1, ix-1, iy+1) +
          +0.634f * tex2D(texSRC1, ix,   iy+1) + 
          +0.183f * tex2D(texSRC1, ix+1, iy+1)
          ) * 0.5f;

    if (normalize) {
        float n = sqrtf(gx*gx + gy*gy);
        dst(ix,iy) = (n > 0)? make_float2(gx / n, gy / n) : make_float2(0);
    } else {
        dst(ix,iy) = make_float2(gx, gy);
    }
}


gpu_image<float2> gpu_grad_scharr( const gpu_image<float>& src, bool normalize ) {
    gpu_image<float2> dst(src.size());
    bind(&texSRC1, src);
    imp_grad_scharr<<<src.blocks(), src.threads()>>>(dst, normalize);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_grad_to_axis( gpu_plm2<float2> dst, const gpu_plm2<float2> src, bool squared ) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;

    float2 g = src(ix, iy);
    float n = g.x*g.x + g.y*g.y;
    if (n > 0) {
        if (!squared) n = sqrtf(n);
        float phi = 2 * atan2(g.y, g.x);
        g = make_float2(n * __cosf(phi), n * __sinf(phi));
    } else {
        g = make_float2(0);
    }
    dst(ix,iy) = g;
}


gpu_image<float2> gpu_grad_to_axis( const gpu_image<float2>& src, bool squared ) {
    gpu_image<float2> dst(src.size());
    imp_grad_to_axis<<<src.blocks(), src.threads()>>>(dst, src, squared);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_grad_from_axis( gpu_plm2<float2> dst, const gpu_plm2<float2> src, bool squared ) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;

    float2 g = src(ix, iy);
    float n = g.x*g.x + g.y*g.y;
    if (n > 0) {
        if (!squared) n = sqrtf(n);
        float phi = 0.5f * atan2(g.y / n, g.x / n);
        g = make_float2(n * __cosf(phi), n * __sinf(phi));
    } else {
        g = make_float2(0);
    }
    dst(ix,iy) = g;
}


gpu_image<float2> gpu_grad_from_axis( const gpu_image<float2>& src, bool squared ) {
    gpu_image<float2> dst(src.size());
    imp_grad_from_axis<<<src.blocks(), src.threads()>>>(dst, src, squared);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_grad_to_angle( gpu_plm2<float> dst, const gpu_plm2<float2> src ) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;

    float2 g = src(ix, iy);
    if (g.y < 0) {
        g = -g;
    }
    float n = sqrtf(g.x*g.x + g.y*g.y);
    float phi = 0;
    if (n > 0) {
        phi = atan2(g.y / n, g.x / n) + CUDART_PIO2_F;
        if (phi < 0) {
            phi += CUDART_PI_F;
        }
    }
    dst(ix,iy) = phi;
}


gpu_image<float> gpu_grad_to_angle( const gpu_image<float2>& src ) {
    gpu_image<float> dst(src.size());
    imp_grad_to_angle<<<src.blocks(), src.threads()>>>(dst, src);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_grad_to_lfm( gpu_plm2<float4> dst, const gpu_plm2<float2> src ) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;

    float2 g = src(ix, iy);
    dst(ix,iy) = make_float4( -g.y, g.x, 1, 1 );
}


gpu_image<float4> gpu_grad_to_lfm( const gpu_image<float2>& src ) {
    gpu_image<float4> dst(src.size());
    imp_grad_to_lfm<<<src.blocks(), src.threads()>>>(dst, src);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_grad_sobel_mag1( gpu_plm2<float> dst ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;

    float u = (-1 * tex2D(texSRC1, ix-1, iy-1) +
               -2 * tex2D(texSRC1, ix-1, iy  ) + 
               -1 * tex2D(texSRC1, ix-1, iy+1) +
               +1 * tex2D(texSRC1, ix+1, iy-1) +
               +2 * tex2D(texSRC1, ix+1, iy  ) + 
               +1 * tex2D(texSRC1, ix+1, iy+1));
          
    float v = (-1 * tex2D(texSRC1, ix-1, iy-1) + 
               -2 * tex2D(texSRC1, ix,   iy-1) + 
               -1 * tex2D(texSRC1, ix+1, iy-1) +
               +1 * tex2D(texSRC1, ix-1, iy+1) +
               +2 * tex2D(texSRC1, ix,   iy+1) + 
               +1 * tex2D(texSRC1, ix+1, iy+1));
          
    dst(ix, iy) = sqrtf(u*u + v*v);
}


gpu_image<float> gpu_grad_sobel_mag( const gpu_image<float>& src ) {
    gpu_image<float> dst(src.size());
    bind(&texSRC1, src);
    imp_grad_sobel_mag1<<<src.blocks(), src.threads()>>>(dst);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_grad_sobel_mag4( gpu_plm2<float> dst ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;

    float3 u = (-1 * make_float3(tex2D(texSRC4, ix-1, iy-1)) +
                -2 * make_float3(tex2D(texSRC4, ix-1, iy  )) + 
                -1 * make_float3(tex2D(texSRC4, ix-1, iy+1)) +
                +1 * make_float3(tex2D(texSRC4, ix+1, iy-1)) +
                +2 * make_float3(tex2D(texSRC4, ix+1, iy  )) + 
                +1 * make_float3(tex2D(texSRC4, ix+1, iy+1)));
          
    float3 v = (-1 * make_float3(tex2D(texSRC4, ix-1, iy-1)) + 
                -2 * make_float3(tex2D(texSRC4, ix,   iy-1)) + 
                -1 * make_float3(tex2D(texSRC4, ix+1, iy-1)) +
                +1 * make_float3(tex2D(texSRC4, ix-1, iy+1)) +
                +2 * make_float3(tex2D(texSRC4, ix,   iy+1)) + 
                +1 * make_float3(tex2D(texSRC4, ix+1, iy+1)));
          
    dst(ix, iy) = sqrtf(dot(u,u) + dot(v,v));
}


gpu_image<float> gpu_grad_sobel_mag( const gpu_image<float4>& src ) {
    gpu_image<float> dst(src.size());
    bind(&texSRC4, src);
    imp_grad_sobel_mag4<<<src.blocks(), src.threads()>>>(dst);
    GPU_CHECK_ERROR();
    return dst;
}
