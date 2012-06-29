//
// by Jan Eric Kyprianidis and Daniel Müller
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
#include "gpu_blend.h"


static texture<float,  2, cudaReadModeElementType> texBack1;
static texture<float4, 2, cudaReadModeElementType> texBack4;
static texture<float,  2, cudaReadModeElementType> texSrc1;
static texture<float4, 2, cudaReadModeElementType> texSrc4;


template <gpu_blend_mode M> inline __device__ float blend(float b, float s);

template <> inline __device__  float blend<GPU_BLEND_NORMAL>(float b, float s) { 
    return s; 
} 

template <> inline __device__ float blend<GPU_BLEND_MULTIPLY>(float b, float s) { 
    return b * s; 
}

template <> inline __device__ float blend<GPU_BLEND_SCREEN>(float b, float s) { 
    return b + s - (b * s); 
}

template <> inline __device__ float blend<GPU_BLEND_HARD_LIGHT>(float b, float s) { 
    return (s <= 0.5f)? blend<GPU_BLEND_MULTIPLY>(b, 2 * s)
                      : blend<GPU_BLEND_SCREEN>(b, 2 * s - 1); 
}

inline __device__ float D(float x) {
    return (x <= 0.25f)? ((16 * x - 12) * x + 4) * x
                       : sqrtf(x);
}

template <> inline __device__ float blend<GPU_BLEND_SOFT_LIGHT>(float b, float s) {
    return (s <= 0.5f)? b - (1 - 2 * s) * b * (1 - b)
                      : b + (2 * s - 1) * (D(b) - b);
}

template <> inline __device__ float blend<GPU_BLEND_OVERLAY>(float b, float s) { 
    return blend<GPU_BLEND_HARD_LIGHT>(s, b); 
}

template <> inline __device__  float blend<GPU_BLEND_LINEAR_BURN>(float b, float s) { 
    return b + s - 1; 
}

template <> inline __device__  float blend<GPU_BLEND_DIFFERENCE>(float b, float s) { 
    return fabs(b - s); 
}

template <> inline __device__  float blend<GPU_BLEND_LINEAR_DODGE>(float b, float s) { 
    return b + s; 
}


template <gpu_blend_mode M>
__global__ void imp_compose(gpu_plm2<float4> dst) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;

    if (ix >= dst.w || iy >= dst.h) 
        return;

    const float4 back = tex2D(texBack4, ix, iy);
    const float4 src = tex2D(texSrc4, ix, iy);

    const float3 b = make_float3(back.x, back.y, back.z);
    const float3 s = make_float3(src.x, src.y, src.z);

    const float  ab = back.w;
    const float  as = src.w;
    const float  ar = ab + as - (ab * as);

    const float3 bs = make_float3(
        blend<M>(b.x, s.x),
        blend<M>(b.y, s.y),
        blend<M>(b.z, s.z) );

    float3 r = (1.0 - as / ar) * b + (as / ar) * ((1.0 - ab) * s + ab * bs);
    dst(ix, iy) = make_float4(r.x, r.y, r.z, ar);
}


template <gpu_blend_mode M>
__global__ void imp_compose(gpu_plm2<float> dst) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;

    if (ix >= dst.w || iy >= dst.h) 
        return;

    const float b = tex2D(texBack1, ix, iy);
    const float s = tex2D(texSrc1, ix, iy);

    dst(ix, iy) = __saturatef( blend<M>(b, s) );
}


template <gpu_blend_mode M>
__global__ void imp_compose_intensity(gpu_plm2<float4> dst, float4 color) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;

    if (ix >= dst.w || iy >= dst.h) 
        return;

    const float4 back = tex2D(texBack4, ix, iy);
    const float  src = tex2D(texSrc1, ix, iy);

    const float3 b = make_float3(back.x, back.y, back.z);
    const float3 s = make_float3(color) * src;

    const float  ab = back.w;
    const float  as = color.w;
    const float  ar = ab + as - (ab * as);

    const float3 bs = make_float3(
        blend<M>(b.x, s.x),
        blend<M>(b.y, s.y),
        blend<M>(b.z, s.z) );

    float3 r = (1.0 - as / ar) * b + (as / ar) * ((1.0 - ab) * s + ab * bs);
    dst(ix, iy) = make_float4(__saturatef(r.x), __saturatef(r.y), __saturatef(r.z), __saturatef(ar));
}


gpu_image<float> gpu_blend( const gpu_image<float>& back,   const gpu_image<float>& src,
                             gpu_blend_mode mode )
{
    gpu_image<float> dst(back.size());

    bind(&texBack1, back);
    bind(&texSrc1, src);

    switch(mode) {
    case GPU_BLEND_NORMAL: 
        imp_compose<GPU_BLEND_NORMAL><<<dst.blocks(), dst.threads()>>>(dst); 
        break;
    case GPU_BLEND_MULTIPLY:
        imp_compose<GPU_BLEND_MULTIPLY><<<dst.blocks(), dst.threads()>>>(dst); 
        break;
    case GPU_BLEND_LINEAR_BURN:
        imp_compose<GPU_BLEND_LINEAR_BURN><<<dst.blocks(), dst.threads()>>>(dst); 
        break;
    case GPU_BLEND_SCREEN:
        imp_compose<GPU_BLEND_SCREEN><<<dst.blocks(), dst.threads()>>>(dst); 
        break;
    case GPU_BLEND_HARD_LIGHT:
        imp_compose<GPU_BLEND_HARD_LIGHT><<<dst.blocks(), dst.threads()>>>(dst); 
        break;
    case GPU_BLEND_SOFT_LIGHT:
        imp_compose<GPU_BLEND_SOFT_LIGHT><<<dst.blocks(), dst.threads()>>>(dst); 
        break;
    case GPU_BLEND_OVERLAY:
        imp_compose<GPU_BLEND_OVERLAY><<<dst.blocks(), dst.threads()>>>(dst); 
        break;
    case GPU_BLEND_DIFFERENCE:
        imp_compose<GPU_BLEND_DIFFERENCE><<<dst.blocks(), dst.threads()>>>(dst); 
        break;
    case GPU_BLEND_LINEAR_DODGE:
        imp_compose<GPU_BLEND_LINEAR_DODGE><<<dst.blocks(), dst.threads()>>>(dst); 
        break;  
    };
    GPU_CHECK_ERROR();
    return dst;
}


gpu_image<float4> gpu_blend( const gpu_image<float4>& back, const gpu_image<float4>& src,
                             gpu_blend_mode mode)
{
    gpu_image<float4> dst(back.size());

    bind(&texBack4, back);
    bind(&texSrc4, src);

    switch(mode) {
    case GPU_BLEND_NORMAL: 
        imp_compose<GPU_BLEND_NORMAL><<<dst.blocks(), dst.threads()>>>(dst); 
        break;
    case GPU_BLEND_MULTIPLY:
        imp_compose<GPU_BLEND_MULTIPLY><<<dst.blocks(), dst.threads()>>>(dst); 
        break;
    case GPU_BLEND_SCREEN:
        imp_compose<GPU_BLEND_SCREEN><<<dst.blocks(), dst.threads()>>>(dst); 
        break;
    case GPU_BLEND_HARD_LIGHT:
        imp_compose<GPU_BLEND_HARD_LIGHT><<<dst.blocks(), dst.threads()>>>(dst); 
        break;
    case GPU_BLEND_SOFT_LIGHT:
        imp_compose<GPU_BLEND_SOFT_LIGHT><<<dst.blocks(), dst.threads()>>>(dst); 
        break;
    case GPU_BLEND_OVERLAY:
        imp_compose<GPU_BLEND_OVERLAY><<<dst.blocks(), dst.threads()>>>(dst); 
        break;
    case GPU_BLEND_LINEAR_BURN:
        imp_compose<GPU_BLEND_LINEAR_BURN><<<dst.blocks(), dst.threads()>>>(dst); 
        break;
    case GPU_BLEND_DIFFERENCE:
        imp_compose<GPU_BLEND_DIFFERENCE><<<dst.blocks(), dst.threads()>>>(dst); 
        break;
    case GPU_BLEND_LINEAR_DODGE:
        imp_compose<GPU_BLEND_LINEAR_DODGE><<<dst.blocks(), dst.threads()>>>(dst); 
        break;  
    };
    GPU_CHECK_ERROR();
    return dst;
}


gpu_image<float4> gpu_blend_intensity( const gpu_image<float4>& back, const gpu_image<float>& src,
                                       gpu_blend_mode mode, float4 color )
{
    gpu_image<float4> dst(back.size());

    bind(&texBack4, back);
    bind(&texSrc1, src);

    switch(mode) {
    case GPU_BLEND_NORMAL: 
        imp_compose_intensity<GPU_BLEND_NORMAL><<<dst.blocks(), dst.threads()>>>(dst, color); 
        break;
    case GPU_BLEND_MULTIPLY:
        imp_compose_intensity<GPU_BLEND_MULTIPLY><<<dst.blocks(), dst.threads()>>>(dst, color); 
        break;
    case GPU_BLEND_SCREEN:
        imp_compose_intensity<GPU_BLEND_SCREEN><<<dst.blocks(), dst.threads()>>>(dst, color); 
        break;
    case GPU_BLEND_HARD_LIGHT:
        imp_compose_intensity<GPU_BLEND_HARD_LIGHT><<<dst.blocks(), dst.threads()>>>(dst, color); 
        break;
    case GPU_BLEND_SOFT_LIGHT:
        imp_compose_intensity<GPU_BLEND_SOFT_LIGHT><<<dst.blocks(), dst.threads()>>>(dst, color); 
        break;
    case GPU_BLEND_OVERLAY:
        imp_compose_intensity<GPU_BLEND_OVERLAY><<<dst.blocks(), dst.threads()>>>(dst, color); 
        break;
    case GPU_BLEND_LINEAR_BURN:
        imp_compose_intensity<GPU_BLEND_LINEAR_BURN><<<dst.blocks(), dst.threads()>>>(dst, color); 
        break;
    case GPU_BLEND_DIFFERENCE:
        imp_compose_intensity<GPU_BLEND_DIFFERENCE><<<dst.blocks(), dst.threads()>>>(dst, color); 
        break;
    case GPU_BLEND_LINEAR_DODGE:
        imp_compose_intensity<GPU_BLEND_LINEAR_DODGE><<<dst.blocks(), dst.threads()>>>(dst, color); 
        break;
    };
    GPU_CHECK_ERROR();
    return dst;
}
