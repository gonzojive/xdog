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
#include "gpu_util.h"


template <typename T>
__global__ void imp_adjust( const gpu_plm2<T> src, gpu_plm2<T> dst, T a, T b) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;

    T c = src(ix, iy);
    dst(ix, iy) = a * c + b;
}                       


gpu_image<float> gpu_adjust( const gpu_image<float>& src, float a, float b ) {
    gpu_image<float> dst(src.size());
    imp_adjust<float><<<dst.blocks(), dst.threads()>>>(src, dst, a, b);
    GPU_CHECK_ERROR();
    return dst;
}


gpu_image<float4> gpu_adjust( const gpu_image<float4>& src, float4 a, float4 b ) {
    gpu_image<float4> dst(src.size());
    imp_adjust<float4><<<dst.blocks(), dst.threads()>>>(src, dst, a, b);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_invert( const gpu_plm2<float> src, gpu_plm2<float> dst ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;

    float c = src(ix, iy);
    dst(ix, iy) = 1 - __saturatef(c);
}                       


gpu_image<float> gpu_invert( const gpu_image<float>& src ) {
    gpu_image<float> dst(src.size());
    imp_invert<<<dst.blocks(), dst.threads()>>>(src, dst);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_invert( const gpu_plm2<float4> src, gpu_plm2<float4> dst ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;

    float4 c = src(ix, iy);
    dst(ix, iy) = make_float4( 1 - __saturatef(c.x), 
                               1 - __saturatef(c.y),
                               1 - __saturatef(c.z), 
                               1 );
}                       


gpu_image<float4> gpu_invert( const gpu_image<float4>& src ) {
    gpu_image<float4> dst(src.size());
    imp_invert<<<dst.blocks(), dst.threads()>>>(src, dst);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_saturate( const gpu_plm2<float> src, gpu_plm2<float> dst ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;

    dst(ix, iy) = __saturatef(src(ix, iy));
}                       


gpu_image<float> gpu_saturate( const gpu_image<float>& src ) {
    gpu_image<float> dst(src.size());
    imp_saturate<<<dst.blocks(), dst.threads()>>>(src, dst);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_saturate( const gpu_plm2<float4> src, gpu_plm2<float4> dst ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;

    float4 c = src(ix, iy);
    dst(ix, iy) = make_float4(__saturatef(c.x), __saturatef(c.y),__saturatef(c.z), 1);
}                       


gpu_image<float4> gpu_saturate( const gpu_image<float4>& src ) {
    gpu_image<float4> dst(src.size());
    imp_saturate<<<dst.blocks(), dst.threads()>>>(src, dst);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_lerp( const gpu_plm2<float4> a, const gpu_plm2<float4> b, gpu_plm2<float4> dst, float t) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;

    float4 ca = a(ix, iy);
    float4 cb = b(ix, iy);
    dst(ix, iy) = (1-t)*ca + t * cb;
}                       


gpu_image<float4> gpu_lerp( const gpu_image<float4>& a, const gpu_image<float4>& b, float t ) {
    assert(a.size() == b.size());
    gpu_image<float4> dst(a.size());
    imp_lerp<<<dst.blocks(), dst.threads()>>>( a, b, dst, t );
    GPU_CHECK_ERROR();
    return dst;
}
