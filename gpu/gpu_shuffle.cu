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
#include "gpu_shuffle.h"


template <typename T,  typename V, unsigned index> T shuffle(const V v);
template<> inline __device__ float shuffle<float, float2, 0x0>(const float2 v) { return v.x; }
template<> inline __device__ float shuffle<float, float2, 0x1>(const float2 v) { return v.y; }

template<> inline __device__ float shuffle<float, float4, 0x0>(const float4 v) { return v.x; }
template<> inline __device__ float shuffle<float, float4, 0x1>(const float4 v) { return v.y; }
template<> inline __device__ float shuffle<float, float4, 0x2>(const float4 v) { return v.z; }
template<> inline __device__ float shuffle<float, float4, 0x3>(const float4 v) { return v.w; }

template<> inline __device__ float2 shuffle<float2, float2, 0x00>(const float2 v) { return make_float2(v.x,v.x); }
template<> inline __device__ float2 shuffle<float2, float2, 0x01>(const float2 v) { return make_float2(v.y,v.x); }
template<> inline __device__ float2 shuffle<float2, float2, 0x10>(const float2 v) { return make_float2(v.x,v.y); }
template<> inline __device__ float2 shuffle<float2, float2, 0x11>(const float2 v) { return make_float2(v.y,v.y); }

template<> inline __device__ float2 shuffle<float2, float4, 0x00>(const float4 v) { return make_float2(v.x,v.x); }
template<> inline __device__ float2 shuffle<float2, float4, 0x01>(const float4 v) { return make_float2(v.y,v.x); }
template<> inline __device__ float2 shuffle<float2, float4, 0x02>(const float4 v) { return make_float2(v.z,v.x); }
template<> inline __device__ float2 shuffle<float2, float4, 0x03>(const float4 v) { return make_float2(v.w,v.x); }
template<> inline __device__ float2 shuffle<float2, float4, 0x10>(const float4 v) { return make_float2(v.x,v.y); }
template<> inline __device__ float2 shuffle<float2, float4, 0x11>(const float4 v) { return make_float2(v.y,v.y); }
template<> inline __device__ float2 shuffle<float2, float4, 0x12>(const float4 v) { return make_float2(v.z,v.y); }
template<> inline __device__ float2 shuffle<float2, float4, 0x13>(const float4 v) { return make_float2(v.w,v.y); }
template<> inline __device__ float2 shuffle<float2, float4, 0x20>(const float4 v) { return make_float2(v.x,v.z); }
template<> inline __device__ float2 shuffle<float2, float4, 0x21>(const float4 v) { return make_float2(v.y,v.z); }
template<> inline __device__ float2 shuffle<float2, float4, 0x22>(const float4 v) { return make_float2(v.z,v.z); }
template<> inline __device__ float2 shuffle<float2, float4, 0x23>(const float4 v) { return make_float2(v.w,v.z); }
template<> inline __device__ float2 shuffle<float2, float4, 0x30>(const float4 v) { return make_float2(v.x,v.w); }
template<> inline __device__ float2 shuffle<float2, float4, 0x31>(const float4 v) { return make_float2(v.y,v.w); }
template<> inline __device__ float2 shuffle<float2, float4, 0x32>(const float4 v) { return make_float2(v.z,v.w); }
template<> inline __device__ float2 shuffle<float2, float4, 0x33>(const float4 v) { return make_float2(v.w,v.w); }

template<> inline __device__ float4 shuffle<float4, float2, 0x0000>(const float2 v) { return make_float4(v.x,v.x,v.x,v.x); }
template<> inline __device__ float4 shuffle<float4, float2, 0x0001>(const float2 v) { return make_float4(v.y,v.x,v.x,v.x); }
template<> inline __device__ float4 shuffle<float4, float2, 0x0010>(const float2 v) { return make_float4(v.x,v.y,v.x,v.x); }
template<> inline __device__ float4 shuffle<float4, float2, 0x0011>(const float2 v) { return make_float4(v.y,v.y,v.x,v.x); }
template<> inline __device__ float4 shuffle<float4, float2, 0x0100>(const float2 v) { return make_float4(v.x,v.x,v.y,v.x); }
template<> inline __device__ float4 shuffle<float4, float2, 0x0101>(const float2 v) { return make_float4(v.y,v.x,v.y,v.x); }
template<> inline __device__ float4 shuffle<float4, float2, 0x0110>(const float2 v) { return make_float4(v.x,v.y,v.y,v.x); }
template<> inline __device__ float4 shuffle<float4, float2, 0x0111>(const float2 v) { return make_float4(v.y,v.y,v.y,v.x); }
template<> inline __device__ float4 shuffle<float4, float2, 0x1000>(const float2 v) { return make_float4(v.x,v.x,v.x,v.y); }
template<> inline __device__ float4 shuffle<float4, float2, 0x1001>(const float2 v) { return make_float4(v.y,v.x,v.x,v.y); }
template<> inline __device__ float4 shuffle<float4, float2, 0x1010>(const float2 v) { return make_float4(v.x,v.y,v.x,v.y); }
template<> inline __device__ float4 shuffle<float4, float2, 0x1011>(const float2 v) { return make_float4(v.y,v.y,v.x,v.y); }
template<> inline __device__ float4 shuffle<float4, float2, 0x1100>(const float2 v) { return make_float4(v.x,v.x,v.y,v.y); }
template<> inline __device__ float4 shuffle<float4, float2, 0x1101>(const float2 v) { return make_float4(v.y,v.x,v.y,v.y); }
template<> inline __device__ float4 shuffle<float4, float2, 0x1110>(const float2 v) { return make_float4(v.x,v.y,v.y,v.y); }
template<> inline __device__ float4 shuffle<float4, float2, 0x1111>(const float2 v) { return make_float4(v.y,v.y,v.y,v.y); }


template <typename T, typename V, unsigned index>
__global__ void imp_shuffle( const gpu_plm2<V> src, gpu_plm2<T> dst) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;
    dst(ix, iy) = shuffle<T,V,index>(src(ix, iy));
}
            

gpu_image<float> gpu_shuffle( const gpu_image<float2>& src, int x) {
    gpu_image<float> dst(src.size());
    switch (x) {
        case 0: imp_shuffle<float,float2,0><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 1: imp_shuffle<float,float2,1><<<dst.blocks(), dst.threads()>>>(src, dst); break;
    }
    GPU_CHECK_ERROR();
    return dst;
}

gpu_image<float> gpu_shuffle( const gpu_image<float4>& src, int x ) {
    gpu_image<float> dst(src.size());
    switch (x) {
        case 0: imp_shuffle<float,float4,0><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 1: imp_shuffle<float,float4,1><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 2: imp_shuffle<float,float4,2><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 3: imp_shuffle<float,float4,3><<<dst.blocks(), dst.threads()>>>(src, dst); break;
    }
    GPU_CHECK_ERROR();
    return dst;
}


gpu_image<float2> gpu_shuffle( const gpu_image<float2>& src, int x, int y ) {
    gpu_image<float2> dst(src.size());
    switch (((y & 1) << 4) | (x & 1)) {
        case 0x00: imp_shuffle<float2,float2,0x00><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 0x01: imp_shuffle<float2,float2,0x01><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 0x10: imp_shuffle<float2,float2,0x10><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 0x11: imp_shuffle<float2,float2,0x11><<<dst.blocks(), dst.threads()>>>(src, dst); break;
    }
    GPU_CHECK_ERROR();
    return dst;
}


gpu_image<float2> gpu_shuffle( const gpu_image<float4>& src, int x, int y ) {
    gpu_image<float2> dst(src.size());
    switch (((y & 3) << 4) | (x & 3)) {
        case 0x00: imp_shuffle<float2,float4,0x00><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 0x01: imp_shuffle<float2,float4,0x01><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 0x02: imp_shuffle<float2,float4,0x02><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 0x03: imp_shuffle<float2,float4,0x03><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 0x10: imp_shuffle<float2,float4,0x10><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 0x11: imp_shuffle<float2,float4,0x11><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 0x12: imp_shuffle<float2,float4,0x12><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 0x13: imp_shuffle<float2,float4,0x13><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 0x20: imp_shuffle<float2,float4,0x20><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 0x21: imp_shuffle<float2,float4,0x21><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 0x22: imp_shuffle<float2,float4,0x22><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 0x23: imp_shuffle<float2,float4,0x23><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 0x30: imp_shuffle<float2,float4,0x30><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 0x31: imp_shuffle<float2,float4,0x31><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 0x32: imp_shuffle<float2,float4,0x32><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 0x33: imp_shuffle<float2,float4,0x33><<<dst.blocks(), dst.threads()>>>(src, dst); break;
    }
    GPU_CHECK_ERROR();
    return dst;
}


/*
gpu_image<float4> gpu_shuffle( const gpu_image<float2>& src, int x, int y, int z, int w ) {
    gpu_image<float4> dst(src.size());
    switch (((w & 1) << 12) | ((z & 1) << 8) | ((y & 1) << 4) | (x & 1)) {
        case 0x0000: imp_shuffle<float4,float2,0x0000><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 0x0001: imp_shuffle<float4,float2,0x0001><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 0x0010: imp_shuffle<float4,float2,0x0010><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 0x0011: imp_shuffle<float4,float2,0x0011><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 0x0100: imp_shuffle<float4,float2,0x0100><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 0x0101: imp_shuffle<float4,float2,0x0101><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 0x0110: imp_shuffle<float4,float2,0x0110><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 0x0111: imp_shuffle<float4,float2,0x0111><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 0x1000: imp_shuffle<float4,float2,0x1000><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 0x1001: imp_shuffle<float4,float2,0x1001><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 0x1010: imp_shuffle<float4,float2,0x1010><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 0x1011: imp_shuffle<float4,float2,0x1011><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 0x1100: imp_shuffle<float4,float2,0x1100><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 0x1101: imp_shuffle<float4,float2,0x1101><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 0x1110: imp_shuffle<float4,float2,0x1110><<<dst.blocks(), dst.threads()>>>(src, dst); break;
        case 0x1111: imp_shuffle<float4,float2,0x1111><<<dst.blocks(), dst.threads()>>>(src, dst); break;
    }
    GPU_CHECK_ERROR();
    return dst;
}
*/