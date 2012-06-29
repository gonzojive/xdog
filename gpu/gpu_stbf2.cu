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
#include "gpu_stbf2.h"
#include "gpu_stgauss2.h"
#include "gpu_st.h"
#include "gpu_binder.h"


static texture<float, 2, cudaReadModeElementType> s_texSRC1;
static texture<float4, 2, cudaReadModeElementType> s_texSRC4;
static texture<float4, 2, cudaReadModeElementType> s_texST;

template <typename T> struct texSRC {
    inline __device__ T operator()(float x, float y) const;
};
template<> inline __device__ float texSRC<float>::operator()(float x, float y) const { return tex2D(s_texSRC1, x, y); } 
template<> inline __device__ float4 texSRC<float4>::operator()(float x, float y) const { return tex2D(s_texSRC4, x, y); } 

struct texST {
   inline __device__ float4 operator()(float x, float y) const { return tex2D(s_texST, x, y); } 
};
struct texSTr {
    texSTr(float2 s) : s_(s) {}
    inline __device__ float4 operator()(float x, float y) const { return tex2D(s_texST, s_.x * x, s_.y * y); } 
    float2 s_;
};

template <typename SRC>
struct stbf2_filter1 {
    typedef float T;

     __device__ stbf2_filter1(const SRC& src, T c0, float sigma_d, float sigma_r, float precision) 
         : src_(src) 
     { 
        radius_ = precision * sigma_d;
        twoSigmaD2_ = 2 * sigma_d * sigma_d;
        twoSigmaR2_ = 2 * sigma_r * sigma_r;
        c0_ = c0;
        c_ = make_zero<T>();
        w_ = 0;
    }

    __device__ float radius() const {
        return radius_;
    }

    __device__ void operator()(float sign, float u, float2 p) {
        T c1 = src_(p.x, p.y);
        T r = c1 - c0_;
        float kd = __expf(-u * u / twoSigmaD2_);
        float kr = __expf(-dot(r,r) / twoSigmaR2_);
        c_ += kd * kr * c1;
        w_ += kd * kr;
    }

    const SRC& src_;
    float radius_;
    float twoSigmaD2_;
    float twoSigmaR2_;
    T c0_;
    T c_;
    float w_;
};


template <typename SRC>
struct stbf2_filter4 {
    typedef float3 T;

     __device__ stbf2_filter4(const SRC& src, T c0, float sigma_d, float sigma_r, float precision) 
         : src_(src) 
     { 
        radius_ = precision * sigma_d;
        twoSigmaD2_ = 2 * sigma_d * sigma_d;
        twoSigmaR2_ = 2 * sigma_r * sigma_r;
        c0_ = c0;
        c_ = make_zero<T>();
        w_ = 0;
    }

    __device__ float radius() const {
        return radius_;
    }

    __device__ void operator()(float sign, float u, float2 p) {
        T c1 = make_float3(src_(p.x, p.y));
        T r = c1 - c0_;
        float kd = __expf(-u * u / twoSigmaD2_);
        float kr = __expf(-dot(r,r) / twoSigmaR2_);
        c_ += kd * kr * c1;
        w_ += kd * kr;
    }

    const SRC& src_;
    float radius_;
    float twoSigmaD2_;
    float twoSigmaR2_;
    T c0_;
    T c_;
    float w_;
};


template<int order, typename SRC, typename ST> 
__global__ void imp_stbf2_filter1( gpu_plm2<float> dst, const SRC src, const ST st, float sigma_d, float sigma_r, float precision, 
                                   float cos_max, bool adaptive, float step_size ) 
{
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h) 
        return;

    float2 p0 = make_float2(ix + 0.5f, iy + 0.5f);
    if (adaptive) {
        float A = st2A(st(p0.x, p0.y));
        sigma_d *= 0.25f * (1 + A)*(1 + A);
    }

    float c0 = src(p0.x, p0.y);
    stbf2_filter1<SRC> f(src, c0, sigma_d, sigma_r, precision);
    if (order == 1) st_integrate_euler(p0, st, f, cos_max, dst.w, dst.h, step_size);
    if (order == 2) st_integrate_rk2(p0, st, f, cos_max, dst.w, dst.h, step_size);
    if (order == 4) st_integrate_rk4(p0, st, f, cos_max, dst.w, dst.h, step_size);
    dst(ix, iy) = f.c_ / f.w_;
}


template<int order, typename SRC, typename ST> 
__global__ void imp_stbf2_filter4( gpu_plm2<float4> dst, const SRC src, const ST st, float sigma_d, float sigma_r, float precision, 
                                   float cos_max, bool adaptive, float step_size ) 
{
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h) 
        return;

    float2 p0 = make_float2(ix + 0.5f, iy + 0.5f);
    if (adaptive) {
        float A = st2A(st(p0.x, p0.y));
        sigma_d *= 0.25f * (1 + A)*(1 + A);
    }

    float3 c0 = make_float3(src(p0.x, p0.y));
    stbf2_filter4<SRC> f(src, c0, sigma_d, sigma_r, precision);
    if (order == 1) st_integrate_euler(p0, st, f, cos_max, dst.w, dst.h, step_size);
    if (order == 2) st_integrate_rk2(p0, st, f, cos_max, dst.w, dst.h, step_size);
    if (order == 4) st_integrate_rk4(p0, st, f, cos_max, dst.w, dst.h, step_size);
    dst(ix, iy) = make_float4(f.c_ / f.w_, 1);
}


gpu_image<float> gpu_stbf2_filter( const gpu_image<float>& src, const gpu_image<float4>& st, 
                                   float sigma_d, float sigma_r, float precision, float max_angle, bool adaptive,
                                   bool src_linear, bool st_linear, int order, float step_size )
{     
    if (sigma_d <= 0) return src;
    gpu_image<float> dst(src.size());

    gpu_binder<float> src_(s_texSRC1, src, src_linear? cudaFilterModeLinear : cudaFilterModePoint);
    gpu_binder<float4> st_(s_texST, st, st_linear? cudaFilterModeLinear : cudaFilterModePoint);
    float cos_max = cosf(radians(max_angle));

    if (src.size() == st.size()) {
        if (order == 1) imp_stbf2_filter1<1><<<dst.blocks(), dst.threads()>>>(dst, texSRC<float>(), texST(), sigma_d, sigma_r, precision, cos_max, adaptive, step_size);
        else if (order == 2) imp_stbf2_filter1<2><<<dst.blocks(), dst.threads()>>>(dst, texSRC<float>(), texST(), sigma_d, sigma_r, precision, cos_max, adaptive, step_size);
        else if (order == 4) imp_stbf2_filter1<4><<<dst.blocks(), dst.threads()>>>(dst, texSRC<float>(), texST(), sigma_d, sigma_r, precision, cos_max, adaptive, step_size);
    } else {
        float2 s = make_float2((float)st.w() / src.w(), (float)st.h() / src.h());
        if (order == 1) imp_stbf2_filter1<1><<<dst.blocks(), dst.threads()>>>(dst, texSRC<float>(), texSTr(s), sigma_d, sigma_r, precision, cos_max, adaptive, step_size);
        else if (order == 2) imp_stbf2_filter1<2><<<dst.blocks(), dst.threads()>>>(dst, texSRC<float>(), texSTr(s), sigma_d, sigma_r, precision, cos_max, adaptive, step_size);
        else if (order == 4) imp_stbf2_filter1<4><<<dst.blocks(), dst.threads()>>>(dst, texSRC<float>(), texSTr(s), sigma_d, sigma_r, precision, cos_max, adaptive, step_size);
    }
    GPU_CHECK_ERROR();
    return dst;
}


gpu_image<float4> gpu_stbf2_filter( const gpu_image<float4>& src, const gpu_image<float4>& st, 
                                    float sigma_d, float sigma_r, float precision, float max_angle, bool adaptive,
                                    bool src_linear, bool st_linear, int order, float step_size )
{     
    if (sigma_d <= 0) return src;
    gpu_image<float4> dst(src.size());

    gpu_binder<float4> src_(s_texSRC4, src, src_linear? cudaFilterModeLinear : cudaFilterModePoint);
    gpu_binder<float4> st_(s_texST, st, st_linear? cudaFilterModeLinear : cudaFilterModePoint);
    float cos_max = cosf(radians(max_angle));

    if (src.size() == st.size()) {
        if (order == 1) imp_stbf2_filter4<1><<<dst.blocks(), dst.threads()>>>(dst, texSRC<float4>(), texST(), sigma_d, sigma_r, precision, cos_max, adaptive, step_size);
        else if (order == 2) imp_stbf2_filter4<2><<<dst.blocks(), dst.threads()>>>(dst, texSRC<float4>(), texST(), sigma_d, sigma_r, precision, cos_max, adaptive, step_size);
        else if (order == 4) imp_stbf2_filter4<4><<<dst.blocks(), dst.threads()>>>(dst, texSRC<float4>(), texST(), sigma_d, sigma_r, precision, cos_max, adaptive, step_size);
    } else {
        float2 s = make_float2((float)st.w() / src.w(), (float)st.h() / src.h());
        if (order == 1) imp_stbf2_filter4<1><<<dst.blocks(), dst.threads()>>>(dst, texSRC<float4>(), texSTr(s), sigma_d, sigma_r, precision, cos_max, adaptive, step_size);
        else if (order == 2) imp_stbf2_filter4<2><<<dst.blocks(), dst.threads()>>>(dst, texSRC<float4>(), texSTr(s), sigma_d, sigma_r, precision, cos_max, adaptive, step_size);
        else if (order == 4) imp_stbf2_filter4<4><<<dst.blocks(), dst.threads()>>>(dst, texSRC<float4>(), texSTr(s), sigma_d, sigma_r, precision, cos_max, adaptive, step_size);
    }
    GPU_CHECK_ERROR();
    return dst;
}
