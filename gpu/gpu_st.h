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
#pragma once

#include "gpu_image.h"

gpu_image<float4> gpu_st_central_diff( const gpu_image<float>& src );
gpu_image<float4> gpu_st_gaussian( const gpu_image<float>& src, float rho );
gpu_image<float4> gpu_st_gaussian( const gpu_image<float4>& src, float rho );
gpu_image<float4> gpu_st_sobel( const gpu_image<float>& src );
gpu_image<float4> gpu_st_sobel( const gpu_image<float4>& src );
gpu_image<float4> gpu_st_scharr( const gpu_image<float>& src );
gpu_image<float4> gpu_st_scharr( const gpu_image<float4>& src );
gpu_image<float4> gpu_st_from_gradient( const gpu_image<float2>& src );
gpu_image<float4> gpu_st_from_tangent( const gpu_image<float2>& src );

gpu_image<float4> gpu_st_tfm( const gpu_image<float4>& st );
gpu_image<float4> gpu_st_lfm( const gpu_image<float4>& st, float alpha=0 );
gpu_image<float> gpu_st_angle( const gpu_image<float4>& st );
gpu_image<float> gpu_st_anisotropy( const gpu_image<float4>& st );

gpu_image<float4> gpu_st_threshold_mag( const gpu_image<float4>& st, float threshold );
gpu_image<float4> gpu_st_normalize( const gpu_image<float4>& st );
gpu_image<float4> gpu_st_flatten( const gpu_image<float4>& st );
gpu_image<float4> gpu_st_rotate( const gpu_image<float4>& st, float angle );


inline __host__ __device__ float st2angle(const float4 g) {
    //return 0.5f * atan2(2 * g.z, g.x - g.y) + CUDART_PIO2_F;
    return 0.5f * atan2(-2 * g.z, g.y - g.x);
}


inline __host__ __device__ float2 st2tangent(const float4 g) {
    float phi = st2angle(g);
    return make_float2(cosf(phi), sinf(phi));
}


inline __host__ __device__ float2 st2gradient(const float4 g) {
    float phi = 0.5f * atan2(2 * g.z, g.x - g.y);
    return make_float2(cosf(phi), sinf(phi));
}


inline __host__ __device__ float2 st2lambda(float4 g) {
    float a = 0.5f * (g.y + g.x); 
    float b = 0.5f * sqrtf(fmaxf(0.0f, g.y*g.y - 2*g.x*g.y + g.x*g.x + 4*g.z*g.z));
    return make_float2(a + b, a - b);
}


inline __host__ __device__ float4 st2tfm(const float4 g) {
    float2 l = st2lambda(g);
    float2 t = st2tangent(g);
    return make_float4(t.x, t.y, l.x, l.y);
}


inline __host__ __device__ float tfm2A(float4 t) {
    float lambda1 = t.z;
    float lambda2 = t.w;
    return (lambda1 + lambda2 > 0)?
        (lambda1 - lambda2) / (lambda1 + lambda2) : 0;
}


inline __host__ __device__ float st2A(float4 g) {
    float a = 0.5f * (g.y + g.x); 
    float b = 0.5f * sqrtf(fmaxf(0.0f, g.y*g.y - 2*g.x*g.y + g.x*g.x + 4*g.z*g.z));
    float lambda1 = a + b;
    float lambda2 = a - b;
    return (lambda1 + lambda2 > 0)?
        (lambda1 - lambda2) / (lambda1 + lambda2) : 0;
}


inline __host__ __device__ float4 st2lfm(float4 g) {
    float2 t = st2tangent(g);
    return make_float4( t.x, t.y, 1, 1 );
}


inline __host__ __device__ float4 st2lfm(float4 g, float alpha) {
    float2 t = st2tangent(g);
    float A = st2A(g);
    return make_float4( 
        t.x, 
        t.y, 
        clamp((alpha + A) / alpha, 0.1f, 2.0f), 
        clamp(alpha / (alpha + A), 0.1f, 2.0f)
    );
}
