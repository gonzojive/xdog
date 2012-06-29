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

#ifdef __CUDACC__

template<typename T, texture<T,2>& texture_ref()> 
struct gpu_sampler {
    __host__ gpu_sampler(const gpu_image<T>& img, cudaTextureFilterMode filter_mode=cudaFilterModePoint) {
        texture_ref().filterMode = filter_mode;
        GPU_SAFE_CALL(cudaBindTexture2D(0, texture_ref(), img.ptr(), img.w(), img.h(), img.pitch()));
    }

    __host__ ~gpu_sampler() {
        texture_ref().filterMode = cudaFilterModePoint;
        cudaUnbindTexture(texture_ref());
    }
    
    __device__ T operator()(float x, float y) const { 
        return tex2D(texture_ref(), x, y); 
    } 
};


template<typename T, texture<T,2>& texture_ref()> 
struct gpu_resampler {
    float2 s_;

    __host__ gpu_resampler(const gpu_image<T>& img, float2 s, cudaTextureFilterMode filter_mode=cudaFilterModePoint) {
        s_ = s;
        texture_ref().filterMode = filter_mode;
        GPU_SAFE_CALL(cudaBindTexture2D(0, texture_ref(), img.ptr(), img.w(), img.h(), img.pitch()));
    }

    __host__ ~gpu_resampler() {
        texture_ref().filterMode = cudaFilterModePoint;
        cudaUnbindTexture(texture_ref());
    }

    __device__ T operator()(float x, float y) const { 
        return tex2D(texture_ref(), s_.x * x, s_.x * y); 
    } 
};


template <typename T> 
struct gpu_constant_sampler {
    T value;

    gpu_constant_sampler(T v) : value(v) { }
    
    __host__ __device__ T operator()(float ix, float iy) const { 
        return value; 
    }
};

#endif
