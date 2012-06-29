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

template <typename T>
struct gpu_plm2 {
    T *ptr;
    unsigned stride;
    unsigned w;
    unsigned h;

    __host__ gpu_plm2() {
        ptr = 0;
        stride = w = h = 0;
    }

    __host__ gpu_plm2(T *ptr, unsigned pitch, unsigned w, unsigned h) {
        this->ptr = ptr;
        this->stride = pitch / sizeof(T);
        this->w = w;
        this->h = h;
    }

    #ifdef __CUDACC__

    inline __device__ T& operator()(int x, int y) { 
        return ptr[y * stride + x];
    }

    inline __device__ const T& operator()(int x, int y) const { 
        return ptr[y * stride + x];
    }

    #endif
};
