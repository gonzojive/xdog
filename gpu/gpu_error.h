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

#include <cuda_runtime_api.h>

extern void gpu_error_msg(cudaError_t err, const char *file, size_t line);

#define GPU_CHECK_ERROR() \
    { cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) gpu_error_msg(err, __FILE__, __LINE__); }

#define GPU_SAFE_CALL( call ) \
    { cudaError_t err = call; if (err != cudaSuccess) gpu_error_msg(err, __FILE__, __LINE__); }
