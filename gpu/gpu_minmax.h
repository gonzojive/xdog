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

void gpu_minmax( const gpu_image<float>& src, float *pmin, float *pmax );
float gpu_min( const gpu_image<float>& src );
float gpu_max( const gpu_image<float>& src );

gpu_image<float> gpu_normalize(const gpu_image<float>& src);
gpu_image<float4> gpu_normalize_gray(const gpu_image<float4>& src);
