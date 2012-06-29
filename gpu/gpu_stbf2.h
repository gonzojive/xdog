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

gpu_image<float> gpu_stbf2_filter( const gpu_image<float>& src, const gpu_image<float4>& st, 
                                   float sigma_d, float sigma_r, float precision, float max_angle, bool adaptive,
                                   bool src_linear, bool st_linear, int order, float step_size );

gpu_image<float4> gpu_stbf2_filter( const gpu_image<float4>& src, const gpu_image<float4>& st, 
                                    float sigma_d, float sigma_r, float precision, float max_angle, bool adaptive,
                                    bool src_linear, bool st_linear, int order, float step_size );
