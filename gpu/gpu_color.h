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

gpu_image<float> gpu_srgb2linear( const gpu_image<float>& src);
gpu_image<float> gpu_linear2srgb( const gpu_image<float>& src);

gpu_image<float4> gpu_srgb2linear( const gpu_image<float4>& src);
gpu_image<float4> gpu_linear2srgb( const gpu_image<float4>& src);

gpu_image<float4> gpu_rgb2lab( const gpu_image<float4>& src);
gpu_image<float4> gpu_lab2rgb( const gpu_image<float4>& src);
gpu_image<float4> gpu_l2rgb( const gpu_image<float>& src );

gpu_image<float4> gpu_gray2rgb( const gpu_image<float>& src, bool saturate=true );
gpu_image<float> gpu_rgb2gray( const gpu_image<float4>& src );

gpu_image<float4> gpu_swap_rgba( const gpu_image<float4>& src );
