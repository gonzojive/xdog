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

gpu_image<float2> gpu_grad_central_diff( const gpu_image<float>& src, bool normalize );
gpu_image<float> gpu_grad_scharr_for_axis( const gpu_image<float>& src, int axis );
gpu_image<float2> gpu_grad_scharr( const gpu_image<float>& src, bool normalize );

gpu_image<float2> gpu_grad_to_axis( const gpu_image<float2>& src, bool squared );
gpu_image<float2> gpu_grad_from_axis( const gpu_image<float2>& src, bool squared );
gpu_image<float> gpu_grad_to_angle( const gpu_image<float2>& src );
gpu_image<float4> gpu_grad_to_lfm( const gpu_image<float2>& src );

gpu_image<float> gpu_grad_sobel_mag( const gpu_image<float>& src );
gpu_image<float> gpu_grad_sobel_mag( const gpu_image<float4>& src );
