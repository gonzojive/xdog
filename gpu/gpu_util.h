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

extern gpu_image<float> gpu_adjust( const gpu_image<float>& src, float a, float b );
extern gpu_image<float4> gpu_adjust( const gpu_image<float4>& src, float4 a, float4 b );

extern gpu_image<float> gpu_invert( const gpu_image<float>& src);
extern gpu_image<float4> gpu_invert( const gpu_image<float4>& src);

extern gpu_image<float> gpu_saturate( const gpu_image<float>& src);
extern gpu_image<float4> gpu_saturate( const gpu_image<float4>& src);

extern gpu_image<float4> gpu_lerp( const gpu_image<float4>& src0, const gpu_image<float4>& src1, float t );
