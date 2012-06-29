//
// by Jan Eric Kyprianidis and Daniel Müller
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

enum gpu_blend_mode {
    GPU_BLEND_NORMAL,
    GPU_BLEND_MULTIPLY,
    GPU_BLEND_SCREEN,
    GPU_BLEND_HARD_LIGHT,
    GPU_BLEND_SOFT_LIGHT,
    GPU_BLEND_OVERLAY,
    GPU_BLEND_LINEAR_BURN,
    GPU_BLEND_DIFFERENCE,
    GPU_BLEND_LINEAR_DODGE
};

gpu_image<float> gpu_blend( const gpu_image<float>& back,   const gpu_image<float>& src,
                             gpu_blend_mode mode );

gpu_image<float4> gpu_blend( const gpu_image<float4>& back, const gpu_image<float4>& src,
                             gpu_blend_mode mode );


//gpu_image<float> gpu_blend_intensity( const gpu_image<float>& back, const gpu_image<float>& src,
//                                       gpu_blend_mode mode,   float color = 1 );

gpu_image<float4> gpu_blend_intensity( const gpu_image<float4>& back, const gpu_image<float>& src,
                                       gpu_blend_mode mode, float4 color = make_float4(1));
