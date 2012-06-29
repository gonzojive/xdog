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

gpu_image<float> gpu_wog_dog( const gpu_image<float>& src, float sigma_e, float sigma_r, 
                              float tau, float phi_e, float epsilon = 0, float precision = 2 );

gpu_image<float4> gpu_wog_luminance_quant( const gpu_image<float4>& src, int nbins, float phi_q );
gpu_image<float4> gpu_wog_luminance_quant( const gpu_image<float4>& src, int nbins, 
                                           float lambda_delta, float omega_delta,
                                           float lambda_phi, float omega_phi );

gpu_image<float4> gpu_wog_warp(const gpu_image<float4>& src, gpu_image<float>& edges, float phi_w);
gpu_image<float4> gpu_wog_warp_sharp( const gpu_image<float4>& src, float sigma_w, float precision_w, float phi_w);

gpu_image<float4> gpu_wog_abstraction( const gpu_image<float4>& src, int n_e = 1, int n_a = 4,
                                       float sigma_d = 3, float sigma_r = 4.25f,
                                       float sigma_e1 = 1, float sigma_e2 = 1.6f, float precision_e = 2,
                                       float tau = 0.99f, float phi_e = 2, float epsilon=0,
                                       bool adaptive_quant = true,
                                       int nbins = 8, float phi_q = 2,
                                       float lambda_delta = 0, float omega_delta = 2, 
                                       float lambda_phi = 0.9f, float omega_phi = 1.6f,
                                       bool warp_sharp = true, float sigma_w = 1.5f,
                                       float precision_w = 2, float phi_w = 2.7 );
