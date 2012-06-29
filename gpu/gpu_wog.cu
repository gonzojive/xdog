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
#include "gpu_wog.h"
#include "gpu_color.h"
#include "gpu_bilateral.h"
#include "gpu_blend.h"
#include "gpu_shuffle.h"
#include "gpu_grad.h"
#include "gpu_gauss.h"
#include "gpu_binder.h"
#include <algorithm>


static texture<float, 2, cudaReadModeElementType> texSRC1;
static texture<float4, 2, cudaReadModeElementType> texSRC4;


__global__ void imp_wog_dog( gpu_plm2<float> dst, float sigma_e, float sigma_r, 
                             float tau, float phi_e, float epsilon, float precision ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;

    float twoSigmaE2 = 2.0f * sigma_e * sigma_e;
    float twoSigmaR2 = 2.0f * sigma_r * sigma_r;
    int halfWidth = int(ceilf( precision * sigma_r ));

    float sumE = 0;
    float sumR = 0;
    float2 norm = make_float2(0);

    for ( int i = -halfWidth; i <= halfWidth; ++i ) {
        for ( int j = -halfWidth; j <= halfWidth; ++j ) {
            float d = length(make_float2(i,j));
            float kE = __expf(-d *d / twoSigmaE2);
            float kR = __expf(-d *d / twoSigmaR2);
            
            float c = tex2D(texSRC1, ix + i, iy + j);
            sumE += kE * c;
            sumR += kR * c;
            norm += make_float2(kE, kR);
        }
    }

    sumE /= norm.x;
    sumR /= norm.y;

    float H = sumE - tau * sumR;
    float edge = ( H > epsilon )? 1 : 1 + tanhf( phi_e * (H - epsilon) );
    dst(ix, iy) = clamp(edge, 0.0f, 1.0f);
}


gpu_image<float> gpu_wog_dog( const gpu_image<float>& src, float sigma_e, float sigma_r, 
                              float tau, float phi_e, float epsilon, float precision ) 
{
    gpu_image<float> dst(src.size());
    bind(&texSRC1, src);
    imp_wog_dog<<<dst.blocks(), dst.threads()>>>(dst, sigma_e, sigma_r, tau, phi_e, epsilon, precision);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_wog_luminance_quant( const gpu_plm2<float4> src, gpu_plm2<float4> dst, int nbins, float phi_q) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;

    float3 c = make_float3(src(ix, iy));
    float delta_q = 100.01f / nbins;
    float qn = delta_q * (floor(c.x / delta_q) + 0.5f);
    float qc = qn + 0.5f * delta_q * tanhf(phi_q * (c.x - qn));
    dst(ix, iy) = make_float4( qc, c.y, c.z, 1 );
}


gpu_image<float4> gpu_wog_luminance_quant( const gpu_image<float4>& src, int nbins, float phi_q) {
    gpu_image<float4> dst(src.size());
    imp_wog_luminance_quant<<<dst.blocks(), dst.threads()>>>(src, dst, nbins, phi_q);
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_wog_luminance_quant( gpu_plm2<float4> dst, int nbins,
                                         float lambda_delta, float omega_delta,
                                         float lambda_phi, float omega_phi ) 
{
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;

    float3 c = make_float3(tex2D(texSRC4, ix, iy));               
    float gx = 0.5f * (tex2D(texSRC4, ix - 1, iy).x - tex2D(texSRC4, ix + 1, iy).x);
    float gy = 0.5f * (tex2D(texSRC4, ix, iy - 1).x - tex2D(texSRC4, ix, iy + 1).x);
    float grad = sqrtf(gx * gx + gy * gy);
    grad = clamp(grad, lambda_delta, omega_delta);
    grad = (grad - lambda_delta) / (omega_delta - lambda_delta);
    
    float phi_q = lambda_phi + grad * (omega_phi - lambda_phi);
    float delta_q = 100.01f / nbins;
    float qn = delta_q * (floor(c.x / delta_q) + 0.5f);
    float qc = qn + 0.5f * delta_q * tanhf(phi_q * (c.x - qn));

    dst(ix, iy) = make_float4( qc, c.y, c.z, 1 );
}


gpu_image<float4> gpu_wog_luminance_quant( const gpu_image<float4>& src, int nbins, 
                                           float lambda_delta, float omega_delta,
                                           float lambda_phi, float omega_phi )
{
    gpu_image<float4> dst(src.size());
    gpu_binder<float4> src_(texSRC4, src);
    imp_wog_luminance_quant<<<dst.blocks(), dst.threads()>>>( dst, nbins, lambda_delta, omega_delta, 
                                                              lambda_phi, omega_phi );
    GPU_CHECK_ERROR();
    return dst;
}


__global__ void imp_wog_warp( gpu_plm2<float4> dst, float phi_w ) 
{
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;

    float gx = 0.5f * (tex2D(texSRC1, ix - 1, iy) - tex2D(texSRC1, ix + 1, iy));
    float gy = 0.5f * (tex2D(texSRC1, ix, iy - 1) - tex2D(texSRC1, ix, iy + 1));

    float4 c = tex2D(texSRC4, ix + gx * phi_w, iy + gy * phi_w);
    dst(ix, iy) = c;
}


gpu_image<float4> gpu_wog_warp(const gpu_image<float4>& src, gpu_image<float>& edges, float phi_w) {
    gpu_image<float4> dst(src.size());
    gpu_binder<float4> src_(texSRC4, src, cudaFilterModeLinear);
    gpu_binder<float> edges_(texSRC1, edges);
    imp_wog_warp<<<dst.blocks(), dst.threads()>>>( dst, phi_w );
    GPU_CHECK_ERROR();
    return dst;
}


gpu_image<float4> gpu_wog_warp_sharp( const gpu_image<float4>& src, 
                                      float sigma_w, float precision_w, float phi_w) 
{
    gpu_image<float> S = gpu_grad_sobel_mag(src);
    S = gpu_gauss_filter_xy(S, sigma_w, precision_w);
    return gpu_wog_warp(src, S, phi_w);
}


gpu_image<float4> gpu_wog_abstraction( const gpu_image<float4>& src, int n_e, int n_a,
                                       float sigma_d, float sigma_r,
                                       float sigma_e1, float sigma_e2, float precision_e,
                                       float tau, float phi_e, float epsilon,
                                       bool adaptive_quant,
                                       int nbins, float phi_q,
                                       float lambda_delta, float omega_delta, 
                                       float lambda_phi, float omega_phi,
                                       bool warp_sharp, float sigma_w,
                                       float precision_w, float phi_w )
{
    gpu_image<float4> img;
    gpu_image<float> L;

    {
        img = gpu_rgb2lab(src);
        gpu_image<float4> E = img;
        gpu_image<float4> A = img;

        int N = std::max(n_e, n_a);
        for (int i = 0; i < N; ++i) {
            img = gpu_bilateral_filter(img, sigma_d, sigma_r);
            if (i == (n_e - 1)) E = img;
            if (i == (n_a - 1)) A = img;
        }
        img = A;

        L = gpu_shuffle(E, 0);
        L = gpu_wog_dog( L, sigma_e1, sigma_e2, tau, phi_e, epsilon, precision_e );
    }

    if (adaptive_quant) {
        img = gpu_wog_luminance_quant( img, nbins, lambda_delta, omega_delta, lambda_phi, omega_phi );
    } else {
        img = gpu_wog_luminance_quant( img, nbins, phi_q );
    }
    
    img = gpu_lab2rgb(img);
    img = gpu_blend_intensity(img, L, GPU_BLEND_MULTIPLY);

    if (warp_sharp) {
        img = gpu_wog_warp_sharp(img, sigma_w, precision_w, phi_w);
    }

    return img;
}
