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
#include "gpu_noise.h"
#include <cstdlib>


static int randi() {
    static bool first = true;
    if (first) { 
        first = false; 
        std::srand(clock()); 
    }
    return std::rand();
}


static float randf() {
    return (float)randi()/RAND_MAX;
}


gpu_image<float> gpu_noise_random(unsigned w, unsigned h, float a, float b) {
    float *n = new float[w * h];
    float *p = n;
    float da = b - a;
    for (unsigned j = 0; j < h; ++j) {
        for (unsigned i = 0; i < w; ++i) {
            *p++ = a + da * randf();
        }
    }

    gpu_image<float> dst(n, w, h);
    delete[] n;
    return dst;
}
