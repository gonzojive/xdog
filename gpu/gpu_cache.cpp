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
#include "gpu_cache.h"
#include "cuda_runtime.h"
#ifdef _MSC_VER
#include <unordered_map>
#else
#include <tr1/unordered_map>
#endif


struct entry_t {
    void *ptr;
    unsigned pitch;
    unsigned w;
    unsigned h;
};

typedef std::tr1::unordered_multimap<unsigned long long, entry_t> cache_map;
static cache_map g_cache;
static size_t g_cache_size = 0;
static size_t g_total_size = 0;


void gpu_cache_alloc( void **ptr, unsigned *pitch, unsigned w, unsigned h ) {
    unsigned long long key = ((unsigned long long)h << 32) | (unsigned long long)w;
    cache_map::iterator i = g_cache.find(key);
    if (i != g_cache.end()) {
        *ptr = i->second.ptr;
        *pitch = i->second.pitch;
        g_cache.erase(i);
        g_cache_size -= *pitch * h;
    } else {
        size_t tmp;
        cudaMallocPitch(ptr, &tmp, w, h);
        *pitch = (unsigned)tmp;
        g_total_size += *pitch * h;
    }
}


void gpu_cache_free( void *ptr, unsigned pitch, unsigned w, unsigned h ) {
    entry_t e = { ptr, pitch, w, h };
    unsigned long long key = ((unsigned long long)h << 32) | (unsigned long long)w;
    g_cache.insert(cache_map::value_type(key, e));
    g_cache_size += pitch * h;
}


void gpu_cache_clear() {
    cache_map::iterator i;
    for (i = g_cache.begin(); i != g_cache.end(); ++i) {
        g_total_size -= i->second.pitch * i->second.h;
        cudaFree(i->second.ptr);
    }
    g_cache.clear();
    g_cache_size = 0;
}


size_t gpu_cache_size() {
    return g_cache_size;
}


size_t gpu_cache_total() {
    return g_total_size;
}
