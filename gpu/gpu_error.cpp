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
#include "gpu_error.h"
#ifdef WIN32
#include <sstream>
#include <windows.h>
#else
#include <cstdio>
#include <cstdlib>
#endif


void gpu_error_msg(cudaError_t err, const char *file, size_t line) {
#ifdef WIN32
    if (!IsDebuggerPresent()) {
        std::ostringstream oss;
        oss << cudaGetErrorString(err) << "\n"
            << file << "(" << line << ")";
        MessageBoxA(NULL, oss.str().c_str(), "CUDA Error", MB_OK | MB_ICONERROR);
    } else {
        OutputDebugStringA("CUDA Error: ");
        OutputDebugStringA(cudaGetErrorString(err));
        OutputDebugStringA("\n");
        DebugBreak();
    }
#else
    fprintf(stderr, "%s(%d): CUDA Error\n", file, (int)line);
    fprintf(stderr, "%s\n", cudaGetErrorString(err));
#endif
    exit(1);
}
