//
// Created by 付聪 on 2017/6/21.
// Cross-platform adaptation: MSVC support
//

#ifndef EFANNA2E_UTIL_H
#define EFANNA2E_UTIL_H
#include <random>
#include <iostream>
#include <cstring>
#include <algorithm>
#ifdef _MSC_VER
#include <malloc.h>
#elif defined(__APPLE__)
#else
#include <malloc.h>
#endif

namespace efanna2e {

static void GenRandom(std::mt19937 &rng, unsigned *addr, unsigned size, unsigned N) {
    for (unsigned i = 0; i < size; ++i) {
        addr[i] = rng() % (N - size);
    }
    std::sort(addr, addr + size);
    for (unsigned i = 1; i < size; ++i) {
        if (addr[i] <= addr[i - 1]) {
            addr[i] = addr[i - 1] + 1;
        }
    }
    unsigned off = rng() % N;
    for (unsigned i = 0; i < size; ++i) {
        addr[i] = (addr[i] + off) % N;
    }
}

#if defined(__AVX__) || defined(__AVX2__)
#define DATA_ALIGN_FACTOR 8
#elif defined(__SSE2__)
#define DATA_ALIGN_FACTOR 4
#else
#define DATA_ALIGN_FACTOR 1
#endif

inline float* data_align(float* data_ori, unsigned point_num, unsigned& dim) {
    float* data_new = nullptr;
    unsigned new_dim = (dim + DATA_ALIGN_FACTOR - 1) / DATA_ALIGN_FACTOR * DATA_ALIGN_FACTOR;

#ifdef _MSC_VER
    data_new = (float*)_aligned_malloc(point_num * new_dim * sizeof(float),
                                        DATA_ALIGN_FACTOR * sizeof(float));
#elif defined(__APPLE__)
    data_new = new float[new_dim * point_num];
#else
    data_new = (float*)memalign(DATA_ALIGN_FACTOR * sizeof(float),
                                 point_num * new_dim * sizeof(float));
#endif

    for (unsigned i = 0; i < point_num; i++) {
        memcpy(data_new + i * new_dim, data_ori + i * dim, dim * sizeof(float));
        memset(data_new + i * new_dim + dim, 0, (new_dim - dim) * sizeof(float));
    }
    dim = new_dim;
#ifdef _MSC_VER
    _aligned_free(data_ori);
#elif defined(__APPLE__)
    delete[] data_ori;
#else
    free(data_ori);
#endif
    return data_new;
}

}

#endif // EFANNA2E_UTIL_H
