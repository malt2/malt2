/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef DSTORM_GPU_HPP
#define DSTORM_GPU_HPP

#include "detail/dstorm_common.h"
#ifdef __CUDACC__
#include "cuda_runtime.h"
// Note: side-effect: for non-cuda compiler, __host__ and __device__ are elided
//       But we'd have to have the header always available, which it might not be.
#define SA_BLOCK 1024 // Previously 128
#define MAX_CU_BLOCKS 65536
#define CUDA_DBG 0

#else

//#warning "non-cuda compile: setting __host__ and __device__ to empty macros"
#define __host__
#define __device__
#endif

#ifdef __CUDACC__
#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaCheckError( const char *file, const int line )
{
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

#ifndef NDEBUG
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif
    return;
}

namespace dStorm {

    template <typename T, typename S>
        struct cu_pair {
            T first;
            S second;
        }; //pair to replace c++ structure 

}//dStorm::
#endif // __CUDACC__
#endif // DSTORM_GPU_HPP
