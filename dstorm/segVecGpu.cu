/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
/** @file
 * instantiate Seg_VecGpu float and double segments (into library)
 */
#if !defined(__CUDACC__)
#error "segVecGpu.cu must be compiled by a cuda compiler"
#endif

#include "segVecGpu.cuh"
#include "segInfo.hh"
#include "segImpl.hh"

namespace dStorm {
    namespace user{

        template class Seg_VecGpu<float>;
        template class Seg_VecGpu<double>;

    } // user::

    //template __global__ void reduce_wrapper(Seg_VecGpu<float>* d_vec, uint32_t* result);
    //template __global__ void reduce_wrapper(Seg_VecGpu<double>* d_vec, uint32_t* result);

    template class detail::SegBase< user::Seg_VecGpu<float>, float >;
    template class detail::SegBase< user::Seg_VecGpu<double>, double >;

} // dStorm::
