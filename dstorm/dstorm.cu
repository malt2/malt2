/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
/** \file
 * forcibly instantiate some template functions into the library ???
 */
#include "dstorm.hh"
#include "dstorm_msg.hpp"
#include "segVecGpu.cuh"

#if 0 // could not get these to compile !!!
    template void  ::dStorm::Dstorm::store<::dStorm::seg::VecGpu<float>,float *>
    ( ::dStorm::SegNum const s, float const* iter, uint32_t const cnt,
      uint32_t const offset, double const wgt );
    template void  ::dStorm::Dstorm::store<::dStorm::seg::VecGpu<float>,float const*>
    ( ::dStorm::SegNum const s, float const* iter, uint32_t const cnt,
      uint32_t const offset, double const wgt );

namespace dStorm {

    // XXX this file might be unnecessary, if dstorm_any2.hh is included in user's nvcc compilation
    using seg::VecGpu;

    template void  Dstorm::store<seg::VecGpu<float>, double const*>
    ( SegNum const s, double const* iter, uint32_t const cnt,
      uint32_t const offset/*=0U*/, double const wgt/*=1.0*/ );

    template void  Dstorm::store<seg::VecGpu<double>, float const*>
    ( SegNum const s, float const* iter, uint32_t const cnt,
      uint32_t const offset/*=0U*/, double const wgt/*=1.0*/ );

    template void  Dstorm::store<seg::VecGpu<double>, double const*>
    ( SegNum const s, double const* iter, uint32_t const cnt,
      uint32_t const offset/*=0U*/, double const wgt/*=1.0*/ );

}//dStorm::
#endif
