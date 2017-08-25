/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
/** \file
 * identical to segnInfo.cpp, except compiled with different compiler
 *
 * New reorganization requires some explicit instantiaons into library for segVecGpu codes
 *
 * This MIGHT be because user forgot to include segImpl.hh ???
 */

#if defined(__CUDACC__)
// instantiate some constexpr SegInfo:: data tables
#include "segInfo.cpp"

#include "dstorm_msg.hpp"
#include "segImpl.hh"
#include "segVecGpu.cuh"

namespace dStorm {

    namespace seg {
        template class VecGpu<double>;
        template class VecGpu<float>;
    }//seg::

    namespace user {
        template class Seg_VecGpu<float>;
        template class Seg_VecGpu<double>;

        // some of above classes have internal template functions that should
        // have "common varieties" instantiated into the library.
        //
        //  store: Seg_VecGpu<T>::store( S* iIter, ...)
        //         for T,S combinations of float, double
        //
        // BUT... this did NOT add the functions to the library! why?
        //  oh... need const iterator versions:
        //        float/double const *iIter
#define SEG_VECGPU__STORE(T,S) \
        template cu_pair< uint32_t /*cnt*/, void* /*dataEnd*/ >* \
            Seg_VecGpu<T>::store( S const* iIter, \
                    uint32_t cnt, uint32_t const offset, \
                    void* const buf, double const wgt/*=1.0*/ )
        SEG_VECGPU__STORE(float,float);
        SEG_VECGPU__STORE(float,double);
        SEG_VECGPU__STORE(double,float);
        SEG_VECGPU__STORE(double,double);
#undef SEG_VECGPU__STORE

        // You may need to add more if you use fancier iterators, for example.
        // OR you might include the template definitions of segVecGpu.cuh into
        //    and instantiate in your own code (using same cuda nvcc version)
    }//user::

    namespace detail {
        template class SegImpl<seg::VecGpu<double>>;
        template class SegImpl<seg::VecGpu<float>>;
#define SEGIMPL__STORE(T,S) \
        template void SegImpl<seg::VecGpu<T>>::store<S const*> \
            (S const*,uint32_t const,uint32_t const,double const);
        SEGIMPL__STORE(float,float);
        SEGIMPL__STORE(float,double);
        SEGIMPL__STORE(double,float);
        SEGIMPL__STORE(double,double);
#undef SEGIMPL__STORE
    }//detail::


}//dStorm::
#endif
