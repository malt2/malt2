/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef SEGVECGPU_HPP
#define SEGVECGPU_HPP
/** @file
 * support Dstorm segment buffer on GPU
 */
#include "dstorm_fwd.hpp"

#include "dstorm.hpp"
#include "segInfo.hpp"
#include "dstorm_msg.hpp"       // dStorm::seg::VecGpu is there
//#include "segInfo.hh"
//#include "segImpl.hh"

namespace dStorm {
    namespace user {

        template< typename T > class Seg_VecGpu;

#if defined(__CUDACC__)
        template< typename T>
            class Seg_VecGpu
            : public detail::SegBase< Seg_VecGpu<T>, T>
            {
            public:
                typedef detail::SegBase< Seg_VecGpu<T>, T> Base;
                typedef SegInfo SegInfo_t; // a.k.a. Base::SegInfo_t
                typedef typename seg::VecGpu<T> Fmt;

                Seg_VecGpu( SegInfo_t::SegPrintf segPrintf = nullptr );
                virtual ~Seg_VecGpu();

                /** SegInfo construction helper -- must set datasize and datacode fields */
                void setSegInfo();
                /**  After SegInfo setup, SegVecGpu wants a const GPU copy of its SegInfoPOD data. */
                void add_segment();
                void delete_segment();

                /** set up obuf msg headers and data for a \c push.
                 * \p iter GPU memory of raw data for segment obuf.
                 * \p cnt  how many raw data items to store
                 *
                 * - ohoh:
                 *   - return value allocated by CudaMalloc !?
                 *   - 3 sequential kernels !?
                 *
                 * - note: cuda_pair is \e only defined for cuda compilations !
                 */
                template< typename S >
                    cu_pair< uint32_t /*cnt*/, void* /*dataEnd*/ > *
                    store( S* iter, uint32_t cnt,
                           uint32_t const offset, void* const buf,
                           double const wgt=1.0 );
                void push() {}
                //uint32_t reduce(uint32_t* const rbufnums, uint32_t* size) const;
                uint32_t reduce(uint32_t* size) const;

            private:
                /** const copy of this segment's SegInfoPOD data on GPU,
                 * to avoid excessive PCI data during store/push/reduce */
                SegInfoPOD const * dev_segInfoPod;
            };
#endif // __CUDACC__

    }//user::
}//dStorm::
#endif // SEGVECGPU_HPP
