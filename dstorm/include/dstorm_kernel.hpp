/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef DSTORM_KERNEL_HPP
#define DSTORM_KERNEL_HPP
/** \file
 * forward declarations of GPU kernels (host begins parallel function on GPU) */
#if !defined(__CUDACC__)
//#warning "skipping dstorm_kernel.hpp (non-cuda compiler)"
#else
#endif // __CUDACC__

#include "dstorm_fwd.hpp"
#include "dstorm_msg.hpp"

namespace dStorm {

    //
    // forward declarations: Note that these are required BEFORE segImpl.hh can be included
    //

    template <typename T>
    __global__ void store_init(T* iter, void* obuf,
                               MsgHeader<seg::Dummy>* mHdr, uint_least8_t const value);

    __global__ void store_impl(cu_pair<uint32_t, void*>* stop, uint32_t const cnt,
                                      MsgHeader<seg::Dummy>* mHdr, void* obuf,
                                      unsigned long bufBytes);

    __global__ void push_init(MsgHeader<seg::Dummy>* obufHdr, SegNum const s,
                                     uint_least32_t* d_result, uint_least8_t *d_pushes);

    __global__ void push_init(MsgHeader<seg::Dummy>* obufHdr, SegNum const s,
                                     uint_least32_t* d_result, uint_least8_t *d_pushes);

    __global__ void push_con(MsgHeader<seg::Dummy>* obufHdr, uint32_t nWrite);

    // NOTE: riter==nullptr actually reflects a compile-time decision with CHK_HDR_ITER
    template <typename FMT>
    __global__ void reduce_init(const detail::SegImpl<FMT>* info, uint32_t* total,
                                uint32_t size, uint32_t* riter = nullptr);

    template <typename FMT>
        __global__ void reduce_impl(const detail::SegImpl<FMT>* info, uint32_t size,
                                    uint32_t* riter=nullptr) ;
}//dStorm:: fwd declarations

#endif // DSTORM_KERNEL_HPP
