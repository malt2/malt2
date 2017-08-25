/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef DSTORM_KERNEL_CUH
#define DSTORM_KERNEL_CUH
#if defined(__CUDACC__)

#include "dstorm_kernel.hpp"

#include "segImpl.hh"		// unified CPU/GPU headers (also + segInfo + segImpl + segVecGpu overrides)

//#include "dstorm_msg.hh"	// to be usable will need full type for MsgHeader

namespace dStorm {

    //
    // template implementations (here, for now)
    //
    // Templated kernels probably should ONLY be instantiated for particular
    // types (i.e. seg::VecGpu / Seg_VecGpu / segImpl specific to GPU).
    //

    template <typename T>
        __global__ void store_init(T* iter, void* obuf, MsgHeader<seg::Dummy>* mHdr, uint_least8_t const value)
        {
            ++ mHdr->hdr.a.iter;
            mHdr->hdr.a.fmt = value; // a.k.a Impl::Fmt::value
            mHdr->hdr.a.pushes = 0U;
            return;
        }

    template <typename FMT>
    __global__ void reduce_init(const detail::SegImpl<FMT>* info, uint32_t* total,
                                uint32_t size, uint32_t* riter /*= nullptr*/)
    {
        // STRIDE not needed, rbuf number is small
        uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
        __shared__ uint32_t total_sum;
        if(x<size) 
        {
            uint32_t offset = info->rbufBeg();
            MsgHeader<seg::Dummy> *rbuf = mem_as<MsgHeader<seg::Dummy>*>(info->ptrBuf(x+offset));
            if(riter!=nullptr) riter[x]=0;
            if( rbuf->hdr.a.fmt != seg::Illegal::value )
            { 
                atomicAdd(&total_sum, 1);
                __syncthreads();
                if(riter!=nullptr) riter[x] = rbuf->hdr.a.iter;
            }
        }
        __syncthreads();
        if(x==0) total[0] = total_sum;
    }

    template <typename FMT>
    __global__ void reduce_impl(const detail::SegImpl<FMT>* info, uint32_t size,
                                uint32_t* riter /*=nullptr*/) 
    {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
      //  if(x==0) printf("SegImpl<FMT>::segimpl_reduce rlist.size()=%u\n", size[0]);
        if(x<size) {
            uint32_t offset = info->rbufBeg();
            MsgHeader<seg::Dummy> *rbuf = mem_as<MsgHeader<seg::Dummy>*>(info->ptrBuf(x+offset));
            if( rbuf->hdr.a.fmt != seg::Illegal::value ) {
                MsgHeader<seg::Dummy>* rbuf = mem_as<MsgHeader<seg::Dummy>*>(info->ptrBuf(x));
                if(riter != nullptr) {
                    // make a small attempt to retransmit ones that might
                    // have changed while we were reducing.
                    if( rbuf->hdr.a.iter == riter[x] )         // possibly torn
                        rbuf->hdr.a.fmt = seg::Illegal::value;
                }else{
                // just mark ALL potential buffers as illegal (reproducible for tests)
                //    printf(" rlist rBuf# %u  marked as seg::Illegal\n", rlist[x]);
                    rbuf->hdr.a.fmt = seg::Illegal::value;
                }
            }
        }
    }
}//dStorm::
#endif //__CUDACC__
#endif // DSTORM_KERNEL_CUH
