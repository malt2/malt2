/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
/** @file
 * Collect all the cuda kernels.
 * These \b must be compiled anyway, so it doesn't make sense to
 * keep them in the header file, except maybe for the template
 * implementations.
 *
 * - Kernels are cuda functions prefixed with __global__
 * - They run on many GPU cores simultaneously
 * - They are called with FOO<<<GRID,THREADS>>>(args) notation
 *   - from code that executes on the CPU
 *   - or (>= compute_35) from other __device__ functions
 * - They can only call __device__ functions (what about inlined non-__device__ fns?)
 */
#include "dstorm_kernel.cuh"
#include "dstorm_msg.hh"	// for the template stuff (which should disappear since only for FMT ~ seg::VecGpu

namespace dStorm {

    __global__ void store_impl(cu_pair<uint32_t, void*>* stop, uint32_t const cnt, MsgHeader<seg::Dummy>* mHdr, void* obuf, unsigned long bufBytes)
    {
        if(stop[0].first != cnt) {
            printf("Error: cu_pair first(%d) is not equal to cnt (%d)...\n", stop[0].first, cnt);
            asm("trap;");
        }
        void* const& dataEnd = stop[0].second;
        mHdr->hdr.a.bytes = mem_as<char*>(dataEnd) - (char*)obuf + sizeof(MsgTrailer);
        if( dataEnd != nullptr ){   /* nData==0 COULD be valid */
            if( (char*)dataEnd >  (char*)obuf
                && (char*)dataEnd <= (char*)obuf + (bufBytes - sizeof(MsgTrailer)) ) {
                MsgTrailer* trailer = mem_as<MsgTrailer*>(dataEnd);
                //TODO: trailer->hdr is nil
                //trailer->hdr.a = mHdr->hdr.a;
            }else{ // throw with some debug message
                printf("SegImpl::store dataEnd out-of-range, programmer error\n");
                asm("trap;");
            }
        }
        return;
    }
    __global__ void push_init(MsgHeader<seg::Dummy>* obufHdr, SegNum const s, uint_least32_t* d_result, uint_least8_t *d_pushes)
    {
        // obufHdr points into the GPU segment memory
        //printf("****Dstorm::push_init(%u), iter = %u, bytes = %u, fmt = %u, pushes= %u\n",s, obufHdr->hdr.a.iter, obufHdr->hdr.a.bytes, (unsigned)obufHdr->hdr.a.fmt, (unsigned)obufHdr->hdr.a.pushes);
        //if(obufHdr->hdr.a.pushes > 0U) printf("duplicate Dstorm::push");
#if 1
        d_pushes[0] = (unsigned)obufHdr->hdr.a.pushes;
        d_result[0] = (uint_least32_t)obufHdr->hdr.a.bytes;
#else
        d_pushes[0] = 0U;
        d_result[0] = 0U;
#endif
    }

    __global__ void push_con(MsgHeader<seg::Dummy>* obufHdr, uint32_t nWrite)
    {
        obufHdr->hdr.a.pushes += (1U + (nWrite>0U?1U:0U) ); // a bool would have been ok
    }
}//dStorm::
