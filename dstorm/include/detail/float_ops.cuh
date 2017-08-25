/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
/** 
 * @file
 * float_ops.cuh
 *
 * @brief Basic float operations
 */

#pragma once

#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

namespace favg {

    template<typename T> 
    __device__ void Memcpy (T* __restrict__ dest, T* const src, uint32_t cnt) {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int STRIDE = gridDim.x * blockDim.x;
        while(x<cnt)
        {
            dest[x] = src[x];
            x+=STRIDE;
        }
    }
    template<typename T>
    __device__  void set_restrict2( T* __restrict__ dest, uint32_t cnt,
                                    const T* __restrict__ s1, const T* __restrict__ s2 ){
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int STRIDE = gridDim.x * blockDim.x;
        while(x < cnt)
        {
            T output = (T)(0.5 * (s1[x] + s2[x]));
            dest[x] = output;
            x+=STRIDE;
        }
    }

    template<typename T>
    __device__  void set_restrict3( T* __restrict__ dest, uint32_t cnt,
                                    const T* __restrict__ s1, const T* __restrict__ s2,
                                    const T* __restrict__ s3){
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int STRIDE = gridDim.x * blockDim.x;
        while(x < cnt)
        {
            dest[x] = 0.33333333333333333 * (s1[x] + s2[x] + s3[x]);
            x+=STRIDE;
        }
    }

    template<typename T>
    __device__  void set_restrict4( T* __restrict__ dest, uint32_t cnt,
                                    const T* __restrict__ s1, const T* __restrict__ s2,
                                    const T* __restrict__ s3, const T* __restrict__ s4){
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int STRIDE = gridDim.x * blockDim.x;
        while(x < cnt)
        {
            dest[x] = 0.25 * (s1[x] + s2[x] + s3[x] + s4[x]);
            x+=STRIDE;
        }
    }

    template<typename T>
    __device__  void set_restrict5( T* __restrict__ dest, uint32_t cnt,
                                    const T* __restrict__ s1, const T* __restrict__ s2,
                                    const T* __restrict__ s3, const T* __restrict__ s4,
                                    const T* __restrict__ s5){
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int STRIDE = gridDim.x * blockDim.x;
        while(x < cnt)
        {
            dest[x] = 0.2 * (s1[x] + s2[x] + s3[x] + s4[x] + s5[x]);
            x+=STRIDE;
        }
    }

    template<typename T>
    __device__  void set_restrict6( T* __restrict__ dest, uint32_t cnt,
                                    const T* __restrict__ s1, const T* __restrict__ s2,
                                    const T* __restrict__ s3, const T* __restrict__ s4,
                                    const T* __restrict__ s5, const T* __restrict__ s6){
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int STRIDE = gridDim.x * blockDim.x;
        while(x < cnt)
        {
            dest[x] = 0.16666666666666667 * (s1[x] + s2[x] + s3[x] + s4[x] + s5[x] + s6[x]);
            x+=STRIDE;
        }
    }

    template<typename T>
    __device__  void set_restrict1( T* __restrict__ dest, uint32_t cnt,
                                    const T* __restrict__ *s, uint32_t const ssz){
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int STRIDE = gridDim.x * blockDim.x;
        T const sszinv = 1.0 / ssz;
        while(x < cnt)
        {
            T sum = T(0);
            #pragma once
            for(uint32_t i=0U; i<ssz; ++i) sum += s[i][x];
            dest[x] = sum * sszinv;
            x+=STRIDE;
        }
    }

    template<typename T, typename S> 
    __device__ void set_restrict( T* __restrict__ dest, uint32_t cnt,
                                    uint32_t size,
                                    S* s){
        T const sszinv = 1.0 / size;
        for(uint32_t i=0U; i<cnt; i++)
        {
            T sum = T(0);
            for(uint32_t j=0U; j<size; j++) sum += s[j].data[i];
            dest[i] = sum * sszinv;
        }
    }
    //TODO: add cudaStream_t * streams
    template<typename T> 
    __device__ inline void set( T* __restrict__ dest, uint32_t cnt,
                                                uint32_t const ssz, const T* __restrict__ * s ){
       /* switch(ssz){
          case(0): break;
          case(1): checkCudaErrors(cudaMemcpy( dest, *s, cnt*sizeof(T), cudaMemcpyDeviceToDevice )); break;
          case(2): favg::set_restrict2( dest, cnt, s[0], s[1] ); break;
          case(3): favg::set_restrict3( dest, cnt, s[0], s[1], s[2] ); break;
          case(4): favg::set_restrict4( dest, cnt, s[0], s[1], s[2], s[3] ); break;
          case(5): favg::set_restrict5 ( dest, cnt, s[0], s[1], s[2], s[3], s[4] ); break;
          case(6): favg::set_restrict6 ( dest, cnt, s[0], s[1], s[2], s[3], s[4], s[5] ); break;
          default: favg::set_restrict1 ( dest, cnt, s, ssz);
        }*/
          if(ssz==0)        return;
          else if (ssz==1)  {favg::Memcpy( (T*)dest, (T*)*s, cnt); return;}
          else if (ssz==2)  {favg::set_restrict2( dest, cnt, s[0], s[1] ); return;}
          else if (ssz==3)  {favg::set_restrict3( dest, cnt, s[0], s[1], s[2] ); return;}
          else if (ssz==4)  {favg::set_restrict4( dest, cnt, s[0], s[1], s[2], s[3] ); return;}
          else if (ssz==5)  {favg::set_restrict5 ( dest, cnt, s[0], s[1], s[2], s[3], s[4] ); return;}
          else if (ssz==6)  {favg::set_restrict6 ( dest, cnt, s[0], s[1], s[2], s[3], s[4], s[5] ); return;}
          else              {favg::set_restrict1 ( dest, cnt, s, ssz); return;}
    }

     template<typename T> __device__ inline void set( T* __restrict__ dest, uint32_t cnt,
                                           uint32_t size,
                                           T const* const s ){
        favg::set( dest, cnt, size, &s[0] );
    }


    template<typename T, typename S> 
    __device__ inline void set( T* __restrict__ dest, uint32_t cnt, uint32_t size, S* __restrict__ s ){
          if(size==0)       return;
          else if(size==1)  {favg::Memcpy( (T*) dest, (T*) s[0].data, cnt ); return;}
          else if(size==2)  {favg::set_restrict2 ( dest, cnt, s[0].data, s[1].data ); return;}
          else if(size==3)  {favg::set_restrict3 ( dest, cnt, s[0].data, s[1].data, s[2].data ); return;}
          else if(size==4)  {favg::set_restrict4 ( dest, cnt, s[0].data, s[1].data, s[2].data, s[3].data ); return;}
          else if(size==5)  {favg::set_restrict5 ( dest, cnt, s[0].data, s[1].data, s[2].data, s[3].data, s[4].data ); return;}
          else if(size==6)  {favg::set_restrict6 ( dest, cnt, s[0].data, s[1].data, s[2].data, s[3].data, s[4].data, s[5].data ); return;}
          else              {favg::set_restrict  ( dest, cnt, size, s); return;}
    }

    template<typename T>
    __device__  void upd_restrict1( T* __restrict__ dest, uint32_t cnt,
                                    const T* __restrict__ s1){
        const unsigned int STRIDE = gridDim.x * blockDim.x;
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        while(x < cnt)
        {
            T output  = (T)(0.5 * (dest[x] + s1[x]));
            dest[x] = output;
            x+=STRIDE;
        }
    }


    template<typename T>
    __device__  void upd_restrict2( T* __restrict__ dest, uint32_t cnt,
                                    const T* __restrict__ s1, const T* __restrict__ s2){
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int STRIDE = gridDim.x * blockDim.x;
        while(x < cnt)
        {
            T output = (T)( 0.33333333333333333 * (dest[x] + s1[x] + s2[x]));
            dest[x] = output;
            x+=STRIDE;
        }
    }

    template<typename T>
    __device__  void upd_restrict3( T* __restrict__ dest, uint32_t cnt,
                                    const T* __restrict__ s1, const T* __restrict__ s2,
                                    const T* __restrict__ s3){
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int STRIDE = gridDim.x * blockDim.x;
        while(x < cnt)
        {
            T output = (T)(0.25 * (dest[x] + s1[x] + s2[x] + s3[x]));
            dest[x] = output;
            x+=STRIDE;
        }
    }

    template<typename T>
    __device__  void upd_restrict4( T* __restrict__ dest, uint32_t cnt,
                                    const T* __restrict__ s1, const T* __restrict__ s2,
                                    const T* __restrict__ s3, const T* __restrict__ s4){
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int STRIDE = gridDim.x * blockDim.x;
        while(x < cnt)
        {
            T output = (T)(0.2 * (dest[x] + s1[x] + s2[x] + s3[x] + s4[x]));
            dest[x] = output;
            x+=STRIDE;
        }
    }

    template<typename T>
    __device__  void upd_restrict5( T* __restrict__ dest, uint32_t cnt,
                                    const T* __restrict__ s1, const T* __restrict__ s2,
                                    const T* __restrict__ s3, const T* __restrict__ s4,
                                    const T* __restrict__ s5){
        const unsigned int STRIDE = gridDim.x * blockDim.x;
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        while(x < cnt)
        {
            dest[x] = 0.16666666666666667 * (dest[x] + s1[x] + s2[x] + s3[x] + s4[x] + s5[x]);
            x += STRIDE;
        }
    }

    template<typename T>
    __device__  void upd_restrict6( T* __restrict__ dest, uint32_t cnt,
                                    const T* __restrict__ s1, const T* __restrict__ s2,
                                    const T* __restrict__ s3, const T* __restrict__ s4,
                                    const T* __restrict__ s5, const T* __restrict__ s6){
        const unsigned int STRIDE = gridDim.x * blockDim.x;
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        while(x < cnt)
        {
            dest[x] = 0.142857142857142857 * (dest[x] + s1[x] + s2[x] + s3[x] + s4[x] + s5[x] + s6[x]);
            x += STRIDE;
        }
    }

    template<typename T>
    __device__  void upd_restrict0( T* __restrict__ dest, uint32_t cnt,
                                    const T* __restrict__ *s, uint32_t const ssz){
        const unsigned int STRIDE = gridDim.x * blockDim.x;
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        T const sszinv = 1.0 / ssz;
        while(x < cnt)
        {
            T sum = *dest;
            #pragma once
            for(uint32_t i=0U; i<ssz; ++i) sum += s[i][x];
            dest[x] = sum * sszinv;
            x += STRIDE;
        }
    }

    template<typename T, typename S>
    __device__ void upd_restrict( T* __restrict__ dest, uint32_t cnt,
                                    S* s, uint32_t const ssz){
        const unsigned int STRIDE = gridDim.x * blockDim.x;
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        T const sszinv = 1.0 / (ssz+1U);
        //for(uint32_t i=0U; i<cnt; i++)
        while(x<cnt)
        {
            T sum = dest[x];
            for(uint32_t j=0U; j<ssz; ++j) sum += s[j].data[x];
            dest[x] = sum * sszinv;
            x += STRIDE;
        }
    }

    template<typename T> 
    __device__ inline void upd( T* __restrict__ dest, uint32_t cnt,
                     uint32_t const ssz, T const* __restrict__ * s ){
        if(ssz==0)      return;
        else if(ssz==1) {favg::upd_restrict1 ( dest, cnt, s[0] ); return;}
        else if(ssz==2) {favg::upd_restrict2 ( dest, cnt, s[0], s[1] ); return;}
        else if(ssz==3) {favg::upd_restrict3 ( dest, cnt, s[0], s[1], s[2] ); return;}
        else if(ssz==4) {favg::upd_restrict4 ( dest, cnt, s[0], s[1], s[2], s[3] ); return;}
        else if(ssz==5) {favg::upd_restrict5 ( dest, cnt, s[0], s[1], s[2], s[3], s[4] ); return;}
        else if(ssz==6) {favg::upd_restrict6 ( dest, cnt, s[0], s[1], s[2], s[3], s[4], s[5] ); return;}
        else            {favg::upd_restrict0 ( dest, cnt, s, ssz); return;}
    }   

    template<typename T> __device__ inline void upd( T* __restrict__ dest, uint32_t const cnt,
                                                     uint32_t ssz, T const* const& s ){
        favg::upd( dest, cnt, ssz, &s[0] );
    }

    template<typename T, typename S> 
    __device__ inline void upd( T* __restrict__ dest, uint32_t cnt,
                     uint32_t const ssz, S* __restrict__ s, int i=0 ){
          if(ssz==0)      return;
          else if(ssz==1) { favg::upd_restrict1( dest, cnt, s[i].data ); return;}
          else if(ssz==2) {favg::upd_restrict2( dest, cnt, s[0].data, s[1].data ); return;}
          else if(ssz==3) {favg::upd_restrict3( dest, cnt, s[0].data, s[1].data, s[2].data ); return;}
          else if(ssz==4) {favg::upd_restrict4( dest, cnt, s[0].data, s[1].data, s[2].data,
                                                      s[3].data ); return;}
          else if(ssz==5) {favg::upd_restrict5( dest, cnt, s[0].data, s[1].data, s[2].data, 
                                                      s[3].data, s[4].data ); return;}
          else if(ssz==6) {favg::upd_restrict6( dest, cnt, s[0].data, s[1].data, s[2].data,
                                                      s[3].data, s[4].data, s[5].data ); return;}
          else            {favg::upd_restrict( dest, cnt, s, ssz); return;}

    }   

    namespace detail {

        template<typename T>
        __device__  void wupd_restrict1( T const dscal, T* __restrict__ dest, uint32_t cnt,
                                        const T* __restrict__ s1){
            const unsigned int STRIDE = gridDim.x * blockDim.x;
            unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
            T const sscal = (1.0 - dscal);
            while(x < cnt)
            {
                dest[x] = dscal * dest[x] + sscal * s1[x];
                x += STRIDE;
            }
        }

        template<typename T>
        __device__  void wupd_restrict2( T const dscal, T* __restrict__ dest, uint32_t cnt,
                                        const T* __restrict__ s1, const T* __restrict__ s2){
            const unsigned int STRIDE = gridDim.x * blockDim.x;
            unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
            T const sscal = (1.0 - dscal) * 0.5;
            while(x < cnt)
            {
                dest[x] = dscal * dest[x] + sscal * ( s1[x] + s2[x] );
                x += STRIDE;
            }
        }

        template<typename T>
        __device__  void wupd_restrict3( T const dscal, T* __restrict__ dest, uint32_t cnt,
                                        const T* __restrict__ s1, const T* __restrict__ s2,
                                        const T* __restrict__ s3){
            const unsigned int STRIDE = gridDim.x * blockDim.x;
            unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
            T const sscal = (1.0 - dscal) * 0.333333333333333333;
            while(x < cnt)
            {
                dest[x] = dscal * dest[x] + sscal * (s1[x] + s2[x] + s3[x]);
                x += STRIDE;
            }
        }

        template<typename T>
        __device__  void wupd_restrict4( T const dscal, T* __restrict__ dest, uint32_t cnt,
                                        const T* __restrict__ s1, const T* __restrict__ s2,
                                        const T* __restrict__ s3, const T* __restrict__ s4){
            const unsigned int STRIDE = gridDim.x * blockDim.x;
            unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
            T const sscal = (1.0 - dscal) * 0.25;
            while(x < cnt)
            {
                dest[x] = dscal * dest[x] + sscal * (s1[x] + s2[x] + s3[x] + s4[x]);
                x += STRIDE;
            }
        }

        template<typename T>
        __device__  void wupd_restrict5( T const dscal, T* __restrict__ dest, uint32_t cnt,
                                        const T* __restrict__ s1, const T* __restrict__ s2,
                                        const T* __restrict__ s3, const T* __restrict__ s4,
                                        const T* __restrict__ s5){
            const unsigned int STRIDE = gridDim.x * blockDim.x;
            unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
            T const sscal = (1.0 - dscal) * 0.2;
            while(x < cnt)
            {
                dest[x] = dscal * dest[x] + sscal * (s1[x] + s2[x] + s3[x] + s4[x] + s5[x]);
                x += STRIDE;
            }
        }

        template<typename T>
        __device__  void wupd_restrict6( T const dscal, T* __restrict__ dest, uint32_t cnt,
                                        const T* __restrict__ s1, const T* __restrict__ s2,
                                        const T* __restrict__ s3, const T* __restrict__ s4,
                                        const T* __restrict__ s5, const T* __restrict__ s6){
            const unsigned int STRIDE = gridDim.x * blockDim.x;
            unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
            T const sscal = (1.0 - dscal) * 0.166666666666666667;
            while(x < cnt)
            {
                dest[x] = dscal * dest[x] + sscal * (s1[x] + s2[x] + s3[x] + s4[x] + s5[x] +s6[x]);
                x += STRIDE;
            }
        }

    }//detail::
}//favg::

namespace fsum {

    template<typename T>
    __device__  void set_restrict2( T* __restrict__ dest, uint32_t cnt,
                                    const T* __restrict__ s1, const T* __restrict__ s2 ){
        const unsigned int STRIDE = gridDim.x * blockDim.x;
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        while(x < cnt)
        {
            dest[x] = (s1[x] + s2[x]);
            x += STRIDE;
        }
    }

    template<typename T>
    __device__  void set_restrict3( T* __restrict__ dest, uint32_t cnt,
                                    const T* __restrict__ s1, const T* __restrict__ s2,
                                    const T* __restrict__ s3){
        const unsigned int STRIDE = gridDim.x * blockDim.x;
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        while(x < cnt)
        {
            dest[x] =  (s1[x] + s2[x] + s3[x]);
            x += STRIDE;
        }
    }

    template<typename T>
    __device__  void set_restrict4( T* __restrict__ dest, uint32_t cnt,
                                    const T* __restrict__ s1, const T* __restrict__ s2,
                                    const T* __restrict__ s3, const T* __restrict__ s4){
        const unsigned int STRIDE = gridDim.x * blockDim.x;
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        while(x < cnt)
        {
            dest[x] = (s1[x] + s2[x] + s3[x] + s4[x]);
            x += STRIDE;
        }
    }

    template<typename T>
    __device__  void set_restrict5( T* __restrict__ dest, uint32_t cnt,
                                    const T* __restrict__ s1, const T* __restrict__ s2,
                                    const T* __restrict__ s3, const T* __restrict__ s4,
                                    const T* __restrict__ s5){
        const unsigned int STRIDE = gridDim.x * blockDim.x;
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        while(x < cnt)
        {
            dest[x] =  (s1[x] + s2[x] + s3[x] + s4[x] + s5[x]);
            x += STRIDE;
        }
    }

    template<typename T>
    __device__  void set_restrict6( T* __restrict__ dest, uint32_t cnt,
                                    const T* __restrict__ s1, const T* __restrict__ s2,
                                    const T* __restrict__ s3, const T* __restrict__ s4,
                                    const T* __restrict__ s5, const T* __restrict__ s6){
        const unsigned int STRIDE = gridDim.x * blockDim.x;
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        while(x < cnt)
        {
            dest[x] =  (s1[x] + s2[x] + s3[x] + s4[x] + s5[x] + s6[x]);
            x += STRIDE;
        }
    }

    template<typename T>
    __device__  void set_restrict1( T* __restrict__ dest, uint32_t cnt,
                                    const T* __restrict__ *s, uint32_t const ssz){
        const unsigned int STRIDE = gridDim.x * blockDim.x;
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        while(x < cnt)
        {
            T sum = T(0);
            #pragma once
            for(uint32_t i=0U; i<ssz; ++i) sum += s[i][x];
            dest[x] = sum;
            x += STRIDE;
        }
    }

    template<typename T, typename S>
    __device__ void set_restrict( T* __restrict__ dest, uint32_t cnt,
                        uint32_t size, S* __restrict__ s){
        const unsigned int STRIDE = gridDim.x * blockDim.x;
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        //for(uint32_t i=0U; i<cnt; i++)
        while(x<cnt)
        {
            T sum = T(0);
            for(uint32_t j=0U; j<size; ++j) sum += s[j].data[x];
            dest[x] = sum;
            x+=STRIDE;
        }
    }

    //TODO: add cudaStream_t * streams
    template<typename T> 
    __device__ inline void set( T* __restrict__ dest, uint32_t cnt,
                                uint32_t const ssz, const T* __restrict__ * s ){
          if(ssz==0)        return;
          else if(ssz==1)   {favg::Memcpy( (T*)dest,(T*) *s, cnt ); return;}
          else if(ssz==2)   {fsum::set_restrict2 ( dest, cnt, s[0], s[1] ); return;}
          else if(ssz==3)   {fsum::set_restrict3 ( dest, cnt, s[0], s[1], s[2] ); return;}
          else if(ssz==4)   {fsum::set_restrict4 ( dest, cnt, s[0], s[1], s[2], s[3] ); return;}
          else if(ssz==5)   {fsum::set_restrict5 ( dest, cnt, s[0], s[1], s[2], s[3], s[4] ); return;}
          else if(ssz==6)   {fsum::set_restrict6 ( dest, cnt, s[0], s[1], s[2], s[3], s[4], s[5] ); return;}
          else              {fsum::set_restrict1 ( dest, cnt, s, ssz); return;}
        
    }

    template<typename T> 
    __device__ void set( T* __restrict__ dest, uint32_t cnt,
                                                    uint32_t size,
                                                    T const* const s ){
        fsum::set( dest, cnt, size, &s[0] );
    }

    template<typename T, typename S> 
    __device__ inline void set( T* __restrict__ dest, uint32_t cnt,
                                                uint32_t const ssz, S* __restrict__ s ){
          if(ssz==0)    return;
          else if(ssz==1)  {favg::Memcpy((T*) dest, (T*)s[0].data, cnt); return;}
          else if(ssz==2)  {fsum::set_restrict2 ( dest, cnt, s[0].data, s[1].data ); return;}
          else if(ssz==3)  {fsum::set_restrict3 ( dest, cnt, s[0].data, s[1].data, s[2].data ); return;}
          else if(ssz==4)  {fsum::set_restrict4 ( dest, cnt, s[0].data, s[1].data, s[2].data, s[3].data ); return;}
          else if(ssz==5)  {fsum::set_restrict5 ( dest, cnt, s[0].data, s[1].data, s[2].data, s[3].data, s[4].data ); return;}
          else if(ssz==6)  {fsum::set_restrict6 ( dest, cnt, s[0].data, s[1].data, s[2].data, s[3].data, s[4].data, s[5].data ); return;}
          else             {fsum::set_restrict( dest, cnt, ssz, s); return;}
    }


    template<typename T>
    __device__  void upd_restrict1( T* __restrict__ dest, uint32_t cnt,
                                    const T* __restrict__ s1){
        const unsigned int STRIDE = gridDim.x * blockDim.x;
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        while(x < cnt)
        {
            dest[x] =  (dest[x] + s1[x]);
            x += STRIDE;
        }
    }
    template<typename T>
    __device__  void upd_restrict2( T* __restrict__ dest, uint32_t cnt,
                                    const T* __restrict__ s1, const T* __restrict__ s2){
        const unsigned int STRIDE = gridDim.x * blockDim.x;
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        while(x < cnt)
        {
            dest[x] =  (dest[x] + s1[x] + s2[x]);
            x += STRIDE;
        }
    }

    template<typename T>
    __device__  void upd_restrict3( T* __restrict__ dest, uint32_t cnt,
                                    const T* __restrict__ s1, const T* __restrict__ s2,
                                    const T* __restrict__ s3){
        const unsigned int STRIDE = gridDim.x * blockDim.x;
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        while(x < cnt)
        {
            dest[x] =  (dest[x] + s1[x] + s2[x] + s3[x]);
            x += STRIDE;
        }
    }

    template<typename T>
    __device__  void upd_restrict4( T* __restrict__ dest, uint32_t cnt,
                                    const T* __restrict__ s1, const T* __restrict__ s2,
                                    const T* __restrict__ s3, const T* __restrict__ s4){
        const unsigned int STRIDE = gridDim.x * blockDim.x;
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        while(x < cnt)
        {
            dest[x] =  (dest[x] + s1[x] + s2[x] + s3[x] + s4[x]);
            x += STRIDE;
        }
    }

    template<typename T>
    __device__  void upd_restrict5( T* __restrict__ dest, uint32_t cnt,
                                    const T* __restrict__ s1, const T* __restrict__ s2,
                                    const T* __restrict__ s3, const T* __restrict__ s4,
                                    const T* __restrict__ s5){
        const unsigned int STRIDE = gridDim.x * blockDim.x;
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        while(x < cnt)
        {
            dest[x] =  (dest[x] + s1[x] + s2[x] + s3[x] + s4[x] + s5[x]);
            x += STRIDE;
        }
    }

    template<typename T>
    __device__  void upd_restrict6( T* __restrict__ dest, uint32_t cnt,
                                    const T* __restrict__ s1, const T* __restrict__ s2,
                                    const T* __restrict__ s3, const T* __restrict__ s4,
                                    const T* __restrict__ s5, const T* __restrict__ s6){
        const unsigned int STRIDE = gridDim.x * blockDim.x;
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        while(x < cnt)
        {
            dest[x] =  (dest[x] + s1[x] + s2[x] + s3[x] + s4[x] + s5[x] + s6[x]);
            x += STRIDE;
        }
    }
} //fsum::
