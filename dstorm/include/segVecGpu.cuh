/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef SEGVECGPU_CUH
#define SEGVECGPU_CUH
/** @file
 * dstorm segment buffer format - Seg_VecGpu implementation header
 */
//#if defined(__CUDACC__)

#include "segVecGpu.hpp"
#include "segInfo.hh"
#include "dstorm_msg.hh"
#include "helper_cuda.h"	// from dua/seample/common/inc/ : CheckCudaErrors
#include "detail/float_ops.cuh"
#include <assert.h>

// ?
//#include "device_functions.hpp" // for integer min/max ...
//        see cuda/targets/x86_64-linux/include/device_functions.hpp
//#include "segInfo.cuh"
//#include "thrust/host_vector.h"
//#include "thrust/device_vector.h"
//#include "thrust/device_ptr.h"
//#include "thrust/pair.h" //TODO: do we have to use thrust here?
//#include <type_traits>
//#include <typeinfo>
//#include "math.h"
//#include <cmath>
//#include "cub/cub.cuh"
//#include "cub/block/block_scan.cuh"
//#include "cub/block/block_load.cuh"
//#include "cub/block/block_store.cuh"
//#define SA_BLOCK 128
//#define CUDA_DBG 0

namespace dStorm {

#if 0
    //typedef cub::BlockScan<size_t, 128> BlockScan;
    //typedef cub::BlockStore<size_t*, 128, 1, cub::BLOCK_STORE_WARP_TRANSPOSE> BlockStore;
    //typedef cub::BlockLoad<size_t*, 128, 1, cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoad;

#define SEG_COUT( STUFF ) do \
    { \
        std::ostringstream oss; \
        oss<< STUFF <<std::endl; \
        this->segPrintf( oss.str().c_str() ); \
    }while(0)
#endif

    namespace user{
        // Implementation of Seg_VecGpu....................................
        template< typename T > inline
            Seg_VecGpu<T>::Seg_VecGpu( SegInfo_t::SegPrintf segPrintf_ /*=nullptr*/)
            : detail::SegBase<Seg_VecGpu<T>, T>( "segVecGpu", segPrintf_)
              , dev_segInfoPod(nullptr) // allocate and copy to GPU during this->add_/delete_segment()
        {
            //TODO: how to make the size of d_rbis adjustable based on the input
            //            checkCudaErrors(cudaMalloc((void**)&d_rbis, sizeof(RbufInfo)*4));
        }

        template< typename T > inline
            Seg_VecGpu<T>::~Seg_VecGpu()
            {  /* checkCudaErrors(cudaFree(d_rbis));*/ }

        /** MUST set SegInfo...::datacode and datasize */
        template< typename T > inline
            void Seg_VecGpu<T>::setSegInfo()
            {
                // dStorm *SegInfo::d has been set up
                assert( SegInfo_t::d != nullptr );
#if !WITH_LIBORM
#error "SegVecGpu kinda depends on liborm"
#endif
                printf(" Seg_VecGpu<T>::setSegInfo!!!\n"); fflush(stdout);
                // and transport MUST be GPU
                //assert( SegInfo_t::d->detail::DstormCommInit::transport == GPU ); // .cuh headers don't allow this

                SegInfo_t& info = *this;
                info.datacode = typeid(typename Base::Tdata).hash_code();
                info.datasize = sizeof(typename Base::Tdata);
            }
        /** finalize new segment, set private data when SegInfo/SegInfoPOD are fully initialized */
        template< typename T > inline
            void Seg_VecGpu<T>::add_segment()
            {
                printf("Seg_VecGpu::add_segment!\n");
                checkCudaErrors(cudaMemset( this->mem, 0, this->segBytes ));   // just do it once.
                printf("Seg_VecGpu::add_segment 2!\n");
                CudaCheckError();
                //SegInfo *sInfo = this->seginfos[ s ];
                //SegInfoPOD const* const sInfoPod = sInfo;
                // 1. cudaMalloc and set dev_segInfoPod
                // 2. copy sInfoPod from host to device
            }
        template< typename T > inline
            void Seg_VecGpu<T>::delete_segment()
            {
                printf("Seg_VecGpu::delete_segment!\n");
                if( this->d->transport == GPU ){
                    //SegInfoPOD const* const sInfoPod = this->...;
                    if( dev_segInfoPod != nullptr ){
                        printf("Seg_VecGpu+Transport<GPU> free the device pointer to SegInfoPOD for segment %lu",
                               (unsigned long)dev_segInfoPod->segNum);
                        // 1. cudaFree and set this->dev_segInfoPod to nullptr
                        dev_segInfoPod = nullptr;
                    }
                }
            }

        template<typename S, typename Fmt, typename T>
            __global__ void  store_wrapper(cu_pair<uint32_t, void*> *ret, MsgHeader<Fmt>* mHdr, T* data, S* iter, uint32_t cnt, uint32_t const offset, void* const buf, double const wgt)
            {
                int const verbose=0;
                const unsigned int STRIDE = gridDim.x * blockDim.x;
                unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
                bool justOnce = (x==0);
                if( justOnce ){
                    ret[0].first = 0U;
                    ret[0].second = (void*)data;
                    if( mHdr->hdr.a.fmt != Fmt::value )
                        printf("Error: store_wrapper: fmt value in consistent\n");
                    mHdr->hdr.u.off = offset;
                    if(wgt == 0.0){
                        mHdr->hdr.u.cnt = 0U;
                        printf("store wgt=0.0 treated as zero-length vector");
                    }else{
                        mHdr->hdr.u.cnt = cnt;
                    }
                    mHdr->hdr.u.wgt = (SCALE_DURING_STORE? 1.0: wgt);
                    mHdr->hdr.u.sz  = sizeof(T);
                    //++mHdr->hdr.a.iter;       // where? count number of 'store' calls
                    //mHdr.a.pushes = 0U;       // where?
                    //    data = mem_as<T*>( buf, sizeof(*mHdr) ); // advance past hdr
                }
                if( wgt == 0.0 ){
                    ;
                }else{
                    //
                    // **ASSUME** data is device memory.  **NO** CudaMemcpy from host!
                    //
                    bool needLoop = ((void const*)(iter+offset) != (void const*)data);
                    if(needLoop){
                        bool needScaling = ( SCALE_DURING_STORE && wgt!=1.0 );
                        if(justOnce && verbose>0){
                            if(! needScaling) printf ("Copying data.\n");
                            else              printf("Copying+Scaling data.\n");
                        }
                        if( needScaling ){
                            while(x<cnt) {
                                data[x] = iter[x+offset] * wgt;     // copy and scale
                                x+=STRIDE;
                            }
                        }else{
                            while(x<cnt) {
                                data[x] = iter[x+offset];           // just copy 
                                x+=STRIDE;
                            }
                        }
                        __syncthreads();
                    }else{
                        bool needScaling = ( SCALE_DURING_STORE && wgt!=1.0 );
                        if(needScaling){
                            while(x<cnt) {
                                data[x] *= wgt;                     // just scale
                                x+=STRIDE;
                            }
                            __syncthreads();
                        }
                    }
                }
                if( justOnce ){
                    ret[0].first  = cnt;
                    ret[0].second = (void*)(data + mHdr->hdr.u.cnt);
                }
                return;
            } //store_wrapper

#if CUDA_DBG
        template<typename Fmt>
            __global__ void print_msg(MsgHeader<Fmt>* mHdr) {
                printf("hdr.a.iter = %u hdr.a.fmt = %d  seg::VecGpu<float>::value = %d\n", mHdr->hdr.a.iter, (int)(mHdr->hdr.a.fmt), (int)(Fmt::value));
            }

        __global__ void print_out(cu_pair<uint32_t, void*> *ret)
        {
            printf("after wrapper:---------ret.first=%u, ret.second@=%p------------\n", ret[0].first, ret[0].second);

        }
#endif

        //STORE
        template< typename T > 
            template< typename S > inline
            cu_pair< uint32_t /*cnt*/, void* /*dataEnd*/ >*
            Seg_VecGpu<T>::store( S * iIter, /* could use general iterator here */
                                  uint32_t cnt, uint32_t const offset,
                                  void* const buf,
                                  // util::Array1D<uint32_t, void> *buf, 
                                  double const wgt/*=1.0*/ )
            {
                uint32_t const nThreads = SA_BLOCK;
                CudaCheckError();
                
                size_t nBlocks = (cnt+nThreads-1U)/nThreads; // divide and round up
                //if (nBlocks > MAX_CU_BLOCKS) 
                //    nBlocks = MAX_CU_BLOCKS;
                assert (nBlocks <= MAX_CU_BLOCKS);
                dim3 grid_construct   (nBlocks, 1, 1);
                dim3 threads_construct(nThreads, 1, 1);
                
                if (CUDA_DBG) 
                CudaCheckError();
                assert( (char*)buf>= (char*)this->mem
                        && (char*)buf< (char*)this->mem + this->segBytes);
                assert( ((char*)buf - (char*)this->mem) % this->bufBytes ==0 );
                assert( this->bufBytes > sizeof(MsgHeader<Fmt>) + sizeof(MsgTrailer) );
                MsgHeader<Fmt>* mHdr = mem_as<MsgHeader<Fmt>*>(buf);
                //                           ^^^ VecGpu, in this case;
#if CUDA_DBG
                int local_rank = atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
                int num_devices = 0;
                checkCudaErrors(cudaGetDeviceCount(&num_devices));
                int local_gpu;// = local_rank % num_devices;
                checkCudaErrors(cudaGetDevice(&local_gpu));
                SEG_COUT("mHdr @ " << mHdr);
                print_msg<<<1,1>>>(mHdr);
                if (CUDA_DBG) 
                CudaCheckError();
#endif
                T* data = mem_as<T*>( buf, sizeof(*mHdr));
                //SEG_COUT(" T* data @ "<<(void*)data << " iIter @ " << (void*)iIter);

                //checkCudaErrors(cudaMalloc((void**)&data, sizeof(T)*cnt));
                // write data
                uint32_t const datasz = /*SegInfo_t*/this->bufBytes
                    - (sizeof(MsgHeader<Fmt>) + sizeof(MsgTrailer));
                if( cnt * sizeof(T) > datasz ){
                    if(0){
                        cnt = datasz / sizeof(T);   // original version silently chopped shorter
                    }else{ // *** NEW ***
                        throw std::length_error("Tried to store too many items in a Dstorm segment");
                    }
                }
                assert(offset==0U);
                struct cu_pair<uint32_t, void*> *d_ret;
                CudaCheckError();
                checkCudaErrors(cudaMalloc((void**)&d_ret, sizeof(cu_pair<uint32_t, void*>)));
                store_wrapper<<<grid_construct, threads_construct>>>(d_ret, mHdr, data, iIter, cnt, offset, buf, wgt);
                if(CUDA_DBG) 
                    CudaCheckError();
                return d_ret;
            }

        template< typename T> 
            __global__ void reduce_wrapper(Seg_VecGpu<T>* d_vec, uint32_t* result)
            {
                unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
                const unsigned int STRIDE = gridDim.x * blockDim.x;
                SegPolicy subviewPolicy = d_vec->policy & RBUF_SUBVIEW_MASK;
                //TODO: a variable size
#define RBUFNUMS_SZ 64 
                if( d_vec->rbufEnd() - d_vec->rbufBeg() > RBUFNUMS_SZ ){
                    printf(" Ohoh. We are in big trouble. RBUFNUMS_SZ is too small");
                    __threadfence();
                    asm("trap;");
                }
                uint32_t rbufnums[RBUFNUMS_SZ];
                if(x==0) result[0] = 0U;
                {
                    int add = 0;
                    for(uint32_t i=d_vec->rbufBeg(); i<d_vec->rbufEnd(); i++){
                        uint32_t const bufnum = i;
                        auto rHdr = mem_as<MsgHeader<seg::VecGpu<T>> const*> (d_vec->ptrBuf(bufnum));
                        if( rHdr->hdr.a.fmt == seg::Illegal::value ) {
                            printf("Illegal rbuf fmt: rbuf %u\n", i);
                            continue;
                        }else if( rHdr->hdr.a.fmt != Seg_VecGpu<T>::Fmt::value ){
                            printf("Consider supporting reducing input fmt %u into VecGpu fmt: %s\n", rHdr->hdr.a.fmt, seg::VecGpu<T>::name);
                            continue;
                        }else if( subviewPolicy == RBUF_SUBVIEW_NONE
                                  && !(rHdr->hdr.u.off==0U && rHdr->hdr.u.cnt == d_vec->SegInfo_t::cnt) ){
                            printf("RBUF_SUBVIEW_NONE exception (rbuf ignored)");
                            continue;
                        }else if( rHdr->hdr.u.off == -1U || rHdr->hdr.u.off == -2U ){
                            printf(" illegal hdr.u.off value in SegVecDense header (rbuf ignored)");
                            continue;
                        }else if( rHdr->hdr.u.cnt + rHdr->hdr.u.off > d_vec->cnt ){
                            printf(" illegal off/cnt (rbuf ignored)");
                            continue;
                        }else if( rHdr->hdr.u.sz != d_vec->datasize/*sizeof(T)*/ ){
                            printf(" illegal data size (rbuf ignored)");
                            continue;
                        }
                        else{                                      // data stored in dense format.
                            rbufnums[add++] = bufnum;
                            if(x==0) result[0] += 1U;
                        }
                    }
                }
                __syncthreads();
                // 2. Handle trivial no-output case
                if( result[0] == 0U ){
                    return;  
                }

                if (result[0] == 1U )//&&  flags[x] == true)
                {
                    auto rHdr = mem_as<MsgHeader<seg::VecGpu<T>> const*> (d_vec->ptrBuf(rbufnums[0]));
                    if(rHdr->hdr.u.cnt == 0U) 
                    {
                        if(x==0) result[0] = 0U;
                        __syncthreads();
                        return;  // zero incoming reduce buffers were averaged
                    }
                }
                // 3. Examine headers, detect homogenous offset + cnt case
                //     Extended to detect other "nice" cases (disjoint also easy)
                uint32_t beg_min;
                uint32_t beg_max;
                uint32_t end_min;
                uint32_t end_max;
                auto rHdr = mem_as<MsgHeader<seg::VecGpu<T>> const*> (d_vec->ptrBuf(rbufnums[0]));
                beg_min = rHdr->hdr.u.off;
                beg_max = beg_min;
                end_min = beg_min + rHdr->hdr.u.cnt;
                end_max = end_min;
                if(result[0]>1)
                {
                    for(int i=0; i<result[0]; i++){
                        auto rHdr = mem_as<MsgHeader<seg::VecGpu<T>> const*> (d_vec->ptrBuf(rbufnums[i]));
                        uint32_t const beg = rHdr->hdr.u.off;
                        uint32_t const end = beg + rHdr->hdr.u.cnt;
                        if( beg < beg_min )      beg_min = beg;
                        else if( beg > beg_max ) beg_max = beg;
                        if( end < end_min )      end_min = end;
                        else if( end > end_max ) end_max = end; 
                    }
                }

                uint32_t n = end_max - beg_min;
                bool     homogenous = (beg_min == beg_max && end_min == end_max);
                // 3b. Reconcile rBuf subview status with requested and implemented behaviors
                if(x==0 && n > d_vec->SegInfo_t::cnt ){ 
                    // won't fit into one output buffer (should never occur)
                    printf("n=%u = end_max:%u - beg_min%u > cnt:%u --reduction won't fit into output buffer\n", n, end_max, beg_min, d_vec->SegInfo_t::cnt);
                    __threadfence();
                    asm("trap;");
                }
                if( subviewPolicy == RBUF_SUBVIEW_NONE ){
                    //if( !(temp[dim/2] == 0U && temp[0+total] == d_vec->cnt ))
                    if(x==0 && !(beg_max == 0U && end_min == d_vec->cnt)) {
                        printf("SegInfo_t Policy Error/Warning: segment policy requires full-sized buffers: begin_max=%u, end_min= %u, but SegInfo_t::cnt is %u\n", beg_max, end_min, d_vec->cnt);
                        //if(x==0) printf("segment policy requires full-sized buffers: begin_max=%u, end_min= %u, but SegInfo_t::cnt is %u\n", temp[dim/2], temp[0+total], d_vec->cnt);
                        result[0] = 0U;
                        return;
                    }
                }else if( subviewPolicy == RBUF_SUBVIEW_HOMOG ){
                    // Seg_VecGpu lacks code to handle any inhomogeneous rBuf subviews.
                    if( x==0 && ! homogenous ){
                        printf("SegInfo_t Policy Error/Warning: reduction does not support inhomogenous rBuf subviews yet\n");
                        result[0] = 0U;
                        __syncthreads();
                        return;
                    }
                }else if( subviewPolicy == RBUF_SUBVIEW_HOMOG_OR_NONOVLP && !homogenous ){
                    bool ovlp = false;
                    if(x<result[0]) {
                        auto rHdr = mem_as<MsgHeader<seg::VecGpu<T>> const*> (d_vec->ptrBuf(rbufnums[x]));
                        uint32_t const ibeg = rHdr->hdr.u.off;
                        uint32_t const iend = ibeg + rHdr->hdr.u.cnt;
                        for(uint32_t j=x+1U; j<result[0]; ++j ){
                            rHdr = mem_as<MsgHeader<seg::VecGpu<T>> const*> (d_vec->ptrBuf(rbufnums[j]));
                            uint32_t const jbeg = rHdr->hdr.u.off;
                            uint32_t const jend = jbeg + rHdr->hdr.u.cnt;
                            if( jbeg < iend && ibeg < jend ){ // OVLP detected
                                printf("OVLP detected\n");
                                ovlp = true;
                                break;
                            }
                        }
                    } // ovlp'ing vector ranges have been detected (or not)
                    __syncthreads();
                    if( ovlp == true ){
                        if(x==0){
                            printf("reduction RBUF_SUBVIEW_HOMOG_OR_NONOVLP detected an overlap\n");
                            result[0] = 0U;
                        }
                        __syncthreads();
                        return;
                    }
                }else if( subviewPolicy == RBUF_SUBVIEW_OVLP_RELAXED ){
                    ; // if inhomogenous,  ** pretend ** no overlaps, and use the no-overlap loop anyway.
                }else if( subviewPolicy == RBUF_SUBVIEW_ANY ){
                    if(x==0){
                        printf("reduction does not support inhomogenous rBuf subviews with strict correctness (RBUF_SUBVIEW_ANY)\n");
                        result[0] = 0U;
                    }
                    __syncthreads();
                    return;
                }
                // 4. Handle trivial single-input case as plain copy
                SegPolicy const op = d_vec->policy & REDUCE_OP_MASK;
                if( !(op == REDUCE_SUM_RBUF || op == REDUCE_AVG_RBUF || op == REDUCE_AVG_RBUF_OBUF) ){
                    //throw std::runtime_error("SegVecDense unimplemented reduction operation");
                    if(x==0) printf("SegVecGpu unimplemented reduction operation\n");
                    // __threadfence();
                    asm("trap;");
                }
                T* redData = nullptr;
                // if(x==0)
                if( result[0] == 1U ) {
                    if( op == REDUCE_SUM_RBUF || op == REDUCE_AVG_RBUF ){
                        // 4a. Set up 'iBuf' output buffer as sub-view matching the rBuf areas
                        //     (full view would need to zero the extra elements, inefficient)
                        void* red = d_vec->ptrBuf( d_vec->ibuf );     // reduce into this buffer
                        MsgHeader<seg::VecGpu<T>>* redHdr = mem_as<MsgHeader<seg::VecGpu<T>>*>( red );
                        auto rHdr = mem_as<MsgHeader<seg::VecGpu<T>> const*> (d_vec->ptrBuf(rbufnums[0]));
                        T const* rData = mem_as<T const*>(d_vec->ptrBuf(rbufnums[0]), sizeof(MsgHeader<seg::VecGpu<T>>));
                        redHdr->hdr.u.off = rHdr->hdr.u.off;
                        redHdr->hdr.u.cnt = rHdr->hdr.u.cnt;
                        redHdr->hdr.u.sz  = sizeof(T);
                        redData = mem_as<T*>(redHdr, sizeof(*redHdr));
                        //memcpy( (void*)redData, (void const*)rbis[0].data, redHdr->hdr.u.cnt * redHdr->hdr.u.sz );
                        //for(int i=0; i<redHdr->hdr.u.cnt; i++)
                        unsigned int xx=x;
                        while(xx<redHdr->hdr.u.cnt) {
                            redData[xx] = rData[xx];
                            xx+=STRIDE;
                        }
                        __syncthreads();
                    }else{ 
                        // 4b. Set up averaging with sub-view of full vector in oBuf
                        void* red    = d_vec->ptrBuf( d_vec->obuf );     // "oBuf += rBuf"
                        MsgHeader<seg::VecGpu<T>>* redHdr = mem_as<MsgHeader<seg::VecGpu<T>>*>( red );
                        auto rHdr = mem_as<MsgHeader<seg::VecGpu<T>> const*> (d_vec->ptrBuf(rbufnums[0]));
                        //assert( redHdr->hdr.u.off == 0U );
                        redHdr->hdr.u.cnt = d_vec->cnt;
                        redHdr->hdr.u.sz  = sizeof(T);
                        redData = mem_as<T*>(redHdr,sizeof(*redHdr));
                        redData += rHdr->hdr.u.off;
                        __syncthreads();
                        T const* rData = mem_as<T const*>(d_vec->ptrBuf(rbufnums[0]), sizeof(MsgHeader<seg::VecGpu<T>>));
                        favg::upd_restrict1( redData, n, rData ); // "redData = 0.5 * (redData + rbiData)"
                    }
                    result[0] = 1U;
                    __syncthreads();
                    return;
                }// end results[0]==1 subcase.
                // 5. Set up output segment for rBuf items spanning range n=end_max-beg_min
                if( op == REDUCE_SUM_RBUF || op == REDUCE_AVG_RBUF ){
                    void* red = d_vec->ptrBuf( d_vec->ibuf );     // reduce into this buffer
                    MsgHeader<seg::VecGpu<T>>* redHdr = mem_as<MsgHeader<seg::VecGpu<T>>*>( red );
                    redHdr->hdr.u.off = beg_min;
                    //redHdr->hdr.u.off = temp[0];
                    redHdr->hdr.u.cnt = n;
                    redHdr->hdr.u.sz  = sizeof(T);
                    redData = mem_as<T*>(redHdr,sizeof(*redHdr));
                }else{
                    if(x==0) 
                        if( op != REDUCE_AVG_RBUF_OBUF )
                        {printf("Error: op != REDUCE_AVG_RBUF_OBUF\n"); asm("trap;");}
                    __syncthreads();
                    void* red = d_vec->ptrBuf( d_vec->obuf );
                    MsgHeader<seg::VecGpu<T>>* redHdr = mem_as<MsgHeader<seg::VecGpu<T>>*>( red );
                    redData = mem_as<T*>(redHdr, sizeof(*redHdr));
                }
                // 6. Perform averaging
                if(x==0) { if( result[0] <= 1U ) {printf("Error: rbis size <= 1U\n"); asm("trap;");} }
                __syncthreads();
                if( homogenous ){
                    // 6a. homogenous span for all rbufs
                    // We now streamline with nice call to RDMA-friendly floating point ops (float_ops.hpp)
                    if( op == REDUCE_AVG_RBUF ){
                        if(result[0]==0) {}
                        else if(result[0]==1) {
                            T const* rData = mem_as<T const*>(d_vec->ptrBuf(rbufnums[0]), sizeof(MsgHeader<seg::VecGpu<T>>));
                            //favg::Memcpy(redData, rData, n); 
                            if(x<n) redData[x] = rData[x];
                        }else{
                            if(x<n){
                                T const sszinv = 1.0 / result[0];
                                T sum = T(0);
                                for(int i=0; i<result[0]; i++){
                                    T const* rData = mem_as<T const*>(d_vec->ptrBuf(rbufnums[i]), sizeof(MsgHeader<seg::VecGpu<T>>));
                                    sum += rData[x];
                                }
                                redData[x] = sum * sszinv;
                            }
                        }
                        //favg::set( redData, n, total, d_vec->d_rbis );
                    }else if( op == REDUCE_SUM_RBUF ){
                        if(result[0]==0) {}
                        else if(result[0]==1) {
                            T const* rData = mem_as<T const*>(d_vec->ptrBuf(rbufnums[0]), sizeof(MsgHeader<seg::VecGpu<T>>));
                            //favg::Memcpy(redData, rData, n); 
                            if(x<n) redData[x] = rData[x];
                        }else{
                            if(x<n){
                                T sum = T(0);
                                for(int i=0; i<result[0]; i++){
                                    T const* rData = mem_as<T const*>(d_vec->ptrBuf(rbufnums[i]), sizeof(MsgHeader<seg::VecGpu<T>>));
                                    sum += rData[x];
                                }
                                redData[x] = sum;
                            }
                        }
                        //fsum::set( redData, n, total, d_vec->d_rbis ); // rbis[i].data[0..n-1] are unaliased T const*
                    }else{
                        if(x==0) {
                            if( op != REDUCE_AVG_RBUF_OBUF ){
                                printf("Error: op != REDUCE_AVG_RBUF_OBUF\n");
                                asm("trap;");
                            }
                        }
                        __syncthreads();
                        //redData += temp[0];      // NOW bump obuf (full vector) up to homogenous offset value
                        redData += beg_min;
                        if(result[0]==0) {}
                        else{
                            if(x<n){
                                T const sszinv = 1.0 / (result[0] + 1U);
                                T sum = redData[x];
                                for(int i=0; i<result[0]; i++){
                                    T const* rData = mem_as<T const*>(d_vec->ptrBuf(rbufnums[i]), sizeof(MsgHeader<seg::VecGpu<T>>));
                                    sum += rData[x];
                                }
                                redData[x] = sum * sszinv ;
                            }
                        }
                        //favg::upd( redData, n, total, d_vec->d_rbis );  
                    }
                }
                else{
                    // 6b. nonovlp inhomogenous case -- loop individually over rbis and
                    //                                  accumulate them, assuming no ovlp
                    //     (generic inhomogenous case IS NOT SUPPORTED YET -- error msg, above)
                    //      Generic MIGHT need further pre-analysis to construct a vector
                    //      of breakpoints and next-normalization factors, and then
                    //      break up the loop over these vector offsets.
                    if(x==0) {
                        if(result[0] < 2U) {
                            printf("Error: rbis size < 2U\n"); asm("trap;");
                        }
                        if(nullptr!="inhomogenous + nonovlp SegVecDense::reduce needs to be reviewd and tested before use") asm("trap;");
                        //assert( total >= 2U );
                        //assert(nullptr=="inhomogenous + nonovlp SegVecDense::reduce needs to be reviewd and tested before use");
                    }
                    __syncthreads();
                    if(x < n) 
                        redData[x] = 0;
                    __syncthreads();
                    //memset( redData, 0, sizeof( n * this->datasize ));
                    // TODO: question: REDUCE_AVG_RBUF and REDUCE_SUM_RBUF are the same
                    if( op == REDUCE_AVG_RBUF ){
                        for(int i=0; i<result[0]; i++){
                            auto rHdr = mem_as<MsgHeader<seg::VecGpu<T>> const*> (d_vec->ptrBuf(rbufnums[i]));
                            size_t const off = rHdr->hdr.u.off;
                            size_t       cnt = rHdr->hdr.u.cnt;
                            T const*     src = mem_as<T const*>(d_vec->ptrBuf(rbufnums[i]), sizeof(MsgHeader<seg::VecGpu<T>>));
                            T*           dest = redData + off - beg_min;
                            if(x < cnt)
                                dest[x] = src[x];
                            __syncthreads();
                        }
                    }else if( op == REDUCE_SUM_RBUF ){
                        for( int i=0; i<result[0]; i++ ){
                            auto rHdr = mem_as<MsgHeader<seg::VecGpu<T>> const*> (d_vec->ptrBuf(rbufnums[i]));
                            size_t const off = rHdr->hdr.u.off;
                            size_t       cnt = rHdr->hdr.u.cnt;
                            T const*     src = mem_as<T const*>(d_vec->ptrBuf(rbufnums[i]), sizeof(MsgHeader<seg::VecGpu<T>>));
                            T*           dest = redData + off - beg_min;
                            if(x<cnt)
                                dest[x] = src[x];
                            __syncthreads();
                        }
                    }else{ 
                        if(x==0)
                            if( op != REDUCE_AVG_RBUF_OBUF ){
                                printf("Error: op != REDUCE_AVG_RBUF_OBUF\n");
                                asm("trap;");
                            }
                        __syncthreads();
                        for(int i=0; i<result[0]; i++){
                            auto rHdr = mem_as<MsgHeader<seg::VecGpu<T>> const*> (d_vec->ptrBuf(rbufnums[i]));
                            size_t const off = rHdr->hdr.u.off;
                            size_t const cnt = rHdr->hdr.u.cnt;
                            T*          dest = redData + off;
                            T const*    src  = mem_as<T const*>(d_vec->ptrBuf(rbufnums[i]), sizeof(MsgHeader<seg::VecGpu<T>>));
                            favg::upd_restrict1(dest, cnt, src);
                            //favg::upd( dest, cnt, 1, d_vec->d_rbis, x);
                        }
                    }
                    if(x==0) result[0] = 1U;
                    __syncthreads();
                    return; // inhomog non-ovlp always are using "1" rBuf for sum/avg
                    //            DO NOT return rbis.size() for non-overlapped case !
                }// end 6b. nonovlp inhomogenous case
                // 7. Check trailer for torn reads
                __syncthreads();
                return;
            }// reduce_wrapper

        template< typename T> inline
            uint32_t Seg_VecGpu<T>::reduce(uint32_t* d_size) const
            {
                uint32_t  h_size;
                checkCudaErrors(cudaMemcpy(&h_size, d_size, sizeof(uint32_t), cudaMemcpyDeviceToHost));
                if((this->policy & REDUCE_OP_MASK) == REDUCE_NOP) return h_size;
                assert( (this->policy & SEG_LAYOUT_MASK) == SEG_FULL ); 
                assert( (this->policy & REDUCE_OP_MASK)  != REDUCE_NOP );
                if(h_size == 0U)
                    return 0U;

                uint32_t rbuf_size = sizeof(uint32_t) * (this->rbufEnd() - this->rbufBeg());
                uint32_t total = this->cnt;
                size_t nThreads = SA_BLOCK;
                bool fullBlocks = ( total % nThreads==0);
                size_t nBlocks = (fullBlocks) ? (total/nThreads) :
                    (total/nThreads+1);
                dim3 grid_construct(nBlocks, 1, 1);
                dim3 threads_construct(nThreads, 1, 1);

                Seg_VecGpu<T> *d_vec;
                checkCudaErrors(cudaMalloc((void**)&d_vec, sizeof(Seg_VecGpu<T>)));
                checkCudaErrors(cudaMemcpy((void*)d_vec, (void*)this, sizeof(Seg_VecGpu<T>), cudaMemcpyHostToDevice));

                reduce_wrapper<<<grid_construct, threads_construct, rbuf_size>>>(d_vec, d_size);
                if(CUDA_DBG) CudaCheckError();
                checkCudaErrors(cudaMemcpy(&h_size, d_size, sizeof(uint32_t), cudaMemcpyDeviceToHost));
                if(d_vec) checkCudaErrors(cudaFree(d_vec));
                return h_size;
            } // end of reduce
        //REDUCE
    } // user::
} // dStorm::
//#endif // defined(__CUDA_ARCH__)
#endif // SEGVECGPU_CUH
