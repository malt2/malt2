/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef SEGIMPL_HH_
#define SEGIMPL_HH_
/** @file
 * SegImpl functions for \ref segInfo.hpp.
 * Boilerplate for CPU, but special-cased for GPU segments. */
#include "segInfo.hh"           // SegBase is derived-from SegInfo

#include "dstorm.hpp"

#include "dstorm_msg.hh"        // need MsgHeader types (as well as seg::FMT metainfo)
#include <cstring>              // memset

#if defined(__CUDACC__)
// for SegImpl template overrides, FMT=seg::VecGpu (Seg_VecGpu)
#include "segVecGpu.cuh"        // need complete type
#include "dstorm_kernel.cuh"    // need cuda kernel declarations
#include "helper_cuda.h"        // from cuda/samples/common/inc/ : CheckCudaErrors
#endif

/** throw on error (so don't need an Orm for printf) */
#define SUCCESS_OR_DIE( STMT, VALUE ) do{ \
    if( (STMT) != (VALUE) )  { \
        throw(std::runtime_error( "(" #STMT ") != (" #VALUE ")")); \
    } \
} while(0)

#define ORM_NEED(X) SUCCESS_OR_DIE(X, true )

namespace dStorm {

    namespace detail {

        template< typename FMT > inline
            void detail::SegImpl<FMT>::segimpl_add_segment( SegNum const s,
                                                            Dstorm* d_,
                                                            IoNet_t const ionet_,
                                                            SegPolicy policy_,
                                                            uint32_t const cnt_ )
            {
                int const verbose=0;
                assert( d_ != nullptr );
#if WITH_LIBORM
                assert( d_->orm != nullptr );
                assert( d_->orm->obj != nullptr );
#endif
                assert( cnt_ > 0U );
                //SegInfo& info = *this;  // cast down to lowest-level base
                assert( /*info.*/this->valid == false );  // add_segment twice is BAD


                // Here is **all** the stuff that depends on FMT
                // ... Could this be done in Dstorm::add_segment(..)  (dstorm_any2.hh) ?
                // GPU code doesn't seem to set these up! oversight?
                //     Who should really be responsible for these logically const SegInfo values?
                this->sizeofMsgHeader = sizeof( MsgHeader<FMT> );
                this->fmtValue = FMT::value; // where?
                //
                // XXX optimization code below is INDEPENDENT of FMT, so punt to a non-templated private helper XXX
                //

                // 1. set up a few things in SegInfo base
                //    NOTE: this describes segment on THIS node.
                //          Other nodes may differ (ex. nbuf corresponds
                //          to ionet graph in-degree, which may vary.
                {
                    // info.d=d not allowed (C++ protected wants "this->")
                    this->d       = d_;
                    //this->segimpl = this;
                    this->ionet   = ionet_; // temporarily?
                    this->segNum  = s;      // logically, the Dstorm segment number
                    this->cnt     = cnt_;   // how many Tdata  (Impl knows Tdata, not us)
                    this->indegree  = this->indegree_max  = d_->netRecvVec(this->ionet).size();
                    this->outdegree = this->outdegree_max = d_->netSendVec(this->ionet).size();
                    {
                        SegPolicy const layout = policy_ & SEG_LAYOUT_MASK;
                        this->obuf    = SegInfo::oBufs[layout];
                        this->ibuf    = SegInfo::iBufs[layout];
                        this->rbuf    = SegInfo::rBufs[layout];

                        SegPolicy const reduction = policy_ & REDUCE_OP_MASK;
                        switch( layout ){
                          case(SEG_FULL):
                              this->nbuf = this->rbuf + this->indegree_max;
                              // NEW: silently remove the iBuf if we reduce will never require it ...
                              {
                                  if( reduction == REDUCE_AVG_RBUF_OBUF ){
                                      if(d_->get_iProc() == 0U ) ORM_COUT(d_->orm, " ** NOTE: ** add_segment SEG_FULL with REDUCE_AVG_RBUG_OBUF removing the iBuf segment");
                                      --this->ibuf;       // This sets iBuf = oBuf (no allocation for a separate iBuf segment)
                                      --this->rbuf;
                                      --this->nbuf;
                                  }
                                  else if( reduction == REDUCE_NOP ){
                                      if(d_->get_iProc() == 0U ) ORM_COUT(d_->orm, " ** WARNING: ** add_segment SEG_FULL with REDUCE_NOP.  Are you running timing measurements?");
                                  }
                                  else if( reduction == REDUCE_STREAM ){
                                      if( (policy_ & SEGSYNC_MASK) != SEGSYNC_NOTIFY_ACK ){
                                          policy_ = (policy_ && ~SEGSYNC_MASK) | SEGSYNC_NOTIFY_ACK;
                                          if(d_->get_iProc() == 0U ) ORM_COUT(d_->orm, " ** WARNING: ** add_segment SEG_FULL with REDUCE_STREAM forced SEGSYNC_NOTIFY_ACK");
                                      }
                                      assert( (policy_ & SEGSYNC_MASK) == SEGSYNC_NOTIFY_ACK );
                                  }
                              }
                              break;
                          case(SEG_ONE):
                              this->nbuf = 1U;  // only an oBuf (oBuf = iBuf = rBuf = 0)
                              { // only some cases have been used so far...
                                  if( reduction == REDUCE_NOP ){
                                      if( this->indegree_max > 1 )
                                          ORM_COUT(d_->orm, "** WARNINNG ** SEG_ONE|REDUCE_NOP, but ionet indegree > 1");
                                      //break;
                                  }else if( reduction == REDUCE_STREAM ){
                                      if( this->indegree_max > 1 ){
                                          throw std::domain_error("SEG_ONE|REDUCE_STREAM, but ionet indegree > 1");
                                      }
                                      //break;    // poll, call user cb of  setStreamFunc(s,cb)
                                  }else{
                                      throw std::runtime_error("SEG_ONE with untested REDUCE_* policy");
                                  }
                              }
                              break;
                          default:      // milde compile generates a warning. Better behavior is to warn on missing case for switch(enum) !
                              throw std::runtime_error("SEG_LAYOUTS is not a valid layout");
                        }
                    }
                    this->policy  = policy_;
                    this->seg_id = s;       // Gaspi idea of segment number (same as segNum, for now)
                    // Note:orm behaviour is undefined if you attempt to define separate
                    //       segments with the same seg_id (leads to SYSSEGV)

                }

                // 2. client Impl MUST now set 2 values about it's "data" type
                {
                    this->datasize = -1U;
                    this->datacode = -1U; // FORCE Impl to change these
                    // User code MUST (at least) set up these SegInfo parms:
                    //      - datasize (size of client's data type; ex sizeof(float))
                    //      - datacode (paranoia checks ? )
                    this->valid    = true;  // even though SegInfo is not quite perfect yet?
                    this->Impl::setSegInfo( );
                    ORM_NEED( this->datasize != -1U );
                    ORM_NEED( this->datacode != -1U );
                }

                // 3. Then we auto-fill rest of SegInfo Base
                {
                    // user data area size can't be obviously too big
                    unsigned long dataBytes = this->cnt * this->datasize;
                    ORM_NEED( dataBytes == (uint64_t)this->cnt * (uint64_t)this->datasize );
                    dataBytes = DSTORM_ALIGN_UP(dataBytes);
                    // now optimized memcpy (if Impl wants it) need not worry about
                    // segfaults when writing "just a little" past last element.
                    // (They could even write into the MsgTrailer area with no ill effect)

                    // back to finalizing SegInfo values (using user info too)
                    this->bufBytes
                        = this->sizeofMsgHeader         // our fmt is VecDense
                        + dataBytes                     // Impl set this for us
                        + sizeof( MsgTrailer );
                    // round the segBytes up (in principle, not needed)
                    this->bufBytes = DSTORM_ALIGN_UP( this->bufBytes );
                    this->segBytes = this->nbuf * this->bufBytes;
                    //this->valid    = true;
                    if(verbose>1 || (verbose==1 && d_->iProc==0U)){
                        SegPolicyIO policyIO{policy_};
                        ORM_COUT(d_->orm, " segimpl_add_segment: "
                                   <<"\tbufBytes="<<this->bufBytes<<"\tsegBytes="<<this->segBytes
                                   <<"\n\tvalid="<<this->valid<<"\tdatacode="<<std::hex<<this->datacode<<std::dec
                                   <<"\tcnt="<<this->cnt<<"\tdatasize="<<this->datasize
                                   <<"\n\tsizeofMsgHeader="<<this->sizeofMsgHeader
                                   <<"\tseg_id="<<(unsigned)(this->seg_id)<<"\tfmtValue="<<unsigned(this->fmtValue)
                                   <<"\n\tsegment policy "<<dStorm::name(policyIO)<<"\n");
                    }
                }
                unsigned long segBytes_up = (this->segBytes + 4095U )/ 4096U * 4096U;

                // 4. Now create the ORM segment buffers,
                //    - remember ORM handle,
                //    - get base ptr of segment
                //    - and zero-initialize if rank 0 or verbose (try to cut down valgrind msgs)
#if WITH_LIBORM
                assert( d_->orm != nullptr );
                assert( d_->orm->obj != nullptr );
                {
                    std::ostringstream oss;
                    oss<<" about to orm->segment_create, orm @ "<< (void*)d_->orm
                        <<" orm->obj @ "<<(void*)d_->orm->obj<<"\n";
                    ORM_COUT(d_->orm, oss.str());
                }
                NEED(d_->orm->segment_create( d_->orm, this->seg_id,
                                              segBytes_up,
                                              ORM_GROUP_ALL, ORM_BLOCK,
                                              ORM_MEM_INITIALIZED));

                // NEW: map dstorm sync policies into liborm sync policies
                orm_sync_t orm_sync;
                switch (policy_ & SEGSYNC_MASK) {
                  case(SEGSYNC_NOTIFY):     orm_sync = orm_sync_t(ORM_SYNC_NOTIFY);     break;
                  case(SEGSYNC_NOTIFY_ACK): orm_sync = orm_sync_t(ORM_SYNC_NOTIFY_ACK); break;
                  default: {
                               // SHOULD be asking for SEGSYNC_NONE, but it
                               // seems MPI/GPU might not send anything without
                               // some sort of synchronization calls (UGGH)
                               //
                              
                                // ORM_SYNC_BARRIER, for OMPI/GPU, will even
                                // work without the win_post/wait//start/complete
                                // calls, so ... for now ...
                                //
                                // ORM_SYNC_NONE *** transfers nothing *** for OMPI/GPU
                                   orm_sync = orm_sync_t(ORM_SYNC_BARRIER);
                            
                               break;
                           }
                }
                NEED(d_->orm->sync( d_->orm, this->seg_id, orm_sync ));
#ifndef NDEBUG
                { // brief sanity check
                    orm_sync_t retrieved;
                    NEED( d_->orm->getsync( d_->orm, this->seg_id, &retrieved ));
                    assert( retrieved == orm_sync );
                }
#endif
#endif

                // ORM_GROUP_ALL:         Group of ranks with whom segment registers
                // ORM_BLOCK:             Timeout in milliseconds (or ORM_BLOCK/ORM_TEST)
                // ORM_MEM_INITIALIZED:   Memory allocation policy
                //  TODO: With many nodes and an I/O network of bounded degree, each
                //        node may well get its own orm group (save NIC resources).
                //d_->barrier();
                this->mem = nullptr;
#if WITH_LIBORM
                NEED(d_->orm->segment_ptr ( d_->orm, this->seg_id, &(this->mem)));
#endif

                if(verbose>1 || (verbose==1 && d_->iProc==0U))
                    ORM_COUT(d_->orm, " segment_create( seg_id="<<(unsigned)(this->seg_id)
                               <<", segBytes="<<this->segBytes<<" )"<<" local mem@"<<(void*)(this->mem));

#if 1 // PARANOIA --- initialize the **entire** segment memory to zero
                // PROBABLY this should be done in SegVecDense/SegVecGPU::add_segment instead of here.
                // (Perhaps zeroing just the MsgHeader regions would be enough)
                // BUT zero entire segment (paranoid?) might catch any writability errors, though!
                //d_->barrier();
#if ! WITH_GPU // USUAL CASE: this->mem is a non-shared host pointer
                memset( this->mem, 0, this->segBytes );
#else
                if( d_->orm->transport != ORM_GPU ){
                    // this->mem is host memory : zero it here
                    memset( this->mem, 0, this->segBytes );   // just do it once.
                }else{
                    if( d_->get_iProc()==0 ) ORM_COUT(d_->orm, " SegImpl for Transport<GPU> **not** zeroing segBytes [yet]");
                    // leave this for SegVecGpu::add_segment, to avoid requiring cuda headers here
                }
#endif
#endif

                // NEW: invoke SegImpl::add_segment **after** SegInfo fully "valid"
                this->Impl::add_segment();
            }

        template< typename FMT >
            void detail::SegImpl<FMT>::delete_segment()
            {
                static_assert( is_segment_format<FMT>::value,
                               "ERROR: require is_segment_format<FMT>");
                // 1st, give client class a chance to do any cleanup
                //      (maybe it wants to output some statistics?)
                this->Impl::delete_segment();       // Note: independent of FMT

                // 2nd, obliterate SegInfo to invalid, unusable state
                //   This is NOT GOOD now that SegInfo is really non-POD ...
                //memset( static_cast<SegInfo*>(this), 0, sizeof(SegInfo) );
                this->d = nullptr;
                this->valid = false;

            }
        //-------------------- store #2, begin,end version
        template< typename FMT >
            template< typename IN_CONST_ITER >
            inline void
            detail::SegImpl<FMT>::store( IN_CONST_ITER iter, IN_CONST_ITER const end,
                                         uint32_t const offset, double const wgt )
            {
                static_assert( is_segment_format<FMT>::value,
                               "ERROR: require is_segment_format<FMT>");
                static_assert( FMT::value != seg::VecGpu<float>::value, "Should not use default SegImpl<FMT>::store(begin,end,offset,wgt) impl for FMT=seg::VecGpu" );
                // easy impl: determine cnt and shunt to other 'store' routine
                auto cnt = std::distance( iter, end );              // slow if using fwd-iter
                //DST_COUT("iter-store:cnt="<<cnt);

                // adjust cnt downward if it would overflow (user impl might forget to do this)
                if( cnt > this->/*SegInfo*/cnt ) cnt = this->cnt;
                this->store( iter, cnt, offset, wgt );
            }

#if defined(__CUDACC__)
//#warning "using cuda compiler"
#endif
#if !defined(__CUDACC__)
        template< typename FMT >
            template< typename IN_CONST_ITER >
            inline void
            detail::SegImpl<FMT>::store( IN_CONST_ITER iter, uint32_t const cnt,
                                         uint32_t const offset, double const wgt )
            {
                static_assert( is_segment_format<FMT>::value,
                               "ERROR: require is_segment_format<FMT>");
                static_assert( FMT::value != seg::VecGpu<float>::value, "Should not use default SegImpl<FMT>::store impl for FMT=seg::VecGpu" );
                int const verbose=0;
                // We will need access to our Dstorm::orm function dispatch table
                assert( this->d != nullptr );
#if WITH_LIBORM
                assert( this->d->orm != nullptr );
#endif
#if WITH_GPU
                if( this->fmtValue == seg::VecGpu<float>::value ){ // float|double both have same fmtValue
                    throw std::runtime_error("SegImpl<seg::VecGpu<T>>::store should be using a specialized version of store!");
                }
#endif

                // Let's precalculate some common stuff so Impl needs to
                // write less lines of code...

                // buffer targetted by \c store ops
                //void * obuf = this->ptrBuf(this->oBuf()); // from SegBase utility class
                void *obuf = this->ptrObuf();               // from SegInfo base class
                if(verbose) ORM_COUT(this->d->orm, "SegImpl::store(iter,cnt="<<cnt<<",offset="<<offset<<") into obuf @ "<<(void*)obuf);

                // cast to MsgHeader type allowing fiddling with HdrT<Internal>
                MsgHeader<seg::Dummy> * mHdr = mem_as<MsgHeader<seg::Dummy>*>(obuf);

                // this would be WRONG for GPU code !!!
                ++ mHdr->hdr.a.iter;
                mHdr->hdr.a.fmt = FMT::value; // a.k.a Impl::Fmt::value
                mHdr->hdr.a.pushes = 0U;
                if(verbose) ORM_COUT(this->d->orm, " a.fmt = "<<(int)(mHdr->hdr.a.fmt));

                // Now client does not need to set up the internal header.
                // (Client may read it for safety checking)
                // NOTE: we have not checked RDMA queue size to see if old
                //       data has been sent out.  And even if not in queue,
                //       the NIC may still be "working on it".  POSSIBLE safety
                //       here is to monitor the inter-send times on client
                //       and check that spacing and writeSeq are being
                //       received on client "most of the time" collision-free.
                //
                //XX if( this->data(this->oBuf()) == &*iter ){ // SegBase<..>::data(obuf) returning TDATA *
                //XX     // above line does not even compile, if iterator is an arbitrary
                //XX     // user-defined iterator, like "BitmapDeltaIterator"
                //XX     DST_COUT(" nilpotent obuf data store detected!\n");
                //XX     // Client "send" header is not modified by us during reduce.
                //XX     // BUT client data length MIGHT have changed, which means
                //XX     // MsgHeader and Trailer MIGHT be different too.
                //XX     // ... so we cannot "just skip" calling client implementation.
                //XX     // ---> client implementation is responsible for optimizing
                //XX     //      store operations to avoid data memcpy
                //XX }
                std::pair< uint32_t /*nData*/, void* /*dataEnd*/ > const stop
                    = Impl::store( iter, cnt, offset, obuf, wgt );
                assert( stop.first == cnt );
                //uint32_t const& nData = stop.first;
                void* const& dataEnd  = stop.second;

                // finalize HdrT<Internal>,  convenient to save total msg bytes:
                mHdr->hdr.a.bytes = mem_as<char*>(dataEnd) + sizeof(MsgTrailer) - (char*)obuf;

                // If data write OK, write MsgTrailer as copy
                // of Internal portion of MsgHeader.
                if( dataEnd != nullptr ){   /* nData==0 COULD be valid */
                    if( (char*)dataEnd >  (char*)obuf
                        && (char*)dataEnd <= (char*)obuf + this->bufBytes - sizeof(MsgTrailer) ) {
                        MsgTrailer* trailer = mem_as<MsgTrailer*>(dataEnd);
                        trailer->hdr.a = mHdr->hdr.a;
                    }else{ // throw with some orm printout
                        ORM_NEED( nullptr=="SegImpl::store dataEnd out-of-range"
                                    " (programmer error)");
                    }
                }
            }
        template< typename FMT >
            inline uint32_t/*more?*/
            detail::SegImpl<FMT>::segimpl_reduce() const
            {
                static_assert( is_segment_format<FMT>::value,
                               "ERROR: require is_segment_format<FMT>");
                static_assert( FMT::value != seg::VecGpu<float>::value, "Should not use default SegImpl<FMT>::reduce() impl for FMT=seg::VecGpu" );
                int const verbose=0;
                uint32_t nReduced;
                // SegInfo::rbufBeg(), rbufEnd()
                std::vector<uint32_t> rlist;                    // thread-local in class to avoid malloc?
#define CHK_HDR_ITER 1
#if CHK_HDR_ITER
                std::vector<uint_least64_t> riter;
#endif
                MsgHeader<seg::Dummy> *rbuf;
                for( size_t r=this->rbufBeg(); r<this->rbufEnd(); ++r )
                { // iterate over all receive buffers
                    rbuf = mem_as<MsgHeader<seg::Dummy>*>(this->ptrBuf(r));
                    // VALGRIND == Following may have a.fmt unitialized
                    if( rbuf->hdr.a.fmt == seg::Illegal::value )
                    { // we marked this buffer as DONE, skip
                        if(verbose) ORM_COUT(this->d->orm, "rBuf#"<<r<<" skipped seg::Illegal");
                        continue;
                    }
                    rlist.push_back( r );
#if CHK_HDR_ITER
                    riter.push_back( rbuf->hdr.a.iter );
#endif
                }

                nReduced = Impl::reduce( rlist );
                if(verbose) ORM_COUT(this->d->orm, "SegImpl<FMT>::segimpl_reduce rlist.size()="<<rlist.size()<<" nReduced="<<nReduced);

#if CHK_HDR_ITER // this MIGHT not be reproducible for tests
                for( size_t r=0U; r<rlist.size(); ++r){
                    rbuf = mem_as<MsgHeader<seg::Dummy>*>(this->ptrBuf(rlist[r]));
                    // make a small attempt to retransmit ones that might
                    // have changed while we were reducing.
                    if( rbuf->hdr.a.iter == riter[r] ){         // possibly torn
                        rbuf->hdr.a.fmt = seg::Illegal::value;
                    }
                }
#else // just mark ALL potential buffers as illegal (reproducible for tests)
                for( size_t r=0U; r<rlist.size(); ++r){
                    if(verbose) DST_COUT(" rlist rBuf#"<<rlist[r]<<" marked as seg::Illegal");
                    rbuf = mem_as<MsgHeader<seg::Dummy>*>(this->ptrBuf(rlist[r]));
                    rbuf->hdr.a.fmt = seg::Illegal::value;
                }
#undef CHK_HDR_ITER
#endif
                return nReduced;
            }
#endif // NOT CUDACC
    }//detail::

#if defined(__CUDACC__)
//#warning "HAVE VecGpu specializations in segImpl.hh"
    template<>
        inline uint32_t/*more?*/
        detail::SegImpl<dStorm::seg::VecGpu<float>>::segimpl_reduce() const
        {
#define CHK_HDR_ITER 1
            typedef seg::VecGpu<float> FMT;
            static_assert( is_segment_format<FMT>::value,
                    "ERROR: require is_segment_format<FMT>");
            static_assert( FMT::value == seg::VecGpu<float>::value, "This impl is only for VecGpu!");
            uint32_t nReduced;
            size_t size = this->SegInfoPOD::rbufEnd() - this->SegInfoPOD::rbufBeg();
            size_t nThreads = SA_BLOCK;
            bool fullBlocks = (size % nThreads==0);
            size_t nBlocks = (fullBlocks) ? (size/nThreads) :
                (size/nThreads+1);
            dim3 grid_construct(nBlocks, 1, 1);
            dim3 threads_construct(nThreads, 1, 1);
            // TODO:   For safety just transfer data -- transferring the virtual table
            //       is nonsense, since SegImpl<FMT> and SegInfo have virtual functions,
            //       that are illegal to use on GPU.  This works, because we happen to
            //       only use inlined device funcs that are non-virtual.
            //         But I think it is fragile.
            // TODO:   Segment info DATA (and map of seg-->seg info DATA) should be transferred
            //       just once, during add_segment (and erase during delete_segment)
            detail::SegImpl<FMT> * d_rbuf;
            checkCudaErrors(cudaMalloc((void**)&d_rbuf, sizeof(detail::SegImpl<FMT>)));
            checkCudaErrors(cudaMemcpy((void*)d_rbuf, (void*)this, sizeof(detail::SegImpl<FMT>), cudaMemcpyHostToDevice));
            uint32_t* total;
            checkCudaErrors(cudaMalloc((void**)&total, sizeof(uint32_t)));

#if CHK_HDR_ITER
            uint32_t* riter;
            checkCudaErrors(cudaMalloc((void**)&riter, size * sizeof(uint32_t)));
            dStorm::reduce_init<<<grid_construct, threads_construct>>>(d_rbuf, total, size, riter);
            if(CUDA_DBG) CudaCheckError();
#else
            dStorm::reduce_init<<<grid_construct, threads_construct>>>(d_rbuf, total, size);
            if(CUDA_DBG) CudaCheckError();
#endif
            nReduced = Impl::reduce(total);
#if CHK_HDR_ITER
            reduce_impl<<<grid_construct, threads_construct>>>(d_rbuf, size, riter);
            if(CUDA_DBG) CudaCheckError();
#else
            reduce_impl<<<grid_construct, threads_construct>>>(d_rbuf, size);
            if(CUDA_DBG) CudaCheckError();
#endif
#undef CHK_HDR_ITER

            return nReduced;
        }

    template<>
        inline uint32_t/*more?*/
        detail::SegImpl<dStorm::seg::VecGpu<double>>::segimpl_reduce() const
        {
#define CHK_HDR_ITER 1
            typedef seg::VecGpu<float> FMT;
            static_assert( is_segment_format<FMT>::value,
                    "ERROR: require is_segment_format<FMT>");
            static_assert( FMT::value == seg::VecGpu<float>::value, "This impl is only for VecGpu!");
            uint32_t nReduced;
            size_t size = this->SegInfoPOD::rbufEnd() - this->SegInfoPOD::rbufBeg();
            size_t nThreads = SA_BLOCK;
            bool fullBlocks = (size % nThreads==0);
            size_t nBlocks = (fullBlocks) ? (size/nThreads) :
                (size/nThreads+1);
            dim3 grid_construct(nBlocks, 1, 1);
            dim3 threads_construct(nThreads, 1, 1);
            // TODO:   For safety just transfer data -- transferring the virtual table
            //       is nonsense, since SegImpl<FMT> and SegInfo have virtual functions,
            //       that are illegal to use on GPU.  This works, because we happen to
            //       only use inlined device funcs that are non-virtual.
            //         But I think it is fragile.
            // TODO:   Segment info DATA (and map of seg-->seg info DATA) should be transferred
            //       just once, during add_segment (and erase during delete_segment)
            detail::SegImpl<FMT> * d_rbuf;
            checkCudaErrors(cudaMalloc((void**)&d_rbuf, sizeof(detail::SegImpl<FMT>)));
            checkCudaErrors(cudaMemcpy((void*)d_rbuf, (void*)this, sizeof(detail::SegImpl<FMT>), cudaMemcpyHostToDevice));
            uint32_t* total;
            checkCudaErrors(cudaMalloc((void**)&total, sizeof(uint32_t)));

#if CHK_HDR_ITER
            uint32_t* riter;
            checkCudaErrors(cudaMalloc((void**)&riter, size * sizeof(uint32_t)));
            dStorm::reduce_init<<<grid_construct, threads_construct>>>(d_rbuf, total, size, riter);
            if(CUDA_DBG) CudaCheckError();
#else
            dStorm::reduce_init<<<grid_construct, threads_construct>>>(d_rbuf, total, size);
            if(CUDA_DBG) CudaCheckError();
#endif
            nReduced = Impl::reduce(total);
#if CHK_HDR_ITER
            reduce_impl<<<grid_construct, threads_construct>>>(d_rbuf, size, riter);
            if(CUDA_DBG) CudaCheckError();
#else
            reduce_impl<<<grid_construct, threads_construct>>>(d_rbuf, size);
            if(CUDA_DBG) CudaCheckError();
#endif
#undef CHK_HDR_ITER

            return nReduced;
        }

    //
    // specialize SegImpl<FMT> for seg::VecGpu<float>
    //
    template<>
        template< typename IN_CONST_ITER >
        inline void
        detail::SegImpl<seg::VecGpu<float>>::store( IN_CONST_ITER iter, uint32_t const cnt,
                uint32_t const offset, double const wgt )
        {
            //typedef seg::VecGpu<float> FMT;
            // cuda compiler *incorrectly* warns about hiding template parameter
            // but in fact the above seems necessary. So change it too ...
            typedef seg::VecGpu<float> Fmt;

            static_assert( is_segment_format<Fmt>::value,
                    "ERROR: require is_segment_format<Fmt>");
            static_assert( Fmt::value == seg::VecGpu<float>::value, "This impl is only for VecGpu!");
            int const verbose=0; 
            assert( this->d != nullptr );
            assert( this->d->orm != nullptr );
            // cast to MsgHeader type allowing fiddling with HdrT<Internal>
            void *obuf = this->ptrObuf();
            if(verbose) ORM_COUT(this->d->orm, "SegImpl<float>::store(iter"<<(void*)iter <<",cnt="<<cnt<<",offset="<<offset<<") into obuf @ "<<(void*)obuf);
            MsgHeader<seg::Dummy> * mHdr = mem_as<MsgHeader<seg::Dummy>*>(obuf);
            int local_rank = atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
            CudaCheckError();
            // Set Fmt and pushes in header
            store_init<<<1,1>>> (iter, obuf, mHdr, Fmt::value);

            CudaCheckError();

            cu_pair< uint32_t /*nData*/, void* /*dataEnd*/ > * const
                //auto const
                stop
                = Impl::store( iter, cnt, offset, obuf, wgt );

            CudaCheckError();
            // Check correctness
            store_impl<<<1,1>>>(stop, cnt, mHdr, obuf, this->bufBytes);
            CudaCheckError();
            return;
        }
    //
    // specialize SegImpl<FMT> for seg::VecGpu<double>
    //
    template<>
        template< typename IN_CONST_ITER >
        inline void
        detail::SegImpl<seg::VecGpu<double>>::store( IN_CONST_ITER iter, uint32_t const cnt,
                uint32_t const offset, double const wgt )
        {
            //typedef seg::VecGpu<float> FMT;
            // cuda compiler *incorrectly* warns about hiding template parameter
            // but in fact the above seems necessary. So change it too ...
            typedef seg::VecGpu<float> Fmt;

            static_assert( is_segment_format<Fmt>::value,
                    "ERROR: require is_segment_format<Fmt>");
            static_assert( Fmt::value == seg::VecGpu<float>::value, "This impl is only for VecGpu!");
            int const verbose=0;
            assert( this->d != nullptr );
            assert( this->d->orm != nullptr );
            // cast to MsgHeader type allowing fiddling with HdrT<Internal>
            void *obuf = this->ptrObuf();
            if(verbose) ORM_COUT(this->d->orm, "SegImpl::store(iter"<<(void*)iter <<",cnt="<<cnt<<",offset="<<offset<<") into obuf @ "<<(void*)obuf);
            MsgHeader<seg::Dummy> * mHdr = mem_as<MsgHeader<seg::Dummy>*>(obuf);
            int local_rank = atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
            // Set Fmt and pushes in header
            store_init<<<1,1>>> (iter, obuf, mHdr, Fmt::value);

            cu_pair< uint32_t /*nData*/, void* /*dataEnd*/ > * const
                //auto const
                stop
                = Impl::store( iter, cnt, offset, obuf, wgt );

            if(CUDA_DBG) CudaCheckError();
            // Check correctness
            store_impl<<<1,1>>>(stop, cnt, mHdr, obuf, this->bufBytes);
            if(CUDA_DBG) CudaCheckError();
            return;
        }
#endif // YES CUDACC

}//dStorm::
#endif // SEGIMPL_HH_
