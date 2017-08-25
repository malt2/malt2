/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef SEGVECDENSE_HH
#define SEGVECDENSE_HH

#include "segVecDense.hpp"
#include "dstorm.hh"            // need SegInfo and SegBase<> to be complete types.
#include "detail/float_ops_omp.hpp"

#define GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)

namespace dStorm
{
    namespace user
    {

        template< typename T > DSTORM_INLINE
            Seg_VecDense<T>::Seg_VecDense( SegInfo::SegPrintf segPrintf_ /*=nullptr*/ )
            : detail::SegBase< Seg_VecDense<T>, T >( "segVecDense", segPrintf_ )
#if TORN_CHECKS
              , nTorn(0U)
#endif
            {}

        template< typename T > DSTORM_INLINE
            Seg_VecDense<T>::~Seg_VecDense()
            {
#if TORN_CHECKS
                if( nTorn > 0U ){
                    SEG_COUT(" -Seg_VecDense, WARNING: "<<nTorn<<" torn buffers during reduce");
                }else{
                    SEG_COUT(" -Seg_VecDense,      ok: "<<nTorn<<" torn buffers during reduce");
                }
#endif
            }

        // and \c add_segment makes us into a usable segment definition
        template< typename T > DSTORM_INLINE
            void Seg_VecDense<T>::setSegInfo()     // NOTE: no args! now cnt
            {                                           // set by parent into info.cnt
                SEG_COUT(" Svd::setSegInfo...");
                // cast down to lowest-level base.
                // Some vars (info. cnt,policy,d,segNum,obuf,ibuf,rbuf,nbuf,seg_id,sizeofMsgHeader) already set.
                SegInfo& info = *this;

                // A SegImpl may decide to not support ANY of the non-default behaviors
                // Seg_VecDense has not yet needed to support inhomogenous subviews for reduce:
                //   (even if they are non-overlapping, which is still "easy" to code)
                if( (info.policy & RBUF_SUBVIEW_MASK) > RBUF_SUBVIEW_HOMOG ){
                    throw std::runtime_error("Seg_VecDense does not yet support the request RBUF_SUBVIEW_* features");
                }

                // required stuff that upper layer doesn't know about
                info.datacode   = typeid(typename Base::Tdata).hash_code(); // paranoia?
                info.datasize   = sizeof(typename Base::Tdata);

                // HERE: our own stuff, in HdrT<fmt=VecDense> portion of MsgHeader?
                // ... We will write our MsgHeader::hdr.u "user" fields fully on every store.
                // ... Seg_VecDense<T> doesn't need to set any magic stuff there.

                // HERE: our own stuff (e.g. private variables?
                // ... We have none, but for example...
                //      typename Seg_VecDense<T>::magic_reduce_parm = 17.3865 "<<red[0]<<" , "<<red[1]<<" ");
                //      this->maxStaleness = 4.0 /*seconds*/;
                //      this->batchSize = 7U;
                //      etc.  any "globals" that the user's store/push/reduce cares about
                //
            }

        /** No private segment chores. */
        template< typename T > DSTORM_INLINE
            void Seg_VecDense<T>::add_segment()
            {
                SEG_COUT(" Svd::add_segment NO-OP...");
            }

        /** No private segment chores, or stats to print. */
        template< typename T > DSTORM_INLINE
            void Seg_VecDense<T>::delete_segment()
            {
            }

        template< typename T >
            inline void* plainPointer( T* t ){
                return static_cast<void*>( t );
            }
        template< typename ITER >
            inline void* plainPointer( ITER iter ){
                return nullptr;
            }
        template< typename T >
            template< typename IN_CONST_ITER >
            DSTORM_INLINE std::pair< uint32_t /*cnt*/, void* /*dataEnd*/ >
            Seg_VecDense<T>::store( IN_CONST_ITER& iIter,
                                    uint32_t cnt, uint32_t const offset,
                                    void* const buf, double const wgt/*=1.0*/ )
            {
                int const verbose=0;

                // SCALE_DURING_STORE might be a SegPolicy item eventually (see dstorm_fwd.hpp) XXX

                std::pair<uint32_t,void*> ret(0U,nullptr);

                if(verbose>1) SEG_COUT(" Seg_VecDense<T>::store(iIter,cnt="<<cnt<<",offset="<<offset<<", buf@"<<(void*)buf<<" )");
                if(verbose>1) SEG_COUT("\t\t bufBytes = "<<this->bufBytes
                                       <<"\n\t\t\t sizeof(MsgHeader<Fmt> = "<<sizeof(MsgHeader<Fmt>)
                                       <<"\n\t\t\t sizeof(MsgTrailer = "<<sizeof(MsgTrailer));
                // NOTE: many of these assertions are here "for show"
                //       (they should be guaranteed by the internal caller)
                //
                // --- preconditions on buf ---
                // buf (ptr to a buf in our segment) must at least be within segment
                assert( (char*)buf >= (char*)this->mem
                        && (char*)buf < (char*)this->mem + this->segBytes );
                // buf must point to a beginning of a segment buffer
                assert( ((char*)buf - (char*)this->mem) % this->bufBytes == 0 );
                // buf must at least be big enough to store header and trailer area
                assert( this->bufBytes > sizeof(MsgHeader<Fmt>) + sizeof(MsgTrailer) );

                auto mHdr = mem_as< MsgHeader<Fmt>* >(buf);
                //                            ^^^ VecDense, in this case
                if(verbose>1) SEG_COUT("  hdr.a.iter = "<<mHdr->hdr.a.iter
                                       <<"  hdr.a.fmt  = "<<(int)(mHdr->hdr.a.fmt)
                                       <<"  seg::VecDense<float>::value = "<<(int)(seg::VecDense<float>::value)
                                      );
                {
                    //mHdr.hdr.a.iter = writeSeq;
                    //mHdr.hdr.a.fmt = VecDense;   // NOT our responsibility (I hope)
                    assert( mHdr->hdr.a.fmt == seg::VecDense<float>::value ); // sanity

                    // write OUR custom part of the header
                    mHdr->hdr.u.off = offset;
                    mHdr->hdr.u.cnt = cnt;
                    mHdr->hdr.u.wgt = (SCALE_DURING_STORE? 1.0: wgt);
                    mHdr->hdr.u.sz  = sizeof(T);
                }
                T *data = mem_as<T*>( buf, sizeof(*mHdr) ); // advance past hdr
                if(verbose>1) SEG_COUT(" T* data @ "<<(void*)data);

                // - Suppose our message has no useful "data" (maybe everything
                //  important is contained in our HdrT<Fmt> POD struct)
                // - One **could** return right away (perhaps good for debug?)
                if(0){
                    ret.first = 0U;                 // nData
                    ret.second = (void*)data;       // dataEnd
                    return ret;
                }

                uint32_t nCopy = 0U;
                uint32_t nCopySkipped = 0U;
                { // write data
                    // don't overflow data area
                    uint32_t const datasz = /*SegInfo*/this->bufBytes
                        - (sizeof(MsgHeader<Fmt>) + sizeof(MsgTrailer));
                    if( cnt * sizeof(T) > datasz ){
                        if(0){
                            cnt = datasz / sizeof(T);   // original version silently chopped shorter
                        }else{ // *** NEW ***
                            throw std::length_error("Tried to store too many items in a Dstorm segment");
                        }
                    }

                    std::advance( iIter, offset );

                    // Skip copying data during store when operating from orm buffer
                    //typename std::iterator_traits<IN_CONST_ITER>::value_type* dataEnd = NULL;
                    //T* dataEnd = NULL;

                    if( wgt == 0.0 ){
                        // weight zero transfer simplified to a zero-length transfer.
                        if(verbose>1) this->segPrintf ("store wgt=0.0 treated as zero-length vector");
                        //memset( (void*)data, 0, cnt );
                        //dataEnd = (data + cnt);  // for later assertion
                        // ... just set vector length to zero
                        mHdr->hdr.u.cnt = 0U;

                    }else if ((void const*)(&*iIter) != (void const*)data)
                        //  ohoh. must be careful if *Iter is an iterator class and not a plain pointer
                        //  because otherwise we will get 'taking address of temporary' error.
                        //if( plainPointer(iIter) != (void const*)data )
                    { 
                        ++nCopy;

                        if( ! SCALE_DURING_STORE || wgt==1.0 ){
                            if(verbose>1) this->segPrintf ("Copying data.\n");
                            // XXX copy_n may not be optimal. For many IN_CONST_ITER
                            // may have SIMD memcpy available. Also, we know we can
                            // write [some] garbage past end of data area w/o segfault
                            /*dataEnd =*/ std::copy_n( iIter, cnt, data );
                        }else{
                            // XXX blas support in dstorm?  Separate from MILDE?
                            if(verbose>1) this->segPrintf ("Copying+Scaling data.\n");
                            for(uint32_t i=0U; i<cnt; ++i){
                                data[i] = *iIter++ * wgt;
                            }
                        }
                    }else{                      // input ptr != data destination
                        if( ! SCALE_DURING_STORE || wgt==1.0 ){
                            ++nCopySkipped;
                            if(verbose>1) SEG_COUT(" SKIPPING copy_n( iIter, cnt = "<<cnt<<", data @ "<<(void*)data<<") because &*iIter == data ******\n\n");
                            //dataEnd = (data + cnt);  // for later assertion
                        }else{ // scale_during_store
#if ! defined(_OPENMP)
                            for(uint32_t i=0U; i<cnt; ++i) data[i] *= wgt;
#else
                            if( cnt >= 65536U ) // <---- just guessing at these cutoffs :(
#if GCC_VERSION > 40900
#pragma omp parallel for simd
#else
#pragma omp parallel for
#endif
                                for(uint32_t i=0U; i<cnt; ++i) data[i] *= wgt;
                            else if( cnt >= 8192U )
#if GCC_VERSION > 40900
#pragma omp parallel for simd num_threads(2)
#else
#pragma omp parallel for num_threads(2)
#endif
                                for(uint32_t i=0U; i<cnt; ++i) data[i] *= wgt;
                            else
                                for(uint32_t i=0U; i<cnt; ++i) data[i] *= wgt;
#endif
                        }
                    }

                    //if( dataEnd < data+cnt ){       // short xfer?
                    //    // (maybe iIter hit end first?)
                    //    // Note: you may prefer to err
                    //    // XXX Is this enough to allow client to "continue" a
                    //    //     short transfer?  enough to implement auto-continue?
                    //    cnt = dataEnd - data;
                    //}
                    ret.first = cnt;
                    ret.second = (void*)(data + mHdr->hdr.u.cnt); // dataEnd
                }
                if(verbose>0){
                    if( nCopy == 0 ){
                        ; // all is good
                    }else if( nCopySkipped == 0 ){
                        this->segPrintf(" Dstorm::store: 100\% copies :(\r");
                    }else{
                        this->segPrintf(" Dstorm::store: %u / %u copies (?)\n",(unsigned)nCopy, (unsigned)nCopy + nCopySkipped);
                    }
                }
                // NOT responsible for writing trailer at dataEnd.
                //    It will be overwritten as soon as we return.
                // In case of error, return nullptr
                return ret;
            }

        template< typename T > DSTORM_INLINE
            uint32_t Seg_VecDense<T>::reduce( std::vector<uint32_t> const& rbufnums ) const
            {
                // XXX Farley: Allow REDUCE_NOP for ADMM reductions.
                if((this->policy & REDUCE_OP_MASK) == REDUCE_NOP) return rbufnums.size();

                static int const verbose=0;
                assert( (this->policy & SEG_LAYOUT_MASK) == SEG_FULL ); // oBuf, iBuf and rBufs all distinct
                assert( (this->policy & REDUCE_OP_MASK)  != REDUCE_NOP );
                // SegImpls can err for any unsupported subview situations within the rBuffs
                //    Seg_VecDense has reduce loops coded only for homogenous or nonovlp cases
                //assert( (this->policy & RBUF_SUBVIEW_MASK) != RBUF_SUBVIEW_ANY );
#ifndef NDEBUG
                {
                    // Apparently oBuf may have been set up yet.
                    MsgHeader<Fmt> const* obufHdr = mem_as<MsgHeader<Fmt> const*>(this->ptrBuf(this->obuf));
                    assert( obufHdr->hdr.u.off == 0U );              // oBuf is full-sized
                    // FAILED: assert( obufHdr->hdr.u.cnt == this->cnt );       // oBuf is full-sized
                    // FAILED: assert( obufHdr->hdr.u.sz  == sizeof(T) );       // oBuf already initialized
                }
#endif

                if( rbufnums.size() == 0U ){
                    return 0U;       // no more work required
                }
                // parent has already remove headers marked with fmt==Illegal

                // 1. Look for header and data pointers that may have new data
                // XXX TODO reduce should make a snapshot copy of MsgHeader
                struct RbufInfo {
                    uint32_t              bufnum;
                    MsgHeader<Fmt> const* hdr;
                    T const*              data;
                    uint_least64_t        iter; // NEW: jul'15, better tear tracking
                };
                std::vector< RbufInfo > rbis;
                // Initial scan to weed out input buffers with illegal headers (torn reads?)
                SegPolicy const subviewPolicy = this->policy & RBUF_SUBVIEW_MASK;
                for( uint32_t r=0U; r<rbufnums.size(); ++r ){
                    uint32_t const bufnum = rbufnums[r];
#if TORN_CHECKS
                    uint_least64_t bytes;
#endif
                    assert( bufnum < this->nbuf );
                    auto rHdr = mem_as<MsgHeader<Fmt> const* >(this->ptrBuf( bufnum ));
                    // prune any obviously illegal values (remember we are async, and may get torn reads!)
                    if( rHdr->hdr.a.fmt == seg::Illegal::value ) {
                        // Can happen if sender set illegal before RDMA sent the buffer (in async mode)
                        SEG_COUT("Illegal rbuf fmt: rbuf "<<r);
                        continue;
                    }else if( rHdr->hdr.a.fmt != Fmt::value ){
                        SEG_COUT("Consider supporting reducing input fmt "<<rHdr->hdr.a.fmt
                                 <<" into VecDense fmt "<<seg::VecDense<T>::name);
                        continue;
                    }else if( subviewPolicy == RBUF_SUBVIEW_NONE // if ASKED not to support subviews
                              && !(rHdr->hdr.u.off==0U && rHdr->hdr.u.cnt == this->SegInfo::cnt) ){
                        SEG_COUT("RBUF_SUBVIEW_NONE exception (rbuf ignored)");
                        continue; // could also be a torn read?
                    }else if( rHdr->hdr.u.off == -1U || rHdr->hdr.u.off == -2U ){
                        //throw std::runtime_error(" illegal marker value encountered in SegVecDense header");
                        SEG_COUT(" illegal hdr.u.off value in SegVecDense header (rbuf ignored)");
                        continue;
#if TORN_CHECKS
                    }else if( (bytes = rHdr->hdr.a.bytes) != rHdr->hdr.u.cnt*sizeof(T) + 2*sizeof(MsgHeader<Fmt>)){
                        SEG_COUT(" inconsistent hdr byte/count field: bytes="<<bytes
                                 <<" but cnt="<<rHdr->hdr.u.cnt<<"*sizeof(T):"<<sizeof(T)
                                 <<" + 2 * sizeof(MsgHeader<Fmt>):"<<sizeof(MsgHeader<Fmt>)
                                 <<" is "<<  rHdr->hdr.u.cnt*sizeof(T)
                                 +  2*sizeof(MsgHeader<Fmt>));
                        continue;
#endif
                    }else if( rHdr->hdr.u.cnt + rHdr->hdr.u.off > this->cnt ){
                        SEG_COUT(" illegal off/cnt (rbuf ignored)");
                        continue; // possible for torn read (retry later)
                    }else if( rHdr->hdr.u.sz != this->datasize/*sizeof(T)*/ ){
                        SEG_COUT(" illegal data size (rbuf ignored)");
                        continue; // should never happen (don't support mixed data types in rbuf)
#if TORN_CHECKS // NEW: skip trailer iter != hdr iter, because the reduce MIGHT be torn:
                    }else if( mem_as<MsgTrailer const*>(this->ptrBuf(bufnum),
                                                        bytes - sizeof(MsgHeader<Fmt>))
                              ->hdr.a.iter != rHdr->hdr.a.iter ){
                        // This MIGHT result in a torn read, or it might not, depending on
                        // whether reduce catches up with the rdma writing or not.
                        if(verbose>0) SEG_COUT(" rdma still in progress");
                        continue;
#endif
                    }else{                                      // data stored in dense format.
                        T const* rData =  mem_as<T const*>( this->ptrBuf(bufnum),
                                                            sizeof(MsgHeader<Fmt>));
                        //assert( sizeof(MsgHeader<Fmt>) == this->hdrsz );
                        rbis.push_back( RbufInfo{ bufnum, rHdr, rData, rHdr->hdr.a.iter } );
                    }
                }

                // 2. Handle trivial no-output case
                if( rbis.size() == 0U
                    || (rbis.size() == 1U && rbis[0].hdr->hdr.u.cnt == 0U) ) {
                    return 0U;  // zero incoming reduce buffers were averaged
                }

                // 3. Examine headers, detect homogenous offset + cnt case
                //     Extended to detect other "nice" cases (disjoint also easy)
                uint32_t beg_min = rbis[0].hdr->hdr.u.off;
                uint32_t beg_max = beg_min;
                uint32_t end_min = beg_min + rbis[0].hdr->hdr.u.cnt;
                uint32_t end_max = end_min;
                for( auto const rbi: rbis ){
                    uint32_t const beg = rbi.hdr->hdr.u.off;
                    uint32_t const end = beg + rbi.hdr->hdr.u.cnt;
                    if( beg < beg_min )      beg_min = beg;
                    else if( beg > beg_max ) beg_max = beg;
                    if( end < end_min )      end_min = end;
                    else if( end > end_max ) end_max = end;
                }
                uint32_t const n = end_max - beg_min;
                bool const homogenous = (beg_min == beg_max && end_min == end_max);

                // 3b. Reconcile rBuf subview status with requested and implemented behaviors
                //     Err , return 0 if we cannot proceed.
                if( n > this->SegInfo::cnt ){ // won't fit into one output buffer (should never occur)
                    throw std::runtime_error("reduction won't fit into output buffer");
                }
                if( subviewPolicy == RBUF_SUBVIEW_NONE ){
                    // reduce policy says rBuf cnt required to go from 0 to cnt exactly
                    // Note: This error ought to be caught at the store/push stage, but
                    //       might it happen due to bit flip error during xmit?
                    // Note: is cnt rounded upward for cache/simd register size?
                    //       If so, "full-length" may have some slop to it!.
                    if( !(beg_max == 0U && end_min == this->cnt/*max data items, from SegInfo*/) ){
                        std::ostringstream oss;
                        oss<<"segment policy requires full-sized buffers: beg_max="<<beg_max<<" end_min="<<end_min<<" but SegInfo::cnt is "<<this->cnt;
                        SegInfo::policyError(oss.str());
                        return 0U;      // if we didn't throw, say "zero rBufs were reduced"
                    }
                }else if( subviewPolicy == RBUF_SUBVIEW_HOMOG ){
                    // Seg_VecDense lacks code to handle any inhomogeneous rBuf subviews.
                    if( ! homogenous ){
                        SegInfo::policyError("reduction does not support inhomogenous rBuf subviews yet");
                        return 0U;
                    }
                }else if( subviewPolicy == RBUF_SUBVIEW_HOMOG_OR_NONOVLP && !homogenous ){
                    // we now support RBUF_SUBVIEW_HOMOG_OR_NONOVLP
                    //            and RBUF_SUBVIEW_OVLP_RELAXED
                    bool ovlp = false;
                    // must explicitly for strict nonolvp.
                    for( uint32_t i=0U; ovlp==false && i<rbis.size(); ++i ){
                        uint32_t const ibeg = rbis[i].hdr->hdr.u.off;
                        uint32_t const iend = ibeg + rbis[i].hdr->hdr.u.cnt;
                        for( uint32_t j=i+1U; j<rbis.size(); ++j ){
                            uint32_t const jbeg = rbis[j].hdr->hdr.u.off;
                            uint32_t const jend = jbeg + rbis[j].hdr->hdr.u.cnt;
                            // assume well-formed ranges, jbeg <= jend and ibeg <= iend
                            if( jbeg < iend && ibeg < jend ){ // OVLP detected
                                ovlp = true;
                                break;
                            }
                        }
                    }
                    if( ovlp == true ){
                        SegInfo::policyError("reduction RBUF_SUBVIEW_HOMOG_OR_NONOVLP detected an overlap");
                        return 0U;
                    }
                }else if( subviewPolicy == RBUF_SUBVIEW_OVLP_RELAXED ){
                    ; // if inhomogenous,  ** pretend ** no overlaps, and use the no-overlap loop anyway.
                }else if( subviewPolicy == RBUF_SUBVIEW_ANY ){
                    SegInfo::policyError("reduction does not support inhomogenous rBuf subviews with strict correctness (RBUF_SUBVIEW_ANY)");
                    return 0U;
                }

                // 4. Handle trivial single-input case as plain copy
                SegPolicy const op = this->policy & REDUCE_OP_MASK;
                if( !(op == REDUCE_SUM_RBUF || op == REDUCE_AVG_RBUF || op == REDUCE_AVG_RBUF_OBUF) ){
                    throw std::runtime_error("SegVecDense unimplemented reduction operation");
                }
                T* redData = nullptr;
                // Note: float_ops.hpp handles rbis.size()==1 case WITHOUT this special logic
                if( rbis.size() == 1U ) {
                    // XXX I think the new favg:: and fsum:: namespace funcs deal with this special case just as efficiently now
                    if( op == REDUCE_SUM_RBUF || op == REDUCE_AVG_RBUF ){
                        // 4a. Set up 'iBuf' output buffer as sub-view matching the rBuf areas
                        //     (full view would need to zero the extra elements, inefficient)
                        void *red = this->ptrBuf( this->ibuf );     // reduce into this buffer
                        MsgHeader<Fmt> * redHdr = mem_as<MsgHeader<Fmt>*>( red );
                        redHdr->hdr.u.off = rbis[0].hdr->hdr.u.off;
                        redHdr->hdr.u.cnt = rbis[0].hdr->hdr.u.cnt;
                        redHdr->hdr.u.sz     = sizeof(T);
                        redData = mem_as<T*>(redHdr,sizeof(*redHdr));
                        memcpy( (void*)redData, (void const*)rbis[0].data, redHdr->hdr.u.cnt * redHdr->hdr.u.sz );
                    }else{ assert( op == REDUCE_AVG_RBUF_OBUF );
                        // 4b. Set up averaging with sub-view of full vector in oBuf
                        void *red = this->ptrBuf( this->obuf );     // "oBuf += rBuf"
                        MsgHeader<Fmt> * redHdr = mem_as<MsgHeader<Fmt>*>( red );

                        // recall:
                        //   assert( obufHdr->hdr.u.off == 0U );              // oBuf is full-sized
                        //   // FAILED: assert( obufHdr->hdr.u.cnt == this->cnt );       // oBuf is full-sized
                        //   // FAILED: assert( obufHdr->hdr.u.sz  == sizeof(T) );       // oBuf already initialized
                        // so let's assume it has been "zeroed out" and fix up just the header (every time)
                        // (Should hdr be set during add_segment, just once?)
                        assert( redHdr->hdr.u.off == 0U );
                        redHdr->hdr.u.cnt = this->cnt;
                        redHdr->hdr.u.sz  = sizeof(T);

                        redData = mem_as<T*>(redHdr,sizeof(*redHdr));
                        redData += rbis[0].hdr->hdr.u.off;

                        favg::upd_restrict1( redData, n, rbis[0].data ); // "redData = 0.5 * (redData + rbiData)"
                        // long-hand version:
                        //uint_least32_t const cnt = rbis[0].hdr->hdr.u.cnt;
                        //T const* rbiData = rbis[0].data;
                        //for(T* const redEnd = redData+cnt; redData != redEnd; ++rbiData, ++redData ){
                        //    *redData = 0.5 * (*redData + *rbiData);
                        //}
                    }
                    // 7. Check and return if potentially torn reads XXX TBD
                    // -- remember 'iter' before memcpy,
                    //    - ok, rbis[r].iter
                    // -- Check it's equal to iter in trailer after mempcy
                    //    - TBD
                    // -- AND check it's equal to iter in header after memcpy 
#if TORN_CHECKS
                    if(1){
                        for( uint32_t r=0U; r<rbis.size(); ++r ){
                            volatile MsgHeader<Fmt> const* rHdr = rbis[r].hdr;

                            uint_least64_t const iter0 = rbis[r].iter;  // original value
                            uint_least64_t const iter = rHdr->hdr.a.iter; // current value
                            uint32_t  torn = 0U;
                            if( iter != iter0 ) torn=1U;
                            else if( rHdr->hdr.a.fmt == seg::Illegal::value ) torn=2U;
                            else if( rHdr->hdr.a.bytes != rHdr->hdr.u.cnt * sizeof(T)
                                     + 2*sizeof(MsgHeader<Fmt>)) torn=3U;
                            else{
                                void const* dataEnd = rbis[r].data/* T const * */
                                    + rHdr->hdr.u.cnt;
                                MsgTrailer const* trailer = mem_as<MsgTrailer const*>(dataEnd);
                                if( trailer->hdr.a.iter != iter0 )
                                    torn=4U;
                            }
                            if( torn ){
                                // Next line can produce lots of output:
                                if(verbose>1) SEG_COUT("post-reduce tear (reason "<<torn<<")");
                                ++this->nTorn;
                            }
                        }
                    }
#endif
                    return 1U;
                }// end rbis.size()==1 subcase.

                // 5. Set up output segment for rBuf items spanning range n=end_max-beg_min
                if( op == REDUCE_SUM_RBUF || op == REDUCE_AVG_RBUF ){
                    void *red = this->ptrBuf( this->ibuf );     // reduce into this buffer
                    MsgHeader<Fmt> * redHdr = mem_as<MsgHeader<Fmt>*>( red );
                    // handle homogenous subview reduce too (set up iBuf header region)
                    redHdr->hdr.u.off = beg_min;
                    redHdr->hdr.u.cnt = n;
                    redHdr->hdr.u.sz  = sizeof(T);
                    redData = mem_as<T*>(redHdr,sizeof(*redHdr));
                    // User beware: if you cut corners and go directly to the reduce
                    // "data" (without checking redHdr cnt & offset) it's up to you
                    // to ensure safety (ex. by --always-- writing full-size dense
                    // vectors).
                }else{ assert( op == REDUCE_AVG_RBUF_OBUF );
                    // In contrast, reduce with obuf assumes obuf is full-length.
                    //              "oBuf = (oBuf + sum(rBufs)) / (rbis.size()+1)"
                    //              ... over the subview spanned by the rBufs
                    void *red = this->ptrBuf( this->obuf );
                    MsgHeader<Fmt> * redHdr = mem_as<MsgHeader<Fmt>*>( red );
                    // Hmmm. if oBuf hdr is not set up, we should do so...
                    redData = mem_as<T*>(redHdr,sizeof(*redHdr));
                    // DO NOT push redData forward for subviews (beg_min > 0) yet.
                }

                // 6. Perform averaging
                assert( rbis.size() > 1U );
                if( homogenous ){
                    // 6a. homogenous span for all rbufs

                    // We now streamline with nice call to RDMA-friendly floating point ops (float_ops.hpp)
                    //
                    // 1. redData[0..cnt-1] is some floating point T*
                    // 2. rbis is a std::vector<S> where S::data are unaliased ptrs to T const*[0..cnt-1]
                    //
                    // NOTE: rbis[i].data members are MODIFIED to point at rbis[i].data+cnt
                    //       This is OK, because we don't need them any more.
                    if( op == REDUCE_AVG_RBUF ){
                        if(verbose) SEG_COUT(" homog-REDUCE_AVG_RBUF, rbufnmums.size()="<<rbufnums.size()
                                             <<" rbis.size()="<<rbis.size()<<" n="<<n);
                        favg::set( redData, n, rbis );  
                    }else if( op == REDUCE_SUM_RBUF ){
                        //assert(nullptr=="untested code block");
                        if(verbose) SEG_COUT(" homog-REDUCE_SUM_RBUF, rbufnmums.size()="<<rbufnums.size()
                                             <<" rbis.size()="<<rbis.size()<<" n="<<n);
                        fsum::set( redData, n, rbis ); // rbis[i].data[0..n-1] are unaliased T const*
                    }else{
                        assert( op == REDUCE_AVG_RBUF_OBUF );
                        redData += beg_min;      // NOW bump obuf (full vector) up to homogenous offset value
                        if(verbose) SEG_COUT(" homog-REDUCE_AVG_RBUF_OBUF, rbufnmums.size()="<<rbufnums.size()
                                             <<" rbis.size()="<<rbis.size()<<" n="<<n);
                        favg::upd( redData, n, rbis );  // rbis[i].data[0..n-1] are unaliased T const* [T=floating point type]
                    }
#if TORN_CHECKS
                    if(1){       // TORN checks... (again)
                        for( uint32_t r=0U; r<rbis.size(); ++r ){
                            volatile MsgHeader<Fmt> const* rHdr = rbis[r].hdr;

                            uint_least64_t const iter0 = rbis[r].iter;  // original value
                            uint_least64_t const iter = rHdr->hdr.a.iter; // current value
                            uint32_t  torn = 0U;
                            if( iter != iter0 ) torn=1U;
                            else if( rHdr->hdr.a.fmt == seg::Illegal::value ) torn=2U;
                            else if( rHdr->hdr.a.bytes != rHdr->hdr.u.cnt * sizeof(T)
                                     + 2*sizeof(MsgHeader<Fmt>)) torn=3U;
                            else{
                                void const* dataEnd = rbis[r].data/* T const * */
                                    + rHdr->hdr.u.cnt;
                                MsgTrailer const* trailer = mem_as<MsgTrailer const*>(dataEnd);
                                if( trailer->hdr.a.iter != iter0 )
                                    torn=4U;
                            }
                            if( torn ){
                                // Next line can produce lots of output:
                                if(verbose>1) SEG_COUT("post-reduce tear (reason #"<<torn<<")");
                                ++this->nTorn;
                            }
                        }
                    }
#endif
                }else{
                    // 6b. nonovlp inhomogenous case -- loop individually over rbis and
                    //                                  accumulate them, assuming no ovlp
                    //     (generic inhomogenous case IS NOT SUPPORTED YET -- error msg, above)
                    //      Generic MIGHT need further pre-analysis to construct a vector
                    //      of breakpoints and next-normalization factors, and then
                    //      break up the loop over these vector offsets.
                    if(1||verbose) SEG_COUT(" inhomog-nonovlp-reduce");
                    assert( rbis.size() >= 2U );
                    assert(nullptr=="inhomogenous + nonovlp SegVecDense::reduce needs to be reviewd and tested before use");
                    // next line sets up output range beg_min..end_max instead, I suppose.
                    memset( redData, 0, sizeof( n * this->datasize ));
                        // step 5 set redData buffer start a u.off = beg_min
                    if( op == REDUCE_AVG_RBUF ){
                        assert(nullptr=="This code block is still wrong: but OK if input vector ranges NEVER overlap");
                        for( auto & rbi: rbis ){
                            size_t const off = rbi.hdr->hdr.u.off;
                            size_t cnt        = rbi.hdr->hdr.u.cnt;
                            T const* src  = rbi.data;
                            T*       dest = redData + off    - beg_min;
                            memcpy( (void*)dest, (void const*)src, cnt );
                            // Also possible (instead of memcpy):
                            //while( cnt-- )
                            //    *dest++ = *src++;
                        }
                    }else if( op == REDUCE_SUM_RBUF ){
                        // NEED: memset( full redData buffer, 0, full-size * sizeof(T) );
                        for( auto & rbi: rbis ){
                            size_t const off = rbi.hdr->hdr.u.off;
                            size_t cnt        = rbi.hdr->hdr.u.cnt;
                            T const* src  = rbi.data;
                            T*       dest = redData + off    - beg_min;
                            memcpy( (void*)dest, (void const*)src, cnt );
                            // Also possible (instead of memcpy):
                            //while( cnt-- )
                            //    *dest++ = *src++;
                        }
                    }else{ assert( op == REDUCE_AVG_RBUF_OBUF );
                        // step 5 set redData buffer start of oBuf (existing, full, offset 0, not beg_min)
                        //        assert( obufHdr->hdr.u.off == 0U );
                        for( auto & rbi: rbis ){
                            size_t const off = rbi.hdr->hdr.u.off;
                            size_t const cnt = rbi.hdr->hdr.u.cnt;
                            T const* src  = rbi.data;
                            // step 5 set redData buffer start a u.off = beg_min
                            T*       dest = redData + off;
                            favg::upd_restrict1( dest, cnt, src );
                        }
                    }
                    return 1U; // inhomog non-ovlp always are using "1" rBuf for sum/avg
                    //            DO NOT return rbis.size() for non-overlapped case !
                }// end 6b. nonovlp inhomogenous case

                // 7. Check trailer for torn reads
                //      Tears with offset/count changed could be really bad.
                // TODO RETURN #tears if detected.
                return rbis.size();
            }

        // just to be sure...
        static_assert( sizeof(MsgHeader<seg::VecDense<float>>) % DSTORM_DATA_ALIGN == 0U,
                       "FIXME: wrong padding");
        static_assert( sizeof(MsgHeader<seg::VecDense<float>>) == MsgHeader<seg::VecDense<float>>::sz,
                       "FIXME: wrong padding");

    }//user::
}//dStorm::
#endif // SEGVECDENSE_HH

