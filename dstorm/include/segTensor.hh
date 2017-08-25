/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef SEGVECDENSE_HH
#define SEGVECDENSE_HH

#include "segTensor.hpp"
#include "dstorm.hh"        // need SegBase, etc. to instantiate

#define ORM_COUT( STUFF ) do \
{ \
    std::ostringstream oss; \
    oss<< STUFF <<std::endl; \
    this->printf_fn( oss.str().c_str() ); \
}while(0)

/** debug */
//#define ORM_DBG( STUFF ) ORM_COUT( STUFF )
#define ORM_DBG( STUFF )

namespace dStorm
{
    namespace user
    {

        /** C++ iostream shortcut to print with this->printf_fn */
        template< typename T > DSTORM_INLINE
            Seg_Tensor<T>::Seg_Tensor( Hdr const& maxview,
                                       Printf_fn printf_override /*=nullptr*/ )
            : detail::SegBase< Seg_Tensor<T>, T >()       // This make us into an unusable stub
            , printf_fn( (printf_override == nullptr) ?  ::printf : printf_override )
            , h( maxview )
            {}

        template< typename T > DSTORM_INLINE
            Seg_Tensor<T>::~Seg_Tensor()
            {}

#if 0
        // and \c setSegInfo makes us into a usable segment definition
        template< typename T > DSTORM_INLINE
            void Seg_Tensor<T>::setSegInfo()           // NOTE: no args! now cnt
            {                                           // set by parent into info.cnt
                ORM_COUT(" Seg_Tensor<T>::setSegInfo...");
                // cast down to lowest-level base.
                // Some vars (info.cnt of our Tdata) already set.
                SegInfo& info = *this;

                // required stuff that upper layer doesn't know about
                info.datacode   = typeid(typename Base::Tdata).hash_code(); // paranoia?
                info.datasize   = sizeof(typename Base::Tdata);

                // our own stuff, in HdrT<fmt=Tensor> portion of MsgHeader?
                // ... We will write our MsgHeader::hdr.u "user" fields fully on every store.
                // ... We don't need to set any magic stuff there.

                // our own stuff (e.g. private variables?
                // ... We have none, but for example...
                //      typename Seg_Tensor<T>::magic_reduce_parm = 17.3865 "<<red[0]<<" , "<<red[1]<<" ");
                //      this->maxStaleness = 4.0 /*seconds*/;
                //      this->batchSize = 7U;
                //      etc.  any "globals" that the user's store/push/reduce cares about
                //
            }
        /** No private segment chores. */
        template< typename T > DSTORM_INLINE
            void Seg_VecDense<T>::add_segment()
            {
            }

        /** No private segment chores, or stats to print. */
        template< typename T > DSTORM_INLINE
            void Seg_VecDense<T>::delete_segment()
            {
            }


        template< typename T >
            template< typename IN_CONST_ITER >
            DSTORM_INLINE std::pair< uint32_t /*cnt*/, void* /*dataEnd*/ >
            Seg_Tensor<T>::store( IN_CONST_ITER& iIter,
                                    uint32_t cnt, uint32_t const offset,
                                    void* const buf )
            {
                static int verbose=0;
                std::pair<uint32_t,void*> ret(0U,nullptr);

                if(verbose) ORM_COUT(" Seg_Tensor<T>::store(iIter,cnt="<<cnt<<",offset="<<offset<<", buf@"<<(void*)buf<<" )");
                if(verbose) ORM_COUT("\t\t bufBytes = "<<this->bufBytes
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
                //                            ^^^ Tensor, in this case
                if(verbose) ORM_COUT("  hdr.a.iter = "<<mHdr->hdr.a.iter
                                     <<"  hdr.a.fmt  = "<<(int)(mHdr->hdr.a.fmt)
                                     <<"  seg::Tensor<float>::value = "<<(int)(seg::Tensor<float>::value)
                                    );
                {
                    //mHdr.hdr.a.iter = writeSeq;
                    //mHdr.hdr.a.fmt = Tensor;   // NOT our responsibility (I hope)
                    assert( mHdr->hdr.a.fmt == seg::Tensor<float>::value ); // sanity

                    // write OUR custom part of the header
                    mHdr->hdr.u.offset = offset;
                    Uidx i=0U;
                    if(1) { // check invariants
                        for(Uidx i=ndim; i<Meta::MaxDimensionality) {
                            assert( dim[i] = 0U; );
                            assert( inc[i] = 0U; );
                        }
                        // etc.
                    }
                    mHdr->hdr = this->hdr;      // plain copy of POD data
                }
                T *data = mem_as<T*>( buf, sizeof(*mHdr) ); // advance past hdr
                if(verbose) ORM_COUT(" T* data @ "<<(void*)data);

                // - Suppose our message has no useful "data" (maybe everything
                //  important is contained in our HdrT<Fmt> POD struct)
                // - One **could** return right away (perhaps good for debug?)
                if(0){
                    ret.first = 0U;                 // nData
                    ret.second = (void*)data;       // dataEnd
                    return ret;
                }

                { // write data
                    // don't overflow data area
                    uint32_t const datasz = /*SegInfo*/this->bufBytes
                        - (sizeof(MsgHeader<Fmt>) + sizeof(MsgTrailer));
                    if( cnt * sizeof(T) > datasz ){
                        cnt = datasz / sizeof(T);
                    }

                    std::advance( iIter, offset );

                    T* dataEnd = NULL;
                    assert( &*iIter != data );
                    if (&*iIter == data) 	{ 
                        dataEnd = &*(iIter + cnt);
                        printf ("Skipping ************\n \n");
                    }
                    else	{
                        dataEnd = std::copy_n( iIter, cnt, data );
                    }

                    // dataEnd ~ where copy stopped.
                    // XXX copy_n may not be optimal. For many IN_CONST_ITER
                    // may have SIMD memcpy available. Also, we know we can
                    // write [some] garbage past end of data area w/o segfault

                    assert( dataEnd == data + cnt );
                    //if( dataEnd < data+cnt ){       // short xfer?
                    //    // (maybe iIter hit end first?)
                    //    // Note: you may prefer to err
                    //    // XXX Is this enough to allow client to "continue" a
                    //    //     short transfer?  enough to implement auto-continue?
                    //    cnt = dataEnd - data;
                    //}
                    ret.first = cnt;
                    ret.second = (void*)dataEnd;
                }
                // NOT responsible for writing trailer at dataEnd.
                //    It will be overwritten as soon as we return.
                // In case of error, return nullptr
                return ret;
            }
        template< typename T > DSTORM_INLINE
            bool Seg_Tensor<T>::reduce( std::vector<uint32_t> const& rbufnums ) const
            {
                static int const verbose=0;
                size_t const rsz = rbufnums.size();
                if( rsz == 0U ){
                    return false;       // no more work required
                }
                // parent has already remove headers marked with fmt==Illegal

                // 1. Look for header and data pointers that may have new data
                struct RbufInfo {
                    uint32_t              bufnum;
                    MsgHeader<Fmt> const* hdr;
                    T const*              data;
                };
                std::vector< RbufInfo > rbis;
                // Initial scan to weed out input buffers with illegal headers (torn reads?)
                for( uint32_t r=0U; r<rbufnums.size(); ++r ){
                    uint32_t bufnum = rbufnums[r];
                    auto rHdr = mem_as<MsgHeader<Fmt> const* >(this->ptrBuf( bufnum ));
                    // prune any obviously illegal values (remember we are async)
                    if( rHdr->hdr.u.cnt + rHdr->hdr.u.off > this->cnt ){
                        continue; // possible for torn read (retry later)
                    }else if( rHdr->hdr.u.sz != this->datasize/*sizeof(T)*/ ){
                        continue; // should never happen
                    }else if( rHdr->hdr.a.fmt != Fmt::value ){
                        if(1||verbose) ORM_COUT("Consider supporting reducing input fmt "<<rHdr->hdr.a.fmt
                                                <<" into Tensor fmt "<<seg::Tensor<T>::name);
                        continue;
                    }else if( rHdr->hdr.u.off == -1U || rHdr->hdr.u.off == -2U ){
                        throw std::runtime_error(" illegal marker value encountered in SegTensor header");
                    }else{                                      // data stored in dense format.
                        T const* rData =  mem_as<T const*>( this->ptrBuf(bufnum),
                                                            sizeof(MsgHeader<Fmt>));
                        //assert( sizeof(MsgHeader<Fmt>) == this->hdrsz );
                        rbis.push_back( RbufInfo{ bufnum, rHdr, rData } );
                    }
                }

                // 2a. Handle trivial no-output cases
                if( rbis.size() == 0U
                    || (rbis.size() == 1U && rbis[0].hdr->hdr.u.cnt == 0U) )
                    return false/*more?*/;

                // 2b. Examine headers, detect homogenous offset + cnt case
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
                if( end_max - beg_min > this->SegInfo::cnt ){
                    // won't fit into one output buffer
                    throw std::runtime_error("Tensor reduction won't fit into output buffer");
                }

                // 3. Set up output segment (not a user buffer)
                void *red = this->ptrBuf( this->iBuf() );     // reduce into this buffer
                MsgHeader<Fmt> * redHdr = mem_as<MsgHeader<Fmt>*>( red );
                // Is the first send a full-size reduce?
                //*redHdr = *rHdrs[0];// FIXME (ok only for full-size reduce)
                uint32_t n = end_max - beg_min;
                redHdr->hdr.u.cnt = n;
                redHdr->hdr.u.off = beg_min;
                redHdr->hdr.u.sz     = sizeof(T);
                T* redData = mem_as<T*>(redHdr,sizeof(*redHdr));
                // User beware: if you cut corners and go directly to the reduce
                // "data" (without checking redHdr cnt & offset) it's up to you
                // to ensure safety (ex. by --always-- writing full-size dense
                // vectors).

                double const rszinv = 1.0/rsz;
                if( beg_min == beg_max && end_min == end_max ){
                    if(verbose) ORM_COUT(" homog-reduce, rbis.size()="<<rbis.size()<<" n="<<n);
                    // optimize homogenous offset,cnt for all input vectors
                    // XXX extend n upwards if can help SIMD reduce
                    // XXX rewrite/compare with blas atlas/MKL timings.
                    //ORM_COUT(" redHdr@"<<(void*)redHdr<<" redData@"<<(void*)redData);
                    redData += redHdr->hdr.u.off;
                    while( n-- ){
                        //ORM_COUT(" -------------- loop n="<<n);
                        for( auto & rbi: rbis ){
                            //ORM_COUT(" *(redData="<<(void*)redData<<") += (*(rbi.data="<<(void*)rbi.data<<") = "<<*rbi.data<<")");
                            *redData += *rbi.data;
                            ++rbi.data;
                            //ORM_COUT("\t\t rbi.data = "<<(void*)rbi.data);
                        }
                        *redData *= rszinv;
                        //ORM_COUT(" *redData *= (rszinv="<<rszinv<<") --> *redData = "<<*redData);
                        ++redData;
                    }
                }else{ // generic inhomogenous case
                    if(1||verbose) ORM_COUT(" inhom-reduce");
                    assert( nullptr == "inhomogenous SegTensor::reduce needs to be tested before use");
                    assert( rbis.size() >= 2U );
                    memset( redData, 0, sizeof( n * this->datasize ));
                    // Note: --could-- optimize memset area a little
                    // XXX this is WRONG.
                    //        At any point, we are averaging some VARYING NUMBER
                    //        of input sources. So the normalization factor for
                    //        the sum VARIES as we hit more-overlapped regions
                    //        of the vector.
                    // Absolutely need further pre-analysis to construct a vector
                    // of breakpoints and next-normalization factors, and then
                    // loop NOT over rbi but over vector-offsets.
                    for( auto & rbi: rbis ){
                        size_t const& off = rbi.hdr->hdr.u.off;
                        size_t cnt        = rbi.hdr->hdr.u.cnt;
                        T const* src  = rbi.data;
                        T*       dest = redData + off    - beg_min;
                        while( cnt-- )
                            *dest++ = *src++;
                    }
                }
                // TODO Check trailer for torn reads, warn if detected.
                //      Tears with offset/count changed could be really bad.
                return false/*more*/; // no more to do for reduce
            }
#endif

        // just to be sure...
        static_assert( sizeof(MsgHeader<seg::Tensor<float>>) % DSTORM_DATA_ALIGN == 0U,
                       "FIXME: wrong padding");
        static_assert( sizeof(MsgHeader<seg::Tensor<float>>) == MsgHeader<seg::Tensor<float>>::sz,
                       "FIXME: wrong padding");

    }//user::
}//dStorm::
#endif // SEGVECDENSE_HH

