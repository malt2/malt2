/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef SEGINFO_HH_
#define SEGINFO_HH_

#include "segInfo.hpp"
#include "dstorm_msg.hpp"       // dStorm::seg:: namespace + mem_as pointer[+offset] conversion

#include <sstream>

/** remove printf differences for WITH_LIBORM (or not).
 * DST_PRINTF = "distributed printf" */
#if WITH_LIBORM
#define DST_PRINTF( ORMPTR, ... ) (ORMPTR)->printf( (ORMPTR), __VA_ARGS__ )
//#define DST_PRINTF( ... ) (*OrmPrintf)( (ORMPTR), __VA_ARGS__ )
/** C++ iostream shortcut for <Orm>_printf */
#define ORM_COUT( ORMPTR, STUFF ) do{ \
    std::ostringstream tmp_oss; \
    tmp_oss<< STUFF <<std::endl; \
    DST_PRINTF( ORMPTR, tmp_oss.str().c_str() ); \
}while(0)
/** cout macro, routing through DST_PRINTF (using liborm if available).
 * DST_COUT = "distributed cout" */
#define DST_COUT( STUFF ) do{ \
    std::ostringstream orm_cout_oss; \
    orm_cout_oss<< STUFF <<std::endl; \
    DST_PRINTF( this->orm, orm_cout_oss.str().c_str() ); \
}while(0)

#else
#error "Do not know how to define ORM_COUT (segInfo.hh)"
#endif

namespace dStorm {
    /** C++ iostream shortcut to print with this->segPrintf. segPrintf is always
     * non-NULL, defaulting to plain printf. Seg_VecDense<T> accepts an argument
     * if you want to override this. orm->printf might be a good choice in a
     * distributed setting, to make sure you see output from all ranks. */
#define SEG_COUT( STUFF ) do \
    { \
        std::ostringstream oss; \
        oss<< STUFF <<std::endl; \
        this->segPrintf( oss.str().c_str() ); \
    }while(0)

    /** debug */
#if 1
#define SEG_DBG( STUFF )
#else
#define SEG_DBG( STUFF ) do { \
    { \
        std::ostringstream oss; \
        oss<< STUFF <<std::endl; \
        if( this->d != nullptr && this->d->orm != nullptr) DST_PRINTF( this->d->orm, oss.str() ) ; \
    }else{ \
        this->segPrintf( oss.str().c_str() ); \
    } \
}while(0)
#endif

    inline void SegInfo::policyError(char const* msg) const
    {
        SegPolicy const handling = this->policy && SUBVIEW_ERR_MASK;
        if( handling == SUBVIEW_ERR_THROW ){
            throw std::runtime_error(msg);
        }else if( handling == SUBVIEW_ERR_WARN ){
            this->segPrintf("%s **warning**: %s\n", segVecName, msg);
        }
        // else SUBVIEW_ERR_IGNORE, do nothing.
    }
    inline void SegInfo::policyError(std::string msg) const
    {
        this->policyError(msg.c_str());
    }

    namespace detail {

        inline void AwaitCountedSet::set( uint32_t const ix ){
            assert( this->awaitCnt > 0U );
            assert( ix < awaitCnt );
            if( done[ix] ){
                std::ostringstream oss;
                oss<<" duplicate AwaitCountedSet:set(ix)";
                if( descr ) oss<<" "<<descr;
                throw std::runtime_error(oss.str());
            }
            done[ix] = true;
            ++nDone;
        }

        /** - some boilerplate SegInfo init could be moved here */
        template< typename IMPL, typename TDATA > inline
            SegBase<IMPL,TDATA>::SegBase( char const* segVecName/*="Vec"*/
                                          , typename TSegInfo::SegPrintf segPrintf /*=nullptr*/ )
            : IMPL::SegInfo_t(segVecName,segPrintf)

              {}

        /** Is this used?
         * - NOTE: only benefits here are:
         *   - cast to proper data type
         *   - sizeof being compile-time-const
         * - otherwise could use segInfo ptrData (with a cast)
         * - <em>consider removing this!</em>
         */
        template< typename IMPL, typename TDATA > inline
            TDATA* SegBase<IMPL,TDATA>::data(uint32_t bufNum) const {
                return mem_as<TDATA*>( TSegInfo::ptrBuf(bufNum)
                                       , sizeof(MsgHeader<typename IMPL::Fmt>));
            }

    }//detail::

    inline detail::AwaitCountedSet& SegInfo::getReduceState() const
    {
        assert( this->reduceState != nullptr );
        assert( (this->policy & REDUCE_OP_MASK) == REDUCE_STREAM );
        assert( (this->policy & SEGSYNC_MASK) == SEGSYNC_NOTIFY_ACK );
        return *this->reduceState;
    }

}//dStorm::
#endif // SEGINFO_HH_
