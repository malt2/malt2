/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef SEGVECDENSE_HPP
#define SEGVECDENSE_HPP

#include "dstorm.hpp"
#include "segInfo.hpp"

/** NEW: print count of torn buffers encountered in reduce during destructor */
#define TORN_CHECKS 1

namespace dStorm {

    namespace user {

        /** A "real" segment handler, with add_segment, store (, etc)
         * implementations of dstorm functions.
         *
         * How to use?  Given SegInfo it's void* impl is reinterpret_cast
         * to typeof(*this) unambiguously.
         *
         * Actually we always have a SegImpl object above us, that knows
         * about message structure and does some grunt work for us, but doesn't
         * know (eg) that we store 'floats'.
         *
         * Knowledge that we store 'floats' is provided by \c SegBase, from whom
         * we should derive. \c SegBase, in turn, derives from \c SegInfo (untemplated).
         *
         * Here, we handle a \c MYTYPE = \c VecDense format for buffers in a segment.
         * You can implement your own MYTYPE by:
         *
         * 1. provide a \c Seg_MYTYPE handler (~ Seg_VecDense),
         * 1. derive \c Seg_MYTYPE from a \c SegBase of that datatype
         *    used by \c Seg_MYTYPE.
         * 1. install a \c MsgFormats enum value named \c MYTYPE
         *    into \ref dstorm_msg.hpp
         * 1. provide a SegTag<VecDense> type-lookup metaclass,
         *    whose \c type typedef is a \c Seg_MYTYPE,
         * 1. provide a fixed-size HdrT<MYTYPE> POD of msg header info that
         *    Seg_MYTYPE will \c store into every \c MsgHeader user-area of the buffer.
         * 1. we need orm access to JUST the printf function, so supply
         * just that part of orm in the constructor
         */
        template< typename T >
            class Seg_VecDense : public detail::SegBase< Seg_VecDense<T>, T >
            {
            public:
                typedef detail::SegBase< Seg_VecDense<T>, T > Base;
                typedef ::dStorm::SegInfo SegInfo_t;

                /** \b required to various internal codes "look the same".
                 * MsgHeader has a client-specified \c HdrT<fmt> area within
                 * \c MsgHeader<fmt> that \c store \em must fill in (typically
                 * we'll write it during every \c store operation).
                 * \pre fmt is compile-time const.
                 *
                 * Also possible \em derive from \c seg::VecDense ... seems confusing :(
                 */
                typedef typename seg::VecDense<T> Fmt;

                /** accept "any" args to construct a segment placeholder class.
                 *
                 * Note that \c Dstorm::add_segment, which creates us, also passes
                 * us a C++11 arg-pack.  We **could** have a constructor with
                 * any set of args here, as long as user calls \c Dstorm::add_segment
                 * appropriately.
                 *
                 * For example here:<br>
                 * <code>
                 *   MY_VD_TYPE( float const my_magic_reduce_parameter,<br>
                 *               float const max_staleness_in_seconds<br>
                 *               // , etc, you get the idea<br>
                 *              )<br>
                 * </code>
                 * paired with (ex.):<br>
                 * <code>
                 *   namespace seg{
                 *       struct MY_VD{
                 *           typedef MY_VD_TYPE type;
                 *           typedef struct{ ... } Hdr; // for MY_VD's part of msg header
                 *           static signed char const value = +77;    // > 0
                 *           static constexpr char* name = "MY_VD";
                 *       };
                 *  }
                 *  template<> struct is_segment_format< seg::MY_VD >: public true_type {};
                 *
                 *   dstorm->add_segment<MY_VD><br>
                 *              ( ionet, cnt       // standard stuff<br>
                 *                3.14159f, 7.22f  // forwarded to My_VecDense constructor<br>
                 *              );
                 * </code>
                 * - Dstorm really doesn't deal in details of parsing any
                 *   particular seg::MY_VD message format.
                 * - \c MY_VD_TYPE provides the extra support for Dstorm::add_segment,store,reduce
                 * - \c MY_VD_TYPE should provide an interface similar to \c Seg_VecDense
                 */
                Seg_VecDense( SegInfo::SegPrintf segPrintf = nullptr );
                virtual ~Seg_VecDense();

                /** Initial construction of the segment. Now this segment
                 * handler is attached and accesible from a real dstorm object.
                 * User is responsible for setting up a small section of HdrT<fmt> (?)
                 * (still trying to shift away as much standard stuff as possible)
                 * NB: \c cnt is now set into SegInfo by parent constructor,
                 *     because it is a "generic" parameter.
                 */
                virtual void setSegInfo( );
                /** finalize construction of segment. SegInfo/SegInfoPod values "fully known" now. */
                virtual void add_segment( );
                void delete_segment();
                /** \pre \c HdrT<Internal> area of our MsgHeader<fmt> already set.
                 * Only we know details like Tdata, so we:
                 *
                 * - finalize MsgHeader<fmt> by initializing HdrT<fmt> area.
                 * - write the data area,
                 * - return count of items written to data area
                 *      - zero might be valid, if whole msg is in the header
                 * - return nullptr for an ignorable error (or throw)
                 *      - TODO beef up SegImpl handling
                 * - if all OK, return dataEnd (where to write MsgTrailer)
                 *
                 * \tparam IN_CONST_ITER convertible to our data type (T)
                 * \p cnt Copy this many T from \c iIter after advancing by
                 * \p offset items in the input stream.
                 * \p buf points to beginning of a buffer described by our SegInfo
                 * \p wgt optional multiplicative weight [default=1.0]
                 *        for Dstorm::reduce (used for push-sum style algorithms)
                 * \return \c nData count of items transfered and \c dataEnd
                 *         positioned to write \c MsgTrailer.
                 * \b NEW: \throw std::length_error if segment obuf capacity exceeded.        
                 */
                template< typename IN_CONST_ITER >
                    std::pair< uint32_t /*cnt*/, void* /*dataEnd*/ >
                    store( IN_CONST_ITER& iIter, uint32_t cnt,
                           uint32_t const offset, void* const buf,
                           double const wgt=1.0 );
                /** TODO, store should also have a begin, end version, without cnt,
                 *       since cnt may be expensive (exact value irrelevant once
                 *       we know it is "too big")
                 * \throw std::length_error if segment obuf capacity exceeded.        
                 */
                template< typename IN_CONST_ITER >
                    std::pair< uint32_t /*cnt*/, void* /*dataEnd*/ >
                    store( IN_CONST_ITER& iIter, IN_CONST_ITER const& iEnd,
                           uint32_t const offset, void* const buf,
                           double const wgt=1.0 )
                    {
                        throw std::exception(" store(begin,end,offset,buff[,wgt]) TBD?");
                    }

                /** push not required for VecDense, but SHOULD exist XXX */
                void push() {}

                /** reduce a list of recv bufnums into iBuf(). Parent \c SegImpl
                 * prunes out some (already-processed) receive buffers, passing
                 * us segment buffer numbers that might have reducable data.
                 *
                 * \ret number of reduce inputs that were averaged, <= rbufnums.size().
                 */
                uint32_t reduce( std::vector<uint32_t> const& rbufnums ) const;

            protected:
#if TORN_CHECKS
                mutable uint32_t nTorn;         ///< count of torn reductions, printed in destructor
#endif
            };
    }//user::
}//dStorm::
#endif // SEGVECDENSE_HPP
