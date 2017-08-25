/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef SEGTENSOR_HPP
#define SEGTENSOR_HPP

//#include "dstorm.hpp"
#include "dstorm.hh"            // need SegInfo and SegBase<> to be complete types.

namespace dStorm {

    namespace user {

        /** Segment storage for serialized 'Tensor' messages.
         *
         * \sa segTensor for implementation requirements.
         *
         * - This is a format for a \b serialized Tensor,
         *   so that it is ready to transmit/recv across a network
         *   - The serialization format is a contiguous view.
         *   - The \b maxview, set during construction, is fixed.
         *   - \b maxview sets the dimensions and strides representable
         *     for sent/received tensors.
         *     - reduce and push operations on segment buffers \b must
         *       touch only offsets addressable by the original \b maxview
         *     - If the \b maxview has max dimension and unit strides,
         *       any tensor view can be transmitted / received.
         *
         * - Alternative impl (NOT HERE)
         *   - add option to have NO iBuf for reduce operations.
         *   - punt all reduce operations to user-space
         */
        template< typename T >
            class Seg_Tensor : public detail::SegBase< Seg_Tensor<T>, T >
            {
            public:
                typedef int (*Printf_fn)(const char* fmt, ...);

                typedef detail::SegBase< Seg_Tensor<T>, T > Base;
                typedef SegInfo SegInfo_t;

                typedef typename seg::Tensor<T> Fmt;
                typedef typename Fmt::Uidx Uidx;
                //typedef typename Fmt::Size Size;
                typedef typename Fmt::Fint Fint;
                typedef typename Fmt::Idx  Idx;                ///< Fint[ ...MaxDimensionality ]
                typedef typename Fmt::value_type value_type;   ///< T
                typedef typename Fmt::Hdr Hdr;

                /** accept "any" args to construct a segment placeholder class.
                 *
                 * - We will accept any lua tensor, as long as the total number
                 *   of elements T fits into size allotted for data storage.
                 * - dim[] and inc[] and offset info may all change from 1 
                 *   store / reduce to the next.
                 */
                Seg_Tensor( Hdr const& maxview,
                            Printf_fn printf_fn = nullptr );
                virtual ~Seg_Tensor();

                /** default this to printf, but it could be the Dstorm::orm->printf
                 * pointer to use (eg) orm_printf or mpi printf or ??? */
                Printf_fn const printf_fn;

#if 0
                /** Finalize construction of the segment.
                 * - Sets up some of HdrT<FMS> (?).
                 * - \c cnt is now set into SegInfo by parent constructor.
                 * - \b must set SegInfo::datasize and datacode fields
                 */
                void setSegInfo( );
                void add_segment( );
                void delete_segment();
                /** \pre \c HdrT<Internal> area of our MsgHeader<fmt> already set.
                 * \tparam IN_CONST_ITER convertible to our data type (T)
                 * \p cnt Copy this many T from \c iIter after advancing by
                 * \p offset items in the input stream.
                 * \p buf points to beginning of a buffer described by our SegInfo
                 * \ret \c nData count of items transfered and \c dataEnd
                 *      positioned to write \c MsgTrailer.
                 *
                 * \sa Seg_VecDense<T>::store.
                 */
                template< typename IN_CONST_ITER >
                    std::pair< uint32_t /*cnt*/, void* /*dataEnd*/ >
                    store( IN_CONST_ITER& iIter, uint32_t cnt,
                           uint32_t const offset, void* const buf );
                /** TODO, store should also have a begin, end version, without cnt,
                 *       since cnt may be expensive (exact value irrelevant once
                 *       we know it is "too big")
                 */
                template< typename IN_CONST_ITER >
                    std::pair< uint32_t /*cnt*/, void* /*dataEnd*/ >
                    store( IN_CONST_ITER& iIter, IN_CONST_ITER const iEnd,
                           uint32_t const offset, void* const buf )
                    {
                        throw std::exception(" store(begin,end,offset,buff) TBD");
                    }

                /** push not required for Tensor, but SHOULD exist XXX */
                void push() {}

                /** reduce a list of recv bufnums into iBuf(). Parent \c SegImpl
                 * prunes out some (already-processed) receive buffers, passing
                 * us segment buffer numbers that might have reducable data.
                 *
                 * \ret true if not all data was able to be reduced (return
                 *      value not yet useful, and may be changed)
                 *
                 * - easy treatment:
                 *   - iBuf view (offset, dim[], inc[])
                 *     is a const \b maxview, set during add_segment.
                 *     - best if iBuf is a "full-ish" view of a model/gradient
                 *   - Not all Tensor inputs are \em compatible with each other
                 *     and our iBuf.
                 *     1. During add_segment, we set a \em maximally sized Tensor
                 *        \b maxview for this segment.
                 *     2. This \b maxview has its data stored contiguously.
                 *     3. Incoming data \em may have its own view, but all incoming
                 *        elements must also be present in the \b segview.
                 *        - i.e. incoming view \b must refer to a subset of the
                 *          Tensor elements of the \b maxview.
                 *   - example
                 *     - add_segment wishes \b maxview as offset=0, dim[4,4,4] and inc[2,2,2]
                 *     - incoming buffer with same view is \b OK
                 *     - incoming buffer with offset=2 dim[3,3,3] and inc[2,2,2] refers
                 *       to some subset of the \b maxview, so it's \b OK
                 *     - incoming buffer with offset=1 dim[4,4,4] and inc[2,2,2] has zero
                 *       overlsp with the \b maxview, so it's \b BAD
                 *     - incoming buffer with offset=2 dim[4,4,4] and inc[2,2,2] has one
                 *       element (the last one) that is not present in \b maxview, so \b BAD.
                 *   - If \b maxview is set to unit inc[] (compact, dense submatrix) this
                 *     may be less confusing :)J
                 *   - silently \b ignore any incompatible view, because it may just be the
                 *     result of a torn read (if 1-sided RDMA is used).  (o/w we could throw).
                 * - Alternate treatment:
                 *   - view of iBuf is non-const.
                 *   - determine the maximally compatible view with all input buffers
                 *   - throw if there's too much data to fit into iBuf
                 *   - \em write new dim[], inc[], offset into iBuf
                 *   - \em zero all the continguous data of iBuf
                 *   - \em add each input Tensor to iBuf.
                 */
                bool reduce( std::vector<uint32_t> const& rbufnums ) const;

#endif
            protected:
#if 0
                /** For deserialization (not too important for our contiguous view) */
                Size offset;
                Idx dim;      ///< dimensions, dim[0..ndim-1] > 0
                Idx inc;      ///< strides for each dimension
                Uidx ndim;    ///< 0 to MaxDimensionality
                Size size;    ///< 0 or product of ndim[i] for i=0..MaxDimensionality-1
                // Hmmm. These match struct seg::Tensor<T>::Hdr today.
#else
                typename Fmt::Hdr h;   // our data
#endif
            };
    }//user::
}//dStorm::
#endif // SEGVECDENSE_HPP
