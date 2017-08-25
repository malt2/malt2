/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef DSTORM_MSG_HH
#define DSTORM_MSG_HH

#include "dstorm_msg.hpp"

#include <assert.h>
#include <type_traits>

namespace dStorm {

#if 0 // unused
    // --- HdrT<X> I/O ---
    template<typename MSGTYPE>
        struct HdrTio {
            typedef typename MSGTYPE::Hdr HDR;
            enum { sz = sizeof(HDR) };
            static void set( void* mem, HDR const& hdr ){
                *dStorm::mem_as<HDR*>( mem ) = hdr;
                //return ptr_inc(mem, sz);
            }
            static void get( void volatile const* mem, HDR& hdr ){
                hdr = *mem_as<HDR*>( mem );
                //*mem = ptr_inc( *mem );
            }
        };
#endif

    //
    // ------------------ MsgHeader<MsgFormats> ---------------------
    //
    inline constexpr size_t round_up( size_t const integer, size_t const multiple )
    {
        return (integer + (multiple-1U)) / multiple;
    }
#define ROUNDUP( INTEGER, MULT ) (((INTEGER) + (MULT-1)) / MULT * MULT)
    /** segment buffer format typically is (header, msg-info, msg-data, trailer),
     * and MsgHeader provides header + msg-info.
     *
     * - header and trailer are always HdrT<seg::Internal>.
     * - \b header gives you a msg sequence number and some type info for dispatching
     * - \b msg-info is some other HdrT telling how much msg-data there is
     * - \b msg-data is some array of POD or builtin type
     * - \b msg-trailer is auto-supplied and matches header for consistency check
     *      after a streaming read. \e msg-trailer is actually a MsgHeader<Dummy>
     *      so that it has nice aligned size, but empty \e msg-info (if paranoid,
     *      \e msg-info would provide a final checksum instead of Dummy).
     *
     * \post sizeof(MsgHeader) is a multiple of DSTORM_DATA_ALIGN,
     *       so that msg-data can always start at a SIMD-friendly address
     */
    template< typename Tfmt >
        struct MsgHeader
        {
            struct MsgHeader_nopad
            {
                typename seg::Internal::Hdr a;    ///< 'auto' header, set by \c store
                typename Tfmt         ::Hdr u;    ///< type-specific header
            }
            hdr;    ///< both auto(a) and user(u) headers

            // g++ workaround: force these to be compile-time const
            /** layour size of Internal header (>= sizeof(HdrT<Internal>) */
            enum { sz_int = offsetof(struct MsgHeader_nopad, u) };

            /** sizeof internal + msg HdrT together */    
            enum { sz_nopad = sizeof(hdr) };

            /** a nice target size for this structure */
            enum { sz = ROUNDUP( sz_nopad, DSTORM_DATA_ALIGN) }; 

            /** SIMD-friendly alignment.
             * MKL likes align(16) (or more for AVX?)
             */
            uint_least8_t pad[ sz - sz_nopad ];
        };
    /** Return struct with iter and fmt members.
     * To allow for stream-style access, user should read just once,
     * creating his own copy of things needing to be remembered.
     */
    inline  seg::Internal::Hdr
        getMsgHeaderInternal( void const* buf ){
            assert( buf != nullptr );
            return (reinterpret_cast<MsgHeader<seg::Dummy> const*>(buf)) ->hdr.a;
        }
    // NO ---- the trailer might not be all way at end !!!!
    inline seg::Internal::Hdr
        getMsgTrailer( void const* dataEnd ){
            assert( dataEnd != nullptr );
            //return mem_as<MsgHeader<Dummy>const *>(dataEnd)->hdr.a;
            return (reinterpret_cast<MsgHeader<seg::Dummy> const*>(dataEnd)) ->hdr.a;
        }
    // let's concentrate on write path.
    /** zero is all-OK */
    struct IoStatus { uint_least8_t err : 8;
        uint_least8_t warn : 8;
        IoStatus() : err(0U), warn(0U) {}
    };
#if 0
    /** MsgHeaderIo (also for MsgTrailer) should be
     * completely generic code, valid for all \c Tfmt
     *
     * Note: this "namespace" could be moved INTO MsgHeader.
     *
     * \deprecated Probably unused.
     */
    template< typename Tfmt, typename TDATA >
        struct MsgHeaderIo
        {
            typedef MsgHeader<Tfmt>     MSGHDR;
            typedef seg::Internal::Hdr  INTHDR;
            typedef Tfmt         ::Hdr  FMTHDR;
            static IoStatus set( void* mem, MSGHDR const& msghdr ) {
                IoStatus ret;
                // These are programmer errors... throw ?
                assert( msghdr.hdr.a.iter // never-decreasing 'iter'
                        > mem_as<MSGHDR*>(mem)->hdr.a.iter );
                assert( msghdr.hdr.a.fmt == Tfmt::value );      // sanity check

                *mem_as<MSGHDR*>( mem ) = msghdr;
                return ret;
            }
            static IoStatus get( void volatile const* mem, MSGHDR& msghdr ) {
                IoStatus ret;
                // INTHDR is at offset zero, always present
                HdrTio<seg::Internal>::get( mem, msghdr.hdr.a );
                if( msghdr.hdr.a.fmt != Tfmt::value ){
                    ++ret.err; return ret;
                }

                mem = ptr_inc( mem, MSGHDR::sz_int );
                HdrTio<Tfmt>::get( mem, msghdr.hdr.u );
                // further checks TBD before data reads:
                // - if previous iter known, check for recency
                //   (maybe someday have real timestamp?)
                // - calc. msg size doesn't overflow buffer size
                // - NEED compatibility with MsgTrailer
                //   both before and after data read.
            }
        };
#endif

}//dStorm::
#endif // DSTORM_MSG_HH

