/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef DSTORM_MSG_HPP_
#define DSTORM_MSG_HPP_

#include "dstorm_fwd.hpp"
#include <cstddef>              // ptrdiff_t
#include <type_traits>
#include <cassert>

#ifndef GCC_VERSION
#define GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#endif//GCC_VERSION
namespace dStorm {

    /** To assert proper segment message format types are provided
     * (and give nice error msg if not)
     *
     * Don't forget to override this for any additions you make
     * to namespace \c seg for you dStorm applications.
     *
     * Ex. <TT>template<> struct is_segment_format<seg::MyCompressedFloats>
     * : public true_type {}</TT>
     *
     * - Derive from \c true_type when the class provides:
     *   - \c type (concrete msg handler type, or void
     *   - \c Hdr  (user content for msg header, or empty struct)
     *   - \c value (for sanity/error checks, mostly, signed char)
     *   - \c name  (segment format c-string, like "VecDense"
     */
    template< typename Tfmt >
        struct is_segment_format     // by default, all are false
        : public std::false_type {};

    template<typename Tfmt>
        struct is_VecGpu
        : public std::false_type {};

    /** Extensible type-lookup metaclasses for dStorm message formats.
     *
     * - The main purpose is to provide short tags to specify the message
     *   format of a [set of] segments.
     *   - Ex:
     *     - using namespace dStorm::seg
     *     - dstorm.add_segment<VecDense<float>>( ... );
     *     - dstorm.add_segemnt<LeonSVector    >( ... );
     *
     * - You may think of \c seg::xxx as a way to specify multiple
     *   \em features with a single, easy-to-read template parameter.
     *
     * - \c seg also provides some metaclass helpers for internal use
     *     - parts of headers:  ex. \c Dummy, \c Internal
     *     - or markers:        ex. \c Illegal
     *
     * Originally this was done by enum MsgFormats.  This made code easy
     * to read <em>d.push<VECDENSE>(...)</em>.  Enums can't be extended
     * by the user. But namespaces can, and can be just as readable.
     *
     * The "value" fields of objects provided by default will all be <= 0
     *
     * In your project, feel free to add seg::xxx with xxx::value > 0.
     * Example:
     *\verbatim
     *  struct LeonSVector {
     *      typedef LeonSVector type;
     *      struct { ...whatever... } Hdr;
     *      static const uint_least8_t value = 17U; // some strictly +ve value
     *      static constexpr char const* name = "LeonSVector";
     *  };
     *\endverbatim
     * - Custom \c value fields have to be distinct within any single
     *   dStorm application (like Leon's asgd, milde_bp, senna, ... ).
     *   - it can serve as an rtti-like safety check for segment-buffer messages
     * - constexpr name initializers in header should be OK for gcc >= 4.7 or so.
     *
     * - \c value field MUST be uint_least8_t (nvcc warnings if int_least8_t)
     * - possibly use typed enum ???
     */
    namespace seg
    {
        //
        //------------------ "real" message formats ---------------
        //
        /** std type lookups \c type, \c Hdr, and a const marker \c value */
        template< typename T > struct VecDense {
            typedef ::dStorm::user::Seg_VecDense<T> type;
            typedef struct {
                uint_least32_t off;     ///< start offset (typically zero for whole vector)
                uint_least32_t cnt;     ///< how many values (maybe even all of them)?
                float wgt;              ///< a user-definable scalar, for PUSH-SUM style algorithms
                uint_least8_t  sz;      ///< sizeof(T), to double-check.
            } Hdr;
            static constexpr uint_least8_t value = 233;
            static constexpr char const* name = "VecDense";
        };
        /** Segment "format" type helper for SegIimpl<FMT> */
        template< typename T > struct VecGpu {
            typedef ::dStorm::user::Seg_VecGpu<T> type;
            typedef struct {
                uint_least32_t off;     ///< start offset (typically zero for whole vector)
                uint_least32_t cnt;     ///< how many values (maybe even all of them)?
                float wgt;              ///< a user-definable scalar, for PUSH-SUM style algorithms
                uint_least8_t  sz;      ///< sizeof(T), to double-check.
            } Hdr;
            static constexpr uint_least8_t value = 222;
            static constexpr char const* name = "VecGpu";
        };
        /** TBD */
        template< typename T > struct VecSparse {
            typedef user::Seg_VecSparse<T> type; // TBD
            typedef struct {
                uint_least32_t cnt; ///< maybe don't even need this?
                uint_least8_t sz;   ///< for safety checks
            } Hdr;
            static constexpr uint_least8_t value = 240; // XXX SHOULD also reflect "T"
            static constexpr char const* name = "VecSparse";
        };
#if GCC_VERSION > 50000 // avoid an ICE in gcc 5.0 - 5.2.1 -- tensor is NOT USED, so this is just to get things to compile
        template< typename T > struct Tensor {
            friend class ::dStorm::user::Seg_Tensor<T>;
            typedef typename ::dStorm::user::Seg_Tensor<T> type;
            typedef T value_type;
            static uint_least8_t const MaxDimensionality = 4;
            typedef uint_least32_t Uidx;
            typedef uint_least64_t Size;
            typedef uint_least32_t Fint;
            typedef Fint Idx[ MaxDimensionality ];
            typedef struct {
                Size offset;            ///< start offset (mostly for deserialization)
                Idx dim;                ///< unused terminal dims to 0
                Idx inc;                ///< strides for each dimension
                // calculable from dim[] (should crosscheck for net transmissions)
                Uidx ndim;              ///< \# of dimensions in dim[]
                Size size;              ///< how many T values, product of nonzero dims
                //uint_least8_t  sz;      ///< sizeof(T), to double-check.
            } Hdr;
            static constexpr uint_least8_t value = 250;      ///< C++11-ish feature, sanity-checking
            static constexpr char const* name = "Tensor";
        };
#else
        /** TBD dense tensor (no offset and increment for a "view") */
        template< typename T > struct Tensor {
            friend class ::dStorm::user::Seg_Tensor<T>;
            typedef typename ::dStorm::user::Seg_Tensor<T> type;
            typedef T value_type;
            static uint_least8_t const MaxDimensionality = 20;  // very loose sanity check :)
            typedef uint_least32_t Uidx;
            //typedef uint_least32_t Size;        // change to uint_least64_t non-trivial
            typedef uint_least32_t Fint;
            typedef Fint Idx[ MaxDimensionality ];
            typedef struct {
                uint_least32_t off;     ///< start offset (typically zero for whole vector)
                uint_least32_t cnt;     ///< how many values, total. \pre cnt=product of dims[0..dim-1].
                uint_least8_t  sz;      ///< sizeof(T), to double-check.
                // in addition to members "just like vector", a tensor has dimension information ...
                uint_least8_t  dim;     ///< \# tensor dimensions
                uint_least8_t  fmt;     ///< dimension encoding fmt
                uint_least8_t  pad[1];  ///< dims[] will begin at offset 12

                /** \c dims(fmt,i) returns the \c i'th item of \c dims in format \c fmt.
                 *
                 * C++11 unions can now provide a "dynamic dispatch"-like feature (you can add member
                 * functions to unions, with a bunch of restrictions). Doxygen seems to be confused
                 * by this. */
                union {
                    struct { uint_least32_t d[5];
#ifndef DOXYGEN
                        template<typename IDX> uint_least32_t operator[](IDX const i) const {
                            static_assert( std::is_integral<IDX>::value, "dim index must be an integer" );
                            assert( static_cast<typename std::make_unsigned<IDX>::type>( i )  < 5U);
                            return d[i]; }
#endif
                    } dim0;
                    struct { uint_least16_t d[10];
#ifndef DOXYGEN
                        template<typename IDX> uint_least32_t operator[](IDX const i) const {
                            static_assert( std::is_integral<IDX>::value, "dim index must be an integer" );
                            assert( static_cast<typename std::make_unsigned<IDX>::type>( i )  < 10U);
                            return d[i]; }
#endif
                    } dim1;
                    struct { uint_least8_t  d[20];
#ifndef DOXYGEN
                        template<typename IDX> uint_least32_t operator[](IDX const i) const {
                            static_assert( std::is_integral<IDX>::value, "dim index must be an integer" );
                            assert( static_cast<typename std::make_unsigned<IDX>::type>( i )  < 20U);
                            return d[i]; }
#endif
                    } dim2;
                    struct { uint_least8_t  varenc[20]; ///< TBD, if ever one of above fmts aren't good enough.
#ifndef DOXYGEN
                        template<typename IDX> uint_least32_t operator[](IDX const i) const {
                            assert( false );
                            return 0U; }
#endif
                    } dim3;
#ifndef DOXYGEN
                    /** \c dims(fmt,i) returns the \c i'th item of \c dims in format \c fmt */
                    uint_least32_t operator()(uint_least8_t fmt, uint_least32_t idx) const {
                        uint_least32_t ret=0U;
                        switch(fmt){
                          case(0): ret = dim0[idx]; break;
                          case(1): ret = dim1[idx]; break;
                          case(2): ret = dim2[idx]; break;
                          default: ;
                        }
                        return ret; }
#endif
                } dims;
                static_assert( sizeof(dims) == 20, " sizeof(fancy C++11 union) should have been 20 " );
            } Hdr;
            static_assert( sizeof(Hdr) == 32U, "Ohoh, tensor hdr was not exactly 32 bytes long" );
            static constexpr uint_least8_t value = 250;      ///< C++11-ish feature
            static constexpr char const* name = "VecTensor";
        };
#endif
        //
        //------------------ markers or parts of messages -----------------
        //
        /** Provide a value that mark something as "illegal" (used, uninitialized, etc) */
        struct Illegal {
            typedef void type;            ///< it is not a whole-message format
            typedef struct { /* empty*/ } Hdr;  ///< also don't care
            static constexpr uint_least8_t value = 0;        ///< special marker value
            static constexpr char const* name = "Illegal";
        };
        /** When don't care (or know) about user HdrT, use a Dummy placeholder */
        struct Dummy {
            typedef void type;  ///< it is not a whole-message format
            typedef struct { /* empty*/ } Hdr;
            static constexpr uint_least8_t value = 255;
            static constexpr char const* name = "Dummy";
        };
        /** values initialized during Dstorm::push, copied to MsgTrailer
         * for verification purposes, and modifed (pushes) to mark the
         * whole message as "sent" */
        struct Internal {
            typedef void type;          ///< just the Hdr struct is significant
            typedef struct {
                /** every 'store' operation increments this counter */
                uint_least64_t iter;

                uint_least32_t bytes; ///< total msg bytes for header + data + trailer

                /** type of ensuing message (safety, conversions?),
                 * reduce sets it to Illegal::value to mark \em reduced.
                 */
                uint_least8_t fmt;

                /** multiple uses:
                 * - oBuf: zero for store/push, counter >0 \em after push()
                 */
                uint_least8_t  pushes;
            } Hdr;
            static constexpr uint_least8_t value = 254;   ///< probably not used
            static constexpr char const* name = "Internal";
        };
        /** just for show (TBD) uninterpreted string
         * ... or entire msg in just the header ... or ... */
        struct Cstring {
            typedef void type; // TBD
            typedef struct{ /* TBD */ } Hdr;
            static constexpr uint_least8_t value = 253;   ///< TBD
            static constexpr char const* name = "Cstring";
        };
    }

    /** Custom extensions also need to provide a traits class.
     * This returns true for seg::FMT::type begin a non-void
     * "full" message implementation. */
    template<>
        struct is_segment_format< seg::Cstring >
        : public std::true_type {};
    //template<>
        template< typename T >
        struct is_segment_format< typename seg::VecDense<T> >
        : public std::true_type {};
    //template<>
        template< typename T >
        struct is_segment_format< typename seg::Tensor<T> >
        : public std::true_type {};
    //template<>
        template< typename T >
        struct is_segment_format< typename seg::VecGpu<T> >
        : public std::true_type {};

    //NO! static_assert( is_segment_format< Cstring >::value, "OHOH" );
    static_assert( is_segment_format< seg::Cstring >::value, "OHOH" );
    static_assert( is_segment_format< seg::VecDense<float> >::value, "OHOH" );
    static_assert( is_segment_format< seg::VecDense<double> >::value, "OHOH" );
    static_assert( is_segment_format< seg::VecDense<char> >::value, "OHOH" );
    static_assert( is_segment_format< seg::VecGpu<float> >::value, "OHOH" );
    //
    // ... and in a particular application, you might provide something
    // like ...
    //static_assert( is_segment_format< seg::LeonSVector >::value, "OHOH" );
    //                                  ^^^^^^^^^^^^^^^^ user-supplied

    //
    // now declare which seg::FMT types satisfy is_VecGpu
    //
    template<typename T>
        struct is_VecGpu< seg::VecGpu<T> >
        : public std::true_type {};

    static_assert( is_VecGpu<seg::VecGpu<float>>::value, "ERROR: is_VecGpu issues");
    static_assert( is_VecGpu<seg::VecGpu<double>>::value, "ERROR: is_VecGpu issues");
    static_assert( is_VecGpu<seg::VecDense<float>>::value == false, "ERROR: is_VecGpu issues");

#if 0
    /** In general, we have an asynchronous, 'best effort' messaging system.
     * MsgFormats is used to allow different message types, and to allow some
     * templated type lookups.
     *
     * We want to support things other than dense vectors, and allow a developer
     * to send custom messages without too much work (actually it might be better
     * to expose the ethernet control plane of ORM transport for such things, but this
     * is not exported by most transports).
     *
     * A message may have several parts.
     *   \e Ex. internal header, message header, message content.
     */
    enum MsgFormats : uint_least8_t {
        Dummy = 0U              ///< internal usage (also can be used as a MsgTrailer)
            , Illegal           ///< mark msg as processed/available
            , Internal          ///< header area set by every \c store operation
            // --- user types ---
            , VecDense          ///< first one we'll support
            , VecSparse         ///< list of <offset,value> pairs
            //, VecSparse2        ///< alternate sparse format
            //, VecCompress       ///< dense compressed vector
            , Custom            ///< \e ie uninterpreted bytes
            , LeonFVector       ///< For Leon's ASGD FVector type (TBD)
            , LeonSVector       ///< For Leon's ASGD SVector type (WIP)
    };
#endif

    /** backwards-compatibility \deprecated */
    template<typename MSGTYPE> struct HdrT;

    /** segment buffer format is (HdrT<Internal>, HdrT<MSGTYPE>, MsgData, MsgTrailer),
     * and \c MsgHeader <EM>provides nice alignment</em>
     * for \e pre-MsgData and \e post-MsgData sections. */
    template<typename MSGTYPE> struct MsgHeader;

    /** putting a copy of HdrT<Internal> at the end of the message
     * allows a quick-n-dirty check of consistency: is hdr.a.iter
     * the same at the end of some streaming read as it was at the
     * beginning?   Database apps may go further for 1-sided I/O
     * by calculating a checksum (overkill for us).
     */
    typedef MsgHeader<seg::Dummy> MsgTrailer;

    //
    // ------------------ HdrT<MsgFormats> ---------------------
    //
    template<typename MSGTYPE> struct HdrT
    {
        typedef typename MSGTYPE::Hdr type;
    };

    /** casting utils (less typing) -- perhaps should be T* mem_as ?? */
    template<typename T> __host__ __device__ inline
        T mem_as(void* mem) {
            return reinterpret_cast<T>(mem);
        }
    template<typename T> __host__ __device__ inline
        T mem_as(void const* mem){
            return reinterpret_cast<T const>(mem);
        }
    /** casting util */
    template<typename T> __host__ __device__ inline
        T mem_as(void *mem, ptrdiff_t bytes ){
            return reinterpret_cast<T>((char*)mem + bytes);
        }
    __host__ __device__ inline
        void* ptr_inc( void* mem, ptrdiff_t bytes ){
            return reinterpret_cast<void*>((char*)(mem) + bytes );
        }
    __host__ __device__ inline
        void const* ptr_inc( void const*mem, ptrdiff_t bytes ){
            return reinterpret_cast<void const*>((char const*)(mem) + bytes );
        }

    //}//msg::
}//dStorm::
#endif // DSTORM_MSG_HPP_
