/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef SEGINFO_HPP
#define SEGINFO_HPP

#include "dstorm_fwd.hpp"
#include <assert.h>
#include <utility>      // std::forward
#include <cstddef>      // size_t
#include <vector>       // vector<bool>  (should be some bitset?)

#include <type_traits>  // is_pod
#include <cstring>      // memset

#if WITH_GPU && defined(__CUDACC__)
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#endif

namespace dStorm {
//#define NOT_HERE throw std::runtime_error(std::string("unresolved SegInfo Vfunc ").append(__func__));
#define NOT_HERE assert(nullptr == "unresolved SegInfo Vfunc")

    namespace detail {
        class AwaitCountedSet;
    }//detail::

    /** suitable for cudaMemCpy etc. These data are "const" so they need to
     * be transferred/removed \b only during add_/delete_segment operations.
     */
    struct SegInfoPOD
    {
#if 0
        // original construction settings can be duplicated with memset to zero
        SegInfoPOD() :
            ionet(0U)           // kinda' meaningless with dstorm?
            , indegree_max(0U)
            , indegree(0U)
            , outdegree_max(0U)
            , outdegree(0U)
            , policy(SEGSYNC_NONE)
            , segNum(0U)
            , obuf(0U)
            , ibuf(0U)
            , rbuf(0U)
            , nbuf(0U)
            , bufBytes(0U)
            , segBytes(0U)
            , mem(nullptr)      // this could be a GPU pointer!
            , datacode(0U)
            , datasize(0U)
            , cnt(0U)
            , sizeofMsgHeader(0U)
            , seg_id(0U)
            , fmtValue(0) // 0 is reserved for seg::Illegal::value in dstorm_msg.hpp
            {}
#endif
        /// \name core layout parameters
        /// This new api makes the network communication model and buffer
        /// layout explicit for each segment.
        ///
        /// A large excess of variables have been removed because a segment
        /// is no longer responsible for storing \b ANY vector type.
        /// Instead, types are runtime-checked for exact match.
        /// (It is possible to relax the exact match check on a
        /// case-by-case basis.)
        ///
        /// - Hmmm. bufBytes and segBytes should maybe be uint32_t?
        /// - ASK changed bufBytes and segBytes to unsigned long /// 07/15/16
        /// - TODO: what can be private? what can be const?
        ///
        /// - Consider also storing (for consistency or sanity checks):
        ///   - \b NEW! FMT::value    (like a typeid for segment format)
        ///   - segNum      (or maybe segKey<FMT>(segNum) ?
        ///@{
        IoNet_t ionet;              ///< net graph --> \# send/recv buffers in segment
        /// \name ionet in/out-degree for this rank
        /// Sometimes useful to have handy, ex for notification id reservations
        //@{
        orm_notification_id_t indegree_max;   ///< indegree at add_segment time = recv list size
        orm_notification_id_t indegree;       ///< current indegree (lower than max by \# dead nodes)
        orm_notification_id_t outdegree_max;  ///< out degree at add_segment time = send list size
        orm_notification_id_t outdegree;      ///< current out degree (lower than max by \# dead nodes)
        //@}
        SegPolicy policy;       ///< how many segments of each type (oBuf for push, iBuf for reduce output, rBuf receive segments)
        // consistency flag ?
        SegNum segNum;          ///< Dstorm handle for this segment (assigned by user). \sa seg_id the orm handle
        uint32_t obuf;          ///< obuf is always buffer number 0 (short for SegInfo::oBufs[policy&SEG_LAYOUT_MASK])
        uint32_t ibuf;          ///< if present, obuf+1, else obuf (short for SegInfo::iBufs[policy&SEG_LAYOUT_MASK])
        uint32_t rbuf;          ///< if present, ibuf+1, else ibuf (short for SegInfo::rBufs[policy&SEG_LAYOUT_MASK])
        uint32_t nbuf;          ///< total \# buffers in segment (1 or more, \# of rbufs may be variable)
        unsigned long bufBytes; ///< sz of each buffer in segment (DSTORM_DATA_ALIGN)
        unsigned long segBytes; ///< == nbuf * bufBytes
        /** Unmanaged base mem pointer for entire segment.
         * - Sometimes this might have 4k alignment (for RDMA).
         * - Sometimes it might be GPU-memory allocated via CudaMalloc
         */
        orm_pointer_t mem;
        //uint_least32_t hdrsz;   ///< sizeof header area (unused) XXX CUDA MIGHT NEED THIS
        std::size_t datacode;   ///< rtti hash_code of buffer data item e.g. ~ typeid(float).hash_code()
        uint_least32_t datasize;///< sizeof each data item, ex sizeof(float)
        uint_least32_t cnt;     ///< max data items (ex how many float in a VecDense)
        uint_least32_t sizeofMsgHeader; ///< \c SegImpl<FMT> sets this to \c sizeof(MsgHeader<FMT>)
        orm_segment_id_t seg_id;  ///< number used by orm to refer to this segment
        uint_least8_t fmtValue;     ///< record the FMT::value of \ref dstorm_msg.hpp (ex. seg::VecDense<T>::value)
        //@}
        /** \name  buffer number functions */
        //@{
        __host__ __device__ uint32_t obufnum() const { /*assert(this->obuf == 0U)*/; return 0U; } ///< output buffer number
        __host__ __device__ uint32_t ibufnum() const { return ibuf; }			  ///< input buffer number (if absent, same as obuf)
        __host__ __device__ uint32_t rbufBeg() const { return rbuf; }			  ///< first receive buffer number
        __host__ __device__ uint32_t rbufEnd() const { return nbuf; }  
        __host__ __device__ void* ptrBuf(uint32_t bufNum) const		///< bufnum'th buffer address
        {
            if( !(bufNum < nbuf) ) {
#if defined(__CUDA_ARCH__)
                printf("ptrBuf: bufNum out of range\n");
                //
                asm("trap;");            // kill kernel with error //TODO: error no such instruction :"trap"
#else
                //dynamic_cast<SegInfo const*>(this)->segPrintf("ptrBuf: bufNum out of range\n");
                throw(std::runtime_error("bufNum out of range"));
                //exit(EXIT_FAILURE);
#endif
            }
            return (char*)mem + bufBytes * bufNum;
        }
        __host__ __device__ void* ptrObuf() const { return (char*)mem;} //assert((void*)mem == ptrBuf(obufnum()));  ///< output [store|push] buffer
        __host__ __device__ void* ptrIbuf() const { return ptrBuf(ibufnum()); }	///< input buffer (reduce output)
        __host__ __device__ void* ptrRbufBeg() const { return ptrBuf(rbufBeg()); }	///< first recv buf (a reduce input)
        __host__ __device__ void* ptrOdata() const { return (uint_least8_t*)this->ptrObuf() + this->sizeofMsgHeader; }
        __host__ __device__ void* ptrIdata() const { return (uint_least8_t*)this->ptrIbuf() + this->sizeofMsgHeader; }
        __host__ __device__ void* ptrData(uint32_t bufnum) const { return (uint_least8_t*)this->ptrBuf(bufnum) + this->sizeofMsgHeader; }
        // not callable from GPU. (Do policy error checking on host side, before entering kernel)
        //__device__ void policyError(char const* msg) const;
        //__device__ void policyError(std::string msg) const; 
        //@}
    };
    static_assert( std::is_pod< SegInfoPOD >::value,
                   "Error: SegInfoPOD **must** be plain old data" );

    /** Originally intended to be a POD, but now tainted with a vtable and pointers.
     * This is currently incompatible with being stored in shared memory, and used
     * by multiple threads or processes. */
    class SegInfo
        : public SegInfoPOD
    {
    public:
        friend class dStorm::Dstorm;
        typedef int (*SegPrintf)(const char* fmt, ...);


        /** default constructor creates object, but unusable until
         * Dstorm* d is initialized to non-null via
         * \c d->add_segment<SegTag<FMT>>.
         * throw if use before full init detected ?
         */
        SegInfo( char const* segVecName="Vec", SegPrintf segPrintf_=nullptr )
            : d(nullptr)       // begin UNCONNECTED to any Dstorm object
            , segVecName(segVecName)
            , reduceState(nullptr)      // non-NULL only for REDUCE_STREAM, set during add_segment
            , segPrintf( (segPrintf_ == nullptr) ?  ::printf : segPrintf_ )
            , valid(false)
            , userfunc(nullptr) // used for REDUCE_STREAM (which implies SEGSYNC_NOTIFY_ACK)
            {
                static_assert( SEGSYNC_NONE == 0U, "Ohoh, cannot use memset for SegInfoPOD" );
                memset( static_cast<SegInfoPOD*>(this), 0, sizeof(SegInfoPOD) );
            }

        /** should do nothing but...
         *
         * userfunc might be a std::bind object, which MUST be destroyed.
         * But now it is just a plain pointer to function (delete userfunc is not allowed).
         *
         * \todo make SegStreamFunc a std::function, which DOES get destroyed,
         * and use a trampoline if a plain "C" version is required for pthreads:
         * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
         * #include <functional>
         * #include <pthread.h>
         * 
         * extern "C" void* trampoline(void* userData) {
         *     return (*static_cast<std::function<void*()>*>(userData))();
         * }
         * 
         * void* start() {
         *     // ...
         *     return 0;
         * }
         * 
         * int main() {
         *     pthread_t thread;
         *     std::function<void*()> entry(start);
         *     pthread_create(&thread, 0, &trampoline, &entry);
         *     // ...
         * }
         * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
         * stackoverflow: "The immediate implication is, however, that the function object
         * life-time isn't easily controlled. In the example above the std::function<void*()>
         * object happens to live long enough but it isn't always as easy.
         */
        virtual ~SegInfo() {
            // TODO XXX
            // if( userfunc ){ delete userfunc; userfunc=nullptr; }
        }
        /** "covariant return type" up-casting trick to get the \em most-derived
         * SegImpl<FMT> version of SegInfo.  (used to be in SegBase)
         *
         * \return NULL for any class in chain that is \em not a SegImpl.
         * CHECK CAREFULLY that this works OK when storing "header info" for orm
         * segments (i.e. the original SegImpl always available.)
         *
         * this \b breaks the original intent to make this a POD type.
         * If so we can insert a template-less class between SegBase
         * and SegInfo whose sole purpose is to do the covariant-return trick.
         *
         * The + operators may be unnecessary.
         */
        virtual SegInfo& operator+() { return *this; }
        virtual SegInfo const& operator+() const { return *this; }
    protected:
        /** SegPolicy & SEG_LAYOUT_MASK selects layout of output, input,
         * and receive buffers in a segment.
         * \deprecated make these private to either SegImpl or Dstorm,
         *             because the member variables can override these values
         *             to elide un-needed segment buffers (ex. iBuf not needed)
         */
        // When these numbers increment (vertically, downwards), it means
        // the buffer exists in the segment.  The total number of buffers
        // is always from SegInfo::nbufs (the number of rBufs may vary)
        //-----------------------------------------------SEG_FULL
        //---------------------------------------------------SEG_ONE
        static constexpr uint32_t oBufs[SEG_LAYOUTS] = { 0U, 0U };
        static constexpr uint32_t iBufs[SEG_LAYOUTS] = { 1U, 0U };
        static constexpr uint32_t rBufs[SEG_LAYOUTS] = { 2U, 0U };

        /** should punt all the way up to SegImpl<T> */
        virtual void delete_segment() { NOT_HERE; }
        virtual uint32_t segimpl_reduce() const { NOT_HERE; return 0U/*zero incoming buffers reduced*/; }
    private:
        //virtual uint32_t gpu_reduce() const { NOT_HERE; return 0U; }


    public:
        /** \name vector implementation utils
         * SegBase provides some common utility functions to dStorm::user::
         * \e vector implementations. */
        //@{
        /** policy-based error handling for subview handling of reduce(or store) operations.
         * - Select whether to throw, print or ignore, according to this->policy.
         * - Default is to throw.
         * - dStorm::user::SegVecXXX user impls may decide to implement just
         *   a basic subset of \c policy features.
         * - Unimplemented features can get standard error handling. */
        void policyError(char const* msg) const;
        void policyError(std::string msg) const;       /// can also supply a std::string
        //@}
        /** \name notification id allocation
         * - \b rbufNotify*:
         *   - set to 1 when rBuf is filled. 
         *   - used to implement \c SEGSYNC_NOTIFY, internal to Dstorm
         * - \todo \b ackNotify*:
         *   - reply to sender that we have processed our rBuf.
         *     - For an XCH_AVG operations, where <B>oBufs are guaranteed
         *       equal when each node continues execution</B>.
         *       - Sender: write_notify oBuf --> iBuf, waitsome for ACK
         *       - Receiver: waitsome,ORM_TEST for iBuf, average into oBuf.
         *       - Receiver: write_notify oBuf --> oBuf back to sender
         *       - Sender: oBuf now has new content, execution continues.
         *
         * - These might be \b useless, \e except to demo how
         *   each rank should be allocating NIDs.
         * - We are more interested in NIDs of the rank that is sending-to-us,
         *   or receiving-from-us.
         *   - That machinery is in \c Dstorm::build_lists.
         */
        //@{
        /** rbufNotifyBeg()+r receipt means new buffer rbufBeg()+r has arrived */
        orm_notification_id_t rbufNotifyBeg() const { return 0U; }
        /** XXX must check that return value is within bounds (no wraparound error)! */
        orm_notification_id_t rbufNotifyEnd() const {
            SegPolicy const segsync = policy&SEGSYNC_MASK;
            return rbufNotifyBeg()
                + (segsync==SEGSYNC_NONE
                   ? orm_notification_id_t{0}                    // no rbuf notification ids
                   : indegree_max ); } // one notification per input edge
        /** rbufNotifyAckBeg()+i signals sendlist[i] was sent, received, and ACKed back to us. */
        orm_notification_id_t rbufNotifyAckBeg() const { return rbufNotifyEnd(); }
        /** XXX must check that return value is within bounds (no wraparound error)! */
        orm_notification_id_t rbufNotifyAckEnd() const {
            SegPolicy const segsync = policy&SEGSYNC_MASK;
            return rbufNotifyAckBeg()
                + (segsync==SEGSYNC_NOTIFY_ACK
                   ? outdegree_max                      // one ack per output edge
                   : orm_notification_t{0} ); }       // no acks required
        //@}
    public:
        SegInfo & operator=( SegInfo const& other ) = default;

        /** checked access to \c reduceState, throw on error.
         * This function is called only from Dstorm::streamCheckEof,
         * so it is seems like a CPU-only function, but ??? */
        detail::AwaitCountedSet& getReduceState() const;
        /** \name CPU-side extensions to SegInfoPOD data */
        //@{ 
    protected:
        dStorm::Dstorm *d;
        char const* const segVecName;
        detail::AwaitCountedSet * reduceState; ///< extension data for REDUCE_STREAM policy, else NULL
        SegPrintf segPrintf;

        /* Trick to avoid runtime type determination completely.
         * Seginfo does not know the type, but dstorm calls all take the
         * FMT metaclass which allows a void* to be "reinterpret_cast"
         * to the SegTag<FMT>::type* with zero ambiguity.
         *
         * NOTE: the upcast from SegBase is never needed.  Is SegBase needed?
         *       Maybe SegInfo is all we need.
         */
        //void * segimpl;
    public:
        bool valid;                 ///< add_segment=>true, delete_segement=>false
        /** REDUCE_STREAM can set a non-null pointer to be executed as writes individually arrive into  rbufs.
         *
         * \sa Dstorm::setStreamFunc( segId, userFunc ) This is \b NEW functionality.
         * This is incompatible with having a shered-memory SegInfo :(
         */
        SegStreamFunc userfunc;
        //@}
    };

    namespace detail {

        /** Maintain a counted set of items, \c ix, each of which should be \c set(ix) true
         * exactly once before \c all() returns true
         *
         * - all items \c 0..await-1 start false.
         * - you can \c set(ix) elements true (assert ix in range, throw if duplicate set(i)
         * - you can \c all() to see if all elements are there
         *
         * - We \b could expose:
         *   - \c reset( [new count] ) to all-zero
         *   - const accessor to awaitCnt, nDone
         *   - const accessor to \c awaitSet[i] 
         *   - \c any() true iff nDone >= 0
         * \todo replace vector<bool> with a fuller-featured dynamically-sized bitset.
         */
        class AwaitCountedSet {
        public:
            /** How many things in range [ 0, \c awaitCnt_ ) are we waiting for? */
            AwaitCountedSet( uint32_t awaitCnt_, char const* descr = nullptr )
                : awaitCnt(awaitCnt_), nDone(0U), descr(descr), done(awaitCnt_,false) {}
            /// \name basic functionality
            //@{
            /** register occurence of item \c i.
             * \pre i < awaitCnt
             * \throw runtime_error if duplicate set(i) : speedup by removing this check?
             */
            void set( uint32_t const ix );
            /** true iff every one of \c awaitCnt items has been \c set(i) true */
            bool all() const { return nDone >= awaitCnt; }
            /** return descriptor string for this set */
            std::string name() const { return std::string( descr? descr: "AwaitCountedSet" ); }
            //@}
            /// \name expanded functionality
            //@{
            /** return number of set(i) items */
            uint32_t nset() const {return nDone;}
            /** reset all items to a common state (default = false = initial state). */
#if WITH_GPU && defined(__CUDACC__)
            void reset( bool arrived=false ) { nDone=0U; thrust::fill(done.begin() , done.end() , arrived); }
#else
            void reset( bool arrived=false ) { nDone=0U; /*INEFFICIENT*/for(auto d: done) d=arrived; }
#endif
            //@}
        private:
            uint32_t awaitCnt;          ///< \# of things we're awaiting
            uint32_t nDone;             ///< \# of things we've \c set(i) so far
            char const* descr;          ///< optional description of items being waited for
#if WITH_GPU && defined(__CUDACC__)
            thrust::device_vector<bool> done;     ///< track things we've seen.
#else
            std::vector<bool> done;     ///< track things we've seen.
#endif
        };

        /** \class dStorm::detail::SegImpl
         * Generic high-level segment-buffer operations flow through
         * here, before going into client-specifiec implementation code.
         *
         * \tparam FMT is now a TYPE tag that represents a \b list of segments
         *             of that TYPE (e.g. VecDense, VecSparse, LeonSVector,...)
         *             There are few types, and multiple segments per type !
         *             \sa seg namespace, for sample FMT metaclasses.
         *
         * - \c FMT both for metaclass type lookup and also
         *    provides a value for consistency checks (maybe for conversions, later)
         *
         * 1. User invokes some \c Dstorm operations (store/push/reduce/...)
         * 1. \c SegImpl<FMT> does some segment-buffer preprocessing ops
         * 1. User supplies FMT metaclass pointing
         *    to user's Impl (ex. user::Seg_VecDense<float>) that
         *    responds to Dstorm store/push/reduce messages.
         *
         * - Flow:
         *   - Dstorm
         *   - operation<FMT>(args)
         *   - SegImpl<FMT>::operation
         *     -   isa   seg::FMT::type, for example user::Seg_VecDense<float>
         *   - FMT::type::operation(args')
         *     -   isa SegBase< Impl, TDATA(ex. float)
         *       - further common support functions
         *       - isa SegInfo (no template parms)
         *         - with ptr to SegImpl<FMT> object.
         *
         *\verbatim
         *   seg::FMT::type  --->  user::Seg_VecDense<float>
         *   (metaclass)               ^  = Impl |
         *__________|________ user  ___|_________|_________
         *       +->+        internal  |         |
         *      /    \                 |         |
         *     /      \---------is a---+         |
         *   SegImpl<FMT>::Impl=FMT::type        |
         *   (object class)                      |
         *                                       v
         *   /<----------------------------------+
         *   |
         *   isa SegBase< Impl, TDATA (i.e. float) >
         *   |
         *___|_______________internal______________________
         *   |                user
         *   |
         *   isa SegInfo: <--------- NO template
         *   isa SegInfoPOD <------- NO virtual functions "plain old data"
         *       Dstorm *dstorm: --> NO template, but members template<FMT> foo(...)
         *\endverbatim
         *
         * Navigation aid:
         *
         * <table><caption>Example, for T = float or double</caption>
         * <tr><th>::dStorm::FOO                        <th>"is"                                                        <th>Notes
         * <tr><td><em>dStorm::seg::</em>FMT<T>         <td><em>dStorm::seg::</em>VecDense<T>                           <td>\ref dstorm_msg.hpp
         * <tr><td>FMT<T>::type                         <td><em>dStorm::user::</em>Seg_VecDense<T>                      <td>\ref segVecDense.hpp
         * <tr><td rowspan="1"><b>SegImpl<FMT<T>></b>   <td>FMT<T>::type = Seg_VecDense<T>                              <td>\ref segImpl.hh
         * <tr><td> : Fmt -->                           <td> FMT<T>                                                     <td>
         * <tr><td> : Impl -->                          <td> Seg_VecDense<T>                                            <td>
         * <tr><td rowspan="3">Seg_VecDense<T>          <td><em>dStorm::detail::</em>SegBase<Seg_VecDense<T>, T>        <td>SegBase::Tdata is T
         * <tr>                                         <td>which is a <em>dStorm::</em>SegInfo                         <td>with virtual functions
         * <tr>                                         <td>which is a <em>dStorm::</em>SegInfoPOD                      <td>plain old data
         * <tr>SegBase<...>::Tdata                      <td>T                                                           <td>ex. float or double
         * <tr><td>sInfo=Dstorm:getSegInfo(SegNum)      <td>SegInfo const&                                              <td>\ref dstorm.hpp
         * <tr>(+sInfo) upcast                          <td><B>SegImpl<...></B>                                         <td>upcast "all the way"
         * <tr>??? (+sInfo)::Impl                       <td><em>not possible</em>                                       <td>but can catch error on \c dynamic_cast
         * <tr>but sInfo::delete_segment()              <td>invokes Seg_VecDense<T>::delete_segment()                   <td>because virtual..SegInfo:delete_segment()
         * <tr>and (+sInfo):data()
         * </table>
         *
         * - SegImpl<FMT> \em does use [orm] primitives, but all below
         *                should be independent of the the network transport
         *   - ?? user implementations, such as user::Seg_VecDense,
         *     take an optional \em pointer-to-printf argument. (ex. orm_printf)
         *
         * - Division of responsibilities changes a bit for GPU support:
         *   - now SegImpl<FMT> has a separate impl for seg::GpuVec::type = Seg_GpuVec
         */
        template<typename FMT>
            class SegImpl : public FMT::type /* e.g. SegVecGPU::type or SegVecDense::type */
        {
        public:
            typedef FMT Fmt; // to avoid a spurious nvcc warning during specialization?
            typedef typename FMT::type Impl;

            /** "export" all possible Impl constructors to this level */
            template<typename... Args>
                SegImpl( Args&&... args )               ///< \p args -->user's Impl constructor
                : Impl( std::forward<Args>(args)... ) {}
            virtual SegImpl&       operator+()       { return *this; }
            virtual SegImpl const& operator+() const { return *this; }
            /** add_segment takes a Dstorm* because the base SegInfo ptr has not been set yet.
             * (It sould still be null). \c policy_ is now non-const, since we may modify
             * the reduce policy to make it consistent with the segment layout. */
            void segimpl_add_segment( SegNum const s,   ///< which segment with FMT?
                              Dstorm* d,                ///< \p d SegImpl / SegInfo internal
                              IoNet_t const ionet,      ///< \p ionet SegImpl / SegInfo internal
                              SegPolicy policy,         ///< \p buffer layout, reduce operation, etc.
                              uint32_t const cnt        ///< \p cnt also internal, for ALL impl
                            );
            /** if impl cares about SegNum, we should stow the [runtime] value
             * into SegInfo at runtime (do not pass it to rest of SegImpl funcs) */
            virtual void delete_segment();
            /** store vector data.  if wgt=0.0 or 1.0 and &*iter is same as
             * segment data address, this becomes a no-op. Iter may have additional
             * restrictions on it from particular segment implmentations. */
            template< typename IN_CONST_ITER >
                void store( IN_CONST_ITER iter, uint32_t const cnt,
                            uint32_t const offset, double const wgt );
            /** NEW, version with begin,end, and throwing overflow_error
             * if output would not fit.
             *
             * - Included for sparse vectors, where \# of interesting outputs
             *   may be expensive to determine exactly).
             */
            template< typename IN_CONST_ITER >
                void store( IN_CONST_ITER iter, IN_CONST_ITER const end,
                            uint32_t const offset, double const wgt );
            // push ? perhaps add this "for consistent form"
            //         (it might be used in some SegTag<FMT>)
            uint32_t/*# iBufs reduced*/ segimpl_reduce() const;
        private:
#if 0
            /** segimpl_reduce() specialization for FMT ~ seg::VecGpu<T>.
             *
             * Each specialization of segimpl_reduce repeats the same code block,
             * </br> so making a separate function that works only for seg::VecGpu<T> types
             * </br> reduces code duplication. */
            uint32_t gpu_reduce() const;
            /** partial specialization helper for all FMT ~ seg::VecGpu<T> types */
            template< typename IN_CONST_ITER >
                void gpu_store( IN_CONST_ITER iter, uint32_t const cnt,
                                uint32_t const offset, double const wgt );
            //** for FMT ~ seg::VecGpu<T> only */
            //template< typename IN_CONST_ITER >
            //    void gpu_store( IN_CONST_ITER iter, IN_CONST_ITER const end,
            //                    uint32_t const offset, double const wgt );
#endif
        protected:
            virtual ~SegImpl() {} // cannot be public, because a derived class MUST store this->d pointer
        };

        /** utility funcs, mainly so I can keep SegInfo a POD type.
         * Expect that many user impls will need a core set of utility
         * functions to ease writing buffer i/o code.
         * \tparam IMPL is a SegImpl<FMT>::type.
         * \tparam TDATA is something like float or double, used to count "items" of "vectors" in each segment
         *
         * Typical user-defined "vector"s may wish to derive from
         * \c SegBase rather than directly from SegInfo.
         *
         * Note: CRTP used here, but likely \b not required.
         * \todo SegInfo base should be protected and available via a const getter?
         */
        template< typename IMPL, typename TDATA > class SegBase
            : public SegInfo // TODO should be protected with a const getter
            // The following is DESIRED, but creates a header requirement "loop"
            //: public IMPL::SegInfo_t // cpu or gpu version of structure
        {
        public:
            /// SegBase and higher know the actual datatype
            typedef TDATA Tdata;          // type of "raw data item"
            //typedef typename IMPL::SegInfo_t TSegInfo; ///< gpu/cpu version of SegInfo?
            typedef SegInfo TSegInfo; ///< gpu/cpu version of SegInfo?
            virtual SegBase & operator+() { return *this; }
            virtual SegBase const& operator+() const { return *this; }

            /** base address of \c bufNum'th buffer of this Segment.
             * I.e., past the MsgHeader array, where \c cnt TDATA items
             * could be stored. \deprecated consider emptying or removing SegBase! */
            TDATA* data(uint32_t bufNum) const;

        protected:
            virtual ~SegBase() {}
            /** constructible from derived dStorm::user::SegVecXXX <em>segment vector</em>. */
            SegBase( char const* segVecName="Vec", typename TSegInfo::SegPrintf segPrintf = nullptr );
        private:
        };

    }//detail::

#if 0 // WITH_GPU need full type of seg::VecGpu ... leave for segImpl.hh
    // declare existence of following specializations
    template<> inline uint32_t
        detail::SegImpl<dStorm::seg::VecGpu<double>>::segimpl_reduce() const;
    template<> inline uint32_t
        detail::SegImpl<dStorm::seg::VecGpu<double>>::segimpl_reduce() const;

    template<> template< typename IN_CONST_ITER > inline void
        detail::SegImpl<seg::VecGpu<float>>::store( IN_CONST_ITER iter,
                uint32_t const cnt, uint32_t const offset, double const wgt );
    template<> template< typename IN_CONST_ITER > inline void
        detail::SegImpl<seg::VecGpu<double>>::store( IN_CONST_ITER iter,
                uint32_t const cnt, uint32_t const offset, double const wgt );
#endif

#undef NOT_HERE
}//dStorm::
#endif // SEGINFO_HPP
