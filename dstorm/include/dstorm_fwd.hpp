/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef DSTORM_FWD_HPP
#define DSTORM_FWD_HPP

/** new features related to unification of GPU/CPU code.
 * 1 = use SegInfoPOD
 * 2 = remove support for SegInfo.cuh*
 */
#define SEGINFO_MOD 2

#include "detail/dstorm_common.h"       // common macros and a few simple types.
#include <functional>
#include <exception>

#include "orm_fwd.h"                    // match our transport constants to liborm defns
#if WITH_LIBORM
#include "orm.h"                        // struct Orm dispatch table to shm/...
#endif

#include "dstorm_gpu.hpp"

/** <B>D</b>i<B>ST</b>ributed <B>O</B>bject <B>R</b>emote <B>M</b>emory */
namespace dStorm {

    /** user-extension objects/code.
     *
     * - To implement \em user-supplied extensions, DStorm uses a combination
     *   of generic and extension objects.
     * - Resulting dstorm class hierarchies can be quite deep
     * - This can be confusing.
     * - To make things clearer, examples of code/objects \em intended to be
     *   adapted for user-defined extensions will \em slowly be moved into
     *   this namespace.
     *
     */
    namespace user
    {
        /** builtin example implementations */
        template< typename T > class Seg_VecGpu;
        template< typename T > class Seg_VecDense;
        template< typename T > class Seg_VecSparse;     ///< TBD
        template< typename T > class Seg_Tensor;        ///< TBD
    }

    // declared in dstorm_msg.hpp
    namespace seg
    {
        template< typename T > struct VecDense;
        template< typename T > struct VecGpu;
        template< typename T > struct VecSparse;
        template< typename T > struct VecTensor;
        struct Illegal;
        struct Dummy;
        struct Internal;
        struct Cstring;
    }//seg::
    template< typename SegFmt > struct is_segment_format;
    template< typename SegFmt > struct is_VecGpu;

    /** Dstorm segments are referred to by handles of type \c SegNum.
     * Handles are unique tag values supplied by the user.
     *
     * - \c SegNum says which segment we are operating on.
     * - Probably only need a couple (< 100) Dstorm
     *   segments of any given seg::type.
     */
    typedef uint_least16_t SegNum;

    /** type of Dstorm::iographs index.
     * Builtin tag values of type \c OldIoNetEnum are reserved values,
     * while user-defined i/o nets get assigned IoNet_t > IONET_MAX
     * within the \c ScalGraphs container of \c Dstorm. */
    typedef uint_least8_t IoNet_t;

    /** Dstorm manages I/O between various segments, supplying
     * asynchronous mapping functions (\c store, \c push) and
     * \c reduce functions.
     *
     * - Segment message types are predetermined at compile time.
     * - Segment types are looked up by \c SEGS compile-time integer
     * - \c SegTag<WHICH> (for WHICH in SEGS...) maps WHICH to user impls
     *   - ex SegTag<0>::type is user::Seg_VecDense<float>
     *   - impls define store/reduce operations and are allowed to
     *     read/write custom message headers in whatever format they wish.
     * 1. dstorm constructor set ionet things. i/o NOT initialized yet
     * 1. add_segment<> initializes orm for each segment type needed.
     *    - this instantiates real implementation objects (impl=user::Seg_VecDense<float>)
     *    - whose SegInfo* components point back to the impl with a void* ptr
     *    - the void* ptr can safely be upcast, since all API funcs are
     *      templated on <WHICH> and fully know the actual type of the void*
     *    - work split between classes so user-defined class has
     *      very little responsibility, and heavy lifting done by internals.
     */
    class Dstorm;

    /** This defines a communication pattern for each segment.
     *
     * \b CHANGE: OldIoNetEnum is now a \em tag for \b builtin io nets.
     * 
     * - The user can dynamically register arbitrary communication graphs via
     *   - via \c Dstorm::add_ionet(...)
     *   - --> \c ScalGraphs::push_back( std::unique_ptr<mm2::UserIoNet>&& ptr )
     *
     * \b FLAW: \em IoNet and segment \em SegLayout are orthogonal
     * concepts. This description muddles them, since <em>segment layout</em>
     * as a concept is not supported.  The default layout supports
     * store/push/reduce with merging OK for asgd algs, but different
     * segment buffer layouts would support e.g. HogWild, NMF, ... .
     *
     * <em>Segment layout</em> is influenced by \c SegPolicy, while
     * the <em>communication graph</em> is given by \c IoNet_t.
     *
     * A segment represents several buffers of similar size and purpose.
     * Simple applications may use a segment to send/receive some type
     * of \em vector. Output buffers are connected to other hosts.
     * Input buffers store incoming data for "reduce" operations.
     *
     * Current communication patterns allot 1 buffer per incoming
     * edge of the I/O graph.  Operating asynchronously, this means
     * that a machine may miss some inputs from a fast host.
     *
     * TBD: The memory of one of the segments input buffers (and often
     * an output buffer too) may be tied tightly to a user's own vector,
     * to avoid a memcpy.
     */
    typedef enum OldIoNetENUM : unsigned short {
        ALL = 0, SELF = 1, CHORD = 2, HALTON = 3,
        RANDOM = 4,             ///< currently not supported
        PARA_SERVER = 5,        ///< TO_WORKER and TO_SERVER removed
	STREAM = 6,
        BUTTERFLY = 7,          ///< currently not supported
        IONET_MAX = 8           ///< always last
    } OldIoNetEnum;


    /** Various ways to specifiy consistency of \e iter message counters when
     * reducing external inputs.  Support lacking in milde_malt2. */
    typedef enum ConsistencyENUM : unsigned short
    {
        BSP = 0,                ///< bulk synchronous (lots of barriers)
        ASYNC = 1,              ///< use any input messages (less barriers)
        SSP = 2                 ///< bounded staleness (iters within 4 or ?)
    } ConsistencyENUM;

    /** Dstorm must be constructed on top of at least one transport. */
    typedef enum TransportEnums {
          OMPI = ORM_OMPI
        , GPU = ORM_GPU
#if WITH_SHM
        , SHM = ORM_SHM
#endif
    } TransportEnum;

    typedef enum NotifyEnums : uint32_t
    {
        NTF_RUNNING=7           ///< NOTIFY_ACK: from stream writer, reply is NTF_ACK
            , NTF_ACK=9         ///< NOTIFY_ACK: after dest ( \sa SAFER_ACK in dstorm.hh )
            , NTF_DONE=13       ///< REDUCE_STREAM: eof from stream writer, and ack from reader
            , NTF_SELECT=5      ///< \b NEW \c Dstorm::push(SegNum,NTF_SELECT,sendlist_index) sends obuf out a single sendlist edge
    } NotifyEnum;
    static_assert( sizeof(uint32_t) == sizeof(orm_notification_t), "NotifyEnum type error" );

    typedef unsigned short SegPolicy;

    /** Segments now have "policies" for layout, reduce operation, (etc),
     * and policies are xored as an argument to Dstorm::add_segment(..)
     *
     * Not all combinations are valid: Ex. SEG_ONE might throw unless you
     * specify REDUCE_NOP to silently ignore any calls to reduce.
     *
     * \todo REDUCE_AVG_RBUF_OBUF should allow a new SEG_LAYOUT_OBUF_RBUF
     *       because the iBuf is unneeded.
     *
     * - Timing measurements
     *   - localhost TCP network.
     *   - display rank (of 8) with lowest % for rbufs.size() == 3
     *   - dstsgd4 run parms: a=async or s=sync, N=notify or n=None
     *   - ** -aN is not yet supported - will hang, fixable with SEGSYNC_NOTIFY_ACK ** WIP.
     *   - gradient measurements were with 3 machines (and ALL)
     *   - -sKH SEGSYNC_NOTIFY_ACK test was with dstsgd5.sh (and dstorm mods) so not perfectly comparable
     *   - -aKH might hang.  But sometimes it ran to completion.
     * 
     * Measurement      | -snH    | -sNH    | -sKH   | -anH   | -aKH  | -gsn  | -gsN  | -gan  |
     * ---------------: | :------ | :------ | :----- | :----- | :---- | :---- | :---- | :---- |
     * Wall Time        | 28.1 s  | 27.8 s  | 28.6 s | 20.7 s | 25.8  | 15.3  | 12.2  | 12.2  |
     * Total            | 12.6    | 12.0    | 11.6   | 5.64   | 9.9   | 5.33  | 4.32  | 3.90  |
     * Barrier          | 7.45    | 6.88    | 3.6    | 0      | 0 (*) | 0.78  | 0.23  | 0     |
     * Loss             | 0.1445  | 0.1447  | 0.1448 | 0.1499 | .1443 | .1445 | .1446 | .1457 |
     * rbufs.size()==0  | 0   %   | 0    %  | 0      | 1.4  % | 0.3   |       |       |       |
     * rbufs.size()==1  | 0       | 0       | 0      | 10.8   | 2.3   |       |       |       |
     * rbufs.size()==2  | 0.6     | 0       | 0      | 40.1   | 17.4  |       |       |       |
     * rbufs.size()==3  | 99.4    | 100     | 100    | 47.8   | 80    |       |       |       |
     * 
     * (*) SegSenderAcks::ackTicks has some measure of 'barrier', but not printed yet.
     *
     * <B> For synchronous, it is good to replace the push/reduce barrier with
     *     SEGSYNC_NOTIFY </B>, particularly if I/O is heavy.
     */
    typedef enum SegPolicyENUM : SegPolicy
    {
        /// policy field masks.
        /// default SegPolicy "0" is SEG_FULL | REDUCE_AVG_RBUF | RBUF_SUBVIEW_NONE | SUBVIEW_ERR_THROW
        //@{
        SEG_LAYOUT_MASK = 0x7,          ///< 3 bits for segment layout
        REDUCE_OP_MASK = 0x7 << 3,      ///< 3 bits for reduce behavior
        RBUF_SUBVIEW_MASK = 0x7 << 6,   ///< 2 bits for desired subview support
        SUBVIEW_ERR_MASK = 0x3 << 9,    ///< 2 bits for subview error behavior
        SEGSYNC_MASK = 0x3 << 11,       ///< 2 bits to require notify/wait-all push/reduce
        //@}

        /// Segment layout: what oBuf, iBuf, rBufs are needed?
        //@{
        /// oBuf, iBuf, rcv bufs all exist as distinct segments, with the
        /// number of rcv bufs for each rank determined by the IoNet graph
        SEG_FULL = 0 /*default*/,
        /// only 1 segment (oBuf), that can be used as a token ring by 
        /// monitoring the MsgHeader \c iter to detect ownership
        SEG_ONE = 1,
        /// number of supported layouts (not itself a valid buffer layout)
        SEG_LAYOUTS = 2,
        // For REDUCE_SUBVIEWS, oBuf and iBuf buffers are absent because
        // they will be of some larger size than the receive buffers.
        // SEG_RBUFS_ONLY = 2,
        //@}

        /// how Dstorm::reduce(..) behaves
        //@{
        /// reduce forms average of rBufs (receive buffers), into iBuf.
        /// This was the only one originally implemented. iBuf may be
        /// a smaller-sized sub-view if rBufs are sub-views.
        REDUCE_AVG_RBUF = 0 << 3 /*default*/,
        /// reduce forms average of rBufs AND oBuf, into oBuf.
        /// So far, this is the most popular operation, it seems.
        /// oBuf always has the full SegInfo::cnt number of data items.
        ///    ** should be the new default **
        REDUCE_AVG_RBUF_OBUF = 1 << 3,
        /// reduce forms sum of rBufs, into iBuf (might require full-sized reduction)
        /// (Maybe for very sparse rBufs and good rBuf subviews support, this
        ///  setting should ignore if rBufs ever overlap? pehaps reduce for this
        ///  should have a scale parameter for the rBufs (which could be gradients))
        REDUCE_SUM_RBUF = 2 << 3,
        /// reduce operations are silently ignored, always return nReduce=0
        REDUCE_NOP = 3 << 3,
        /// reduce each rbuf individually by invoking a user-definable handler function.
        /// This REQUIRES SEGSYNC_NOTIFY_ACK to work properly (which requires orm).
        /// Dstorm::reduce operates in <em>polled mode</em> for this reduction,
        /// which (for any available rbufs) calls a handler function installed
        /// by Dstorm::setStreamFunc.
        /// For politeness, tight REDUCE_STREAM polling loops should sched_yield().
        REDUCE_STREAM = 4 << 3,
        // reduce sets user buffer to avg of rBufs
        //REDUCE_AVG_SET_USER
        // reduce averages user buffer with rBufs
        //REDUCE_AVG_MOD_USER
        //@}

        /// Required level of subview support for reductions.
        /// User implementations can vary in how much they support, and
        /// should throw or warn about unsupported subview reductions.
        //@{
        /// rBuf subviews not supported at all, all rBufs must full-sized.
        /// NB: restrictive -- you are not allowed to reserve 100 floats
        /// and just push 2 floats!
        RBUF_SUBVIEW_NONE = 0 << 6 /*default*/,
        /// rBuf homogeneous subviews are desired.
        /// All rBufs must agree, for example, on offset and cnt of data items.
        /// (producing an average by averaging a constant number of things)
        RBUF_SUBVIEW_HOMOG = 1 << 6,
        // ----------- the next ones seems useful, easy to code, but not yet implemented
        /// rBufs may contain homogenous or non-overlapping sub-views
        RBUF_SUBVIEW_HOMOG_OR_NONOVLP = 2 << 6,
        /// rBufs may be homogenous, and overlap of sub-views is ignored.
        /// Processing loop still reduces each rBuf individually as if it were unique
        /// Overlap regions might yield "slightly wrong" values, but this is OK.
        RBUF_SUBVIEW_OVLP_RELAXED = 3 << 6,
        // ------------ the most general one, strictly correct, is somewhat painful to code.
        /// rBuf reduction support for arbitrary overlaps is required to
        /// be strictly correct (nReduce > 1 for every ovlp region).
        /// Harder to implement! Need to precalulate the nReduce change points
        /// and break apart the reduce loop!
        RBUF_SUBVIEW_ANY = 4 << 6,
        //@}

        /// User implementations are allowed to vary in what subview
        /// reductions are coded, so errors need to be handled somehow.
        ///
        /// For example, Seg_VecDense supports only up to RBUF_SUBVIEW_HOMOG,
        /// simply because the other cases require more code and have not
        /// yet been needed.
        ///
        /// post-reduce torn read checks? nobody has implemented them.
        //@{
        SUBVIEW_ERR_THROW = 0 << 9 /*default*/, ///< subview errors throw
        SUBVIEW_ERR_WARN = 1 << 9,              ///< warn and return nReduce=0
        SUBVIEW_ERR_IGNORE = 2 << 9,            ///< always ignore and return nReduce=0
        //SUBVIEW_ERR_IGNORE_OVLP = 3 << 8,       ///< like _IGNORE, but if non-homog subview overlaps occur, ignore, continue reduction, return nReduce=1
        // note: non-homog reduce analyzes ovlp pattern, and if any ovlp uses _IGNORE_OVLP to decide whether to continue.
        //     - RBUF_SUBVIEW_ANY means extra work MUST be done to use the proper nReduce value for each overlap region.
        //     - RBUF_SUBVIEW_HOMOG_OR_NONOVLP reduce loop goes through rbis individually, each time using nReduce of 1.
        //       - Above behavior can be forced even if overlaps occur, using SUBVIEW_ERR_IGNORE_OVLP.
        //       - SUBVIEW_ERR_IGNORE_OVLP, on other types of error, 
        //@}

        /// Originally SEGSYNC_NONE meant user code would explicitly insert
        /// if(sync)d.barrier() exlicit call to a costly global \c Dstorm::barrier().
        ///
        /// Now, some of the .cpp tests forget the if(sync)d.barrier() so you ask for
        /// "-s" (sync) on command line and it is actually async !!
        ///
        /// \todo add a SEGSYNC_BARRIER that forces a full barrier before reduce
        ///
        /// - SEGSYNC_NOTIFY will use orm_write_notify during push, matched
        ///   with orm_waitsome (on entire set of iBufs) during reduce.
        ///   - Old t_barrier timings would be folded in with t_reduce, I suppose.
        ///   - Total 'sync' time may be reduced,
        ///   - and IO/memory bandwidth spikes may be reduced.
        ///   - SEGSYNC_NOTIFY \b should use a \c reduce() timeout (but doesn't yet)
        ///   - SEGSYNC_NOTIFY \b will be active even for REDUCE_NOP
        ///   - SEGSYNC_NOTIFY \b may be implemented as an full barrier,
        ///     if write_notify is not available (i.e. WITH_LIBORM)
        ///   - still susceptible to reducing mixed-version vectors, but less often than SEGSYNC_NONE
        ///
        /// - SEGSYNC_NOTIFY_ACK (\b new)
        ///   - reduces mixed-version issues by sending an ACK that, <em>at least</em>,
        ///     means "You can modify your obuf"
        ///   - Now, acks are sent \b after the reduce operation always,
        ///     which can give longer delay, but stronger semantics:
        ///     - ACK => "Now you can send new data to me"
        ///     - This protocol should be free of mixed-version issues.
        //@{
        SEGSYNC_NONE = 0 << 11,         ///< user can still call orm_barrier to do global sync.
        SEGSYNC_NOTIFY = 1 << 11,       ///< write_notify/notify_waitsome if possible for send/reduce
        SEGSYNC_NOTIFY_ACK = 2 << 11,   ///< src:write_notify --> dest:notify_waitsome && ACK --> src
        //@}

        SEGPOLICY_MAX = SEGSYNC_NOTIFY_ACK + (SEGSYNC_NOTIFY_ACK-1)  ///< for crude range check
    } SegPolicyENUM;

    /** \typedef SegPolicy
     * SegPolicyENUM values are xored together to describe Dstorm segment behavior.
     * While a struct might be nicer, it is kept as a POD type intentionally (for shm).
     * \sa SegPolicyENUM */

    /** Temporary: with introduction of \em wgt weighting during Dstorm::store,
     * we have the option of applying the scaling factor during store, or during
     * reduce.  For now we ONLY support a single policy for this, so it is not yet
     * within SegPolicy as an option. */
#define SCALE_DURING_STORE 1

    /** dstorm barrier has a default timeout < ORM_BLOCK.  This can be
     * changed with Dstorm::set_default_barrier_timeout_ms().  For unit
     * tests with small data sets, 60000 (one minute) may work.  For larger
     * datasets, this may need to be increased. */
#define DSTORM_BARRIER_TIMEOUT_MS 900000
    // 900000 = 15 minutes

    /** For more configuration control over transport, you can supply
     * specialized transport classes.
     * Templated mostly for notational convenience.
     * Transport<SHM|MPI|IB|GPU|...> forces all transports to have object names
     * written in a particular way.  Various constructor declarations
     * "look the same", for example, and collapse into fewer lines of code.
     *
     * - Implementation requirements:
     *   - static constexpr TransportEnum const transport = TR;
     *   - default-constructible (with default reasonable for short tests)
     */
    template< TransportEnum TR > class Transport;

    class Dstorm;

    /** Segment information
     */
    class SegInfo;
    //typedef void (*SegStreamFunc)( uint32_t bufnum );   // exposed from SegInfo which store such a pointer
    /** NEW: now a std:function. This is copy constructible and copy assignable, but
     * is now a non-POD data member of segInfo.  Install via Dstorm::setStreamFun,
     * (because it requires non-const access to SegInfo). This function is called
     * for REDUCE_STREAM <em>poll-mode segments only</em>, and after it returns,
     * Dstorm will return an ack to the sender of that buffer. */
    typedef std::function<void(uint32_t const bufnum)> SegStreamFunc;

    /** dstorm implementation helpers */
    namespace detail
    {
        /** virtual base trick (C++ ensures the constructor called only once).
         * An aid to get important stuff initialized first.
         */
        struct DstormComminit;

        /** Support for multiple ranks running on the same node.
         *
         * Running multiple ranks on the same node may be faster via shared memory.
         *
         * Also provide some interprocess synchronization support. Ex. interprocess
         * sequentialization of Dstorm::push(), because otherwise segment queue size
         * may easily overflow.
         */
        struct DstormIPC;

        /** fault tolerance, specific to IB Infiniband for now.
         * Supplies \c barrier() (not including current dead nodes). */
        class DstormLiveness;

        /** Not sure if this needs to be exposed. If we succeed in doing
         * compile-time correctness checks, it may only be nice for debug.
         */
        class SegInfoMap;

        /** The dStorm ranks express a "liveness" concept.
         *
         * - This is a 'hidden' class,  with 2 basic
         *   functions (\c live(r) and \c update()).
         * - One implementation would use IB functions
         *   to mark bad nodes as \em not-live.
         * - main API functions:
         *   - bool dead( Tnode r )
         *   - bool live( Tnode r )
         *   - update()
         *   - vector<Tnode> getDead()
         */
        class LiveBase;

        /** top-most SegInfo functionality */
        template<typename FMT> class SegImpl;

        /** wrapper around SegInfo */
        template< typename IMPL, typename TDATA > class SegBase;

        // --- declarations ---

    }//detail::

    /** type lookup helper.  Specialize for all segment types you need.
     *
     * - Example:
     \verbatim
     * template<> struct SegTag<111> {
     *     typedef user::Seg_VecDense<float> type;        // required
     *     enum { value=111 };                      // unused !!
     * };
     \endverbatim
     *   - 111 is a tag that the user will use for store/reduce operations.
     *   - \c type is the class handling those ops
     *     - User knows 111, not the orm segment id.
     *   - \c type is derived from a type::Base that includes much common functionality
     *   - \c type MUST ultimately derive from SegInfo
     *   - custom SegTag<WHICH>::type should be \b easy to write, allowing
     *     custom reduce operations, for example.
     */
    template < unsigned short USERTAG > struct SegTag;

    /** a specialization of std::runtime_error */
    class orm_error;

    /** create error message, print on stderr, throw orm_error exception.
     * The \c NEED macro calls this library function to reduce inlining bloat.
     * \throw dStorm::orm_error
     */
    void throw_orm_error( orm_return_t const orm_return_code,
                            char const * file, unsigned line,
                            char const * msg );

    /** provided in dstorm_fwd.hpp to simplify header inclusions */
    struct orm_error : public std::runtime_error
    {
        explicit orm_error( std::string const& what )
            : std::runtime_error(what) {}
        explicit orm_error( char const* what)
            : std::runtime_error(what)
        {}
        // what, operator = std ones OK
    };

    struct OldIoNet { OldIoNetEnum const ionet; };
    std::string name( OldIoNet const ionet);

    struct SegPolicyIO { SegPolicy const policy; };
    std::string name( SegPolicyIO const policy );

    struct OrmReturn{ orm_return_t const orm_xxx; };
    char const* name( OrmReturn const retcode );

    /** Mnemonics for \c Dstorm::wait(Qpause,queue_id,timeout_ms). */
    typedef enum {
        QPAUSE_NONE,
        QPAUSE_WAIT_HALF,
        QPAUSE_WAIT,
        QPAUSE_BARRIER,
    } Qpause;

//#if ! WITH_LIBORM
    /** transport queue id's for writes, acks, ... (different message types) may benefit
     * from not having sequential time-depencies if they were all in the same transport queue */
    enum : orm_queue_id_t { GQUEUE_write=0, GQUEUE_ack=1 };
//#endif

}//dStorm::
#endif // DSTORM_FWD_HPP
