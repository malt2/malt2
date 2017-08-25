/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef SHORMOPS_H
#define SHORMOPS_H
/** @file
 * 'C' interface to dstorm shared memory
 *
 * While many apps may be OK with default implementation of \c Orm,
 * as used in \c Dstorm, some may wish a little bit more access to
 * particulars of the shared memory implementation.
 *
 * (Ex. debug info, statistics, tuning share memory parameters
 *      or policies)
 *
 * Note: liborm must be linked with -lrt to pull in routines like shm_open, shm_unlink
 *
 * - ipc_barrier.hpp :
 *   - this, from boost, has issues with being placed in shm, so reverted to pthreads
 *   - what pthreads (and most barrier impls) \b lack is the ability to
 *     reset or trigger the barrier automatically.
 *   - Why would this be nice?
 *     - A \e resettable barrier simplifies startup,
 *     - and allowing dynamic process/thread "join"/"detach" from
 *       an existing barrier may also be useful if processes die
 *       (or get spawned) to adapt to resource availability.
 * - Note: there is a nice "Latches and Barriers" proposal that might
 *   be better... see "google-concurrency-library" which implements
 *   the new proposed standard.
 *
 * Mar'15: <B>using SHM_THREAD == SHM_STD_THREAD barrier,
 *            with pre-determined number of participants</B>
 */
#define SHM_IPC 0
//#define SHM_PTHREAD 1
#define SHM_STD_THREAD 2
#define SHM_THREAD SHM_STD_THREAD

#include "orm.h"                // struct Orm dispatch table

#if SHM_THREAD == SHM_IPC

#include "ipc_barrier.hpp"      // patterned after boost/thread/barrier.hpp
#include <boost/version.hpp>
#if BOOST_VERSION < 105400
#error "Please use boost version >= 1.54"
#endif
#error "ipc barrier is no longer supported"

#elif SHM_THREAD == SHM_STD_THREAD // revert to std::thread barrier
#include <mutex>
#include <condition_variable>

#else
//#include "pthread.h"
#error "SHM_THREAD setting not supported"
#endif // SHM_PTHREAD

#include <stdint.h>

/** @name ShmConf constants
 * \c ShmConf is more easily handled if it is fixed-size.
 * Since this is also 'C' code, use preprocessor macros. */
//@{

/// max segments (increase if nec.)
#define SHCNF_MAXSEG 8U

/// max shm processes/threads/ranks using shared memory segments for write operations.
/// Typically will use nThreads <= \# of CPUs << SHCNF_MAXRANKS.
#define SHCNF_MAXRANKS 128U

/// max length of segment name C-string
#define SHCNF_MAXNAM 16U

/// magic version tag for Orm shared memory data structures
#define SHCNF_MAGIC 0xcafaceb4

//@}



#ifdef __cplusplus
extern "C" {
#endif
    struct Shorm;
    struct ShormConf;
    struct ShormSeg;    // no idea what needs to go here yet.

    /** Dstorm control region static config. */
    struct ShormConf {
        // empty, for now, but may end up with different "types"
        // of shared memory implementations
        /** NEW: passed in from dStorm::Transport<TR> via orm->obj
         *       during Orm construction */
        uint_least16_t const nThreads;
    };

    enum NetPolicy {
        TargRead              ///< push/reduce ONLY from specific machines according to IoNet_t graph.
            , RandRead        ///< reduce ScopedTryPop gets ANY AVAILABLE msg according to graph degree only
    };
    /** Part of \c ShormCtrl is an array of per-segment info.
     *
     * - These are "constants" set up when a segment is constructed.
     * - May want ipc someday,
     *   - so thread-local info (like pointers) do not belong here
     * - libdstorm depends on liborm, not reverse,
     *   - so Dstorm structures or types do not belong here
     */
    struct ShormSegInfo {       // in shared memory as ShormCtrl::segs[i]
        char segName[SHCNF_MAXNAM];     ///< "SegSSS" where SSS is segment number
        int seglock;                    ///< ? ipc lock to serialize seg create/delete (perhaps piggyback on rdCntMax as a seq lock?)
        uint32_t state;                 ///< 0=free, 1=in use
        uint_least16_t nRanks;          ///< how many ranks are involved in SHM push/reduce for this segment
#if 0
        orm_rank_t orm_ranks[nRanks]; ///< translate rank for this segment to global orm_rank_t
        uint_least8_t out_degrees[nRanks];///< how many shm outputs for each of nRanks
        uint_least8_t in_degrees[nRanks];
        uint_least8_t in_ranks[nRanks][ in_degrees[nRanks] ]; ///< for each consumer, what are shm ranks of each connected producer

        uint_least8_t iBufBeg[nRanks+1];  ///< producer P has iBufs[II] where II is in range iBufBeg[P] .. iBufBeg[P+1]-1
        /// we might not even need any iBufs (for "reduce" output destination)
        ShmOffset iBufs[ iBufBeg[nRanks] ];

        /** Producer P is assigned oBufs[OO] where OO is in range oBufBeg[P] to oBufBeg[P+1]-1.
         *
         * How many oBufs per P?  Well, this is related to how much \em staleness
         * you are willing to tolerate.  It is also related to how much variance
         * you expect to see in training example processing time.
         *
         * - It is safer to make this a bit larger, because code is safer if staleness
         *   is detected on the consumer side of the spsc queues.
         *  - If a producer invalidates, it runs a slight risk of buggering up
         *    the async client "reduce".
         *  - Similar problems are there for producer-side reductions of the stale
         *    data buffers (which might seem nice if some P always has stale buffers?)
         *
         * - I think perhaps 4-8 oBufs per producer is plenty fine,
         *   - because when speeds are reasonably matched, I
         *     expect queue size <= 2 most of the time,
         *     peaking at 1 element.
         */
        uint32_t oBufBeg[nRanks+1];
        ShmOffset oBufs[ oBufBeg[nRanks] ];            ///< all oBufs, for all producers
        /** Each P has out_degrees[P] spsc Output Queues, \ oq.
         * Each oq gets "oBuf is ready" message. */
        ShmOffset oq[ oBufBeg[nRanks] ];
        uint_least8_t oqRanks[ oBufBeg[nRanks] ]; ///< map oq[Q] to consumer's rank
#endif
#if 0
        // for now, let's put dynamic array sizes and worry about offsets later
        // These SHOULD all be encapsulated with accessor functions to calculate offsets
        //
        // this is version 0. next version will use layout::Lay class to manage remembering the various offsets.
        // Lay globInfo[nRanks];
        // for each lay in globInfo[nRanks];
        //   oThisNode   = lay.add<bool8>(); // or maybe machine number
        //   oShmRank    = lay.add<u8>();
        //
        // Lay rankInfo[nShmRanks];
        // for each ShmRank r
        //   Lay& lay = rankInfo[r]
        //   oGaspi_rank = lay.add<gRank>();
        //
        //   oOutDegree  = lay.add<u8>();
        //   oShmOutRank = lay.add<u8>( out_degree of r );
        //   oInDegree   = lay.add<u8>();
        //   oShmInRank  = lay.add<u8>( in_degree of r );
        //   // buffer assignments within segment
        //   oShmIBufs   = lay.add<u8>();
        //   oShmIranks  = lay.add<ShmRank>( *oShmIbufs );
        //   oShmObufs   = lay.add<u8>();
        //   oShmOranks  = lay.add<ShmRank>( *oShmObufs );
        //   // buffer sizing: see transport struct for this.
        //   // (they don't use the layout framework)

        /** Each rank gets a number of output buffers related to some maximum staleness.
         * Each rank also gets many spsc queues to handle all the outgoing connections.
         *
         * - Producers, \em P,  use a bounded cyclic buffer for "push" data (single producer).
         * 
         * - For SHM, client-side read is the fastest mechanism,
         *   - because it avoids useless memcpy to client buffers (all buffers remain with producer)
         *   - and it allows producer to work with vector data maintained in shm.
         *   - some memcpy might be necessary to produce new versions and maintain old
         *     ones for the async client reads, but this may often be bundled with some
         *     'update' step fairly naturally.
         *   - [Opt.] if P notices that each output queue in
         *
         * - this is contrary to transport implementation based on RDMA write!
         *
         * - <B>How "reduce" works</B>
         * - P determines next free oBuf \em O, if any, by looking for \c ack_count==0
         *   - P bumps generation number of that oBuf, and sets \c ack_count to zero.
         *   - locking out extremely-delayed consumers, who can notice that O is stale
         *     - \em hope that they notice before beginning the reduce, rather than at end
         * - Data is copied only once into P's next free oBuf.
         * - P sets \c ack_count to out_degrees[P]
         *    - ack_count is decremented as each reader is finished with the oBuf.
         * - P publishes existence by pushing onto out_degrees[P] spsc queues
         *   - Every consumer-side spsc queue is initialized consumer-side knowing:
         *     - who P is,
         *     - which of P's oBufs is to be read,
         *     - shm segment offset of P's oBuf.
         *   - queue message has ___ parts:
         *     - oBuf generation number (allows producer-side "revoke" for any oBuf, in extremis)
         *   - consumer \b C determines addresses of spsc queues for each of its
         *     incoming producers, via \c in_ranks[client][...]
         *   - C peeks at one Q entry, say from producer P and in P's shm segment
         *     - determine ptr to oBuf, ...
         *     - Q.pop ignoring the entry if fishypop Q entry immediately if fishy:
         *       - i.e. generation doesn't match or Q.ack_count==0
         *     - <EM>REDUCE HERE</em>
         *     - --Q.ack_count
         *     - popFront
         *   - OR C peeks at ALL Qs from ALL different Ps,
         *        does wholesale reduce (into an iBuf or into C client-side buffer?)
         *        and then does the full set of { --ack_count and popFront } ops.
         */
        uint_least8_t oBufs[nRanks][maxStale * 2];
        uint32_t oBufsTotal;            ///< for 
#endif
        uint32_t segSz;                 ///< byte length of ShormSeg

        NetPolicy policy;               ///< RandRead (default) or TargRead (transport-like, targeted)

        uint_least8_t segnum;           ///< [optional] which segment are we, again?

        /** Memory for array of MPMC queue C++ objects. align CACHE_LINE_SIZE?
         * 
         * elements are triples of { read count, thread \#, and oBuf \# }
         * with data types { uint8_t, uint16_t, uint8_t };
         *
         * These are C++ objects, so all details are left to the Shorm object.
         * lua, for sure, still gets easy read only access to segment
         */
        char mem_for_arrayOfMpmcQueues[ 0 ];
    };

    /** SegSSS share memory blocks are mostly a set of ShormSetProducers.
     *
     * - Each producer has an oBuf array, so ...
     */
    struct ShormSegProducer {
        uint_least8_t out_degree;       ///< how many consumers for each oBuf we produce?


        // NOTE: detailed buffer layout might be better in Shorm
        uint_least8_t oBufBeg;          ///< from libdstorm IoNet_t
        uint_least8_t oBufEnd;          ///< oBufEnd - oBufBeg == out_degree

        /** after this many reads, some gc activity should happen.
         * ?? This may be a C++ structure, bundled int the mpmc memory region. */
        uint_least8_t rdCntMax;

    };
    /** Shared memory segment could be a orm-like segment of many buffers.
     * \em or possibly a single large message queue \em or a reduced
     * set of i/o buffers for streamlined sender-side "reduce" ops.  */
    struct ShormSeg {           // entire shared memory segment
        uint32_t state;         ///< inactive, open, closed ?
        uint32_t sz;            ///< byte length of \c data[] region
        // The orm transport layout is:
        //    producers[ nShm ]         every thread/process gets one entry
        // producers[*] contains:
        //    
        //uint_least8_t in_degree;        ///< how many producers might send to us
    };

    class StdBarrier
    {
    private:
        std::mutex _mutex;
        std::condition_variable _cv;
        std::size_t _count;
    public:
        explicit StdBarrier(std::size_t count) : _count{count} { }
        void wait()
        {
            std::unique_lock<std::mutex> lock{_mutex};
            if (--_count == 0) {
                _cv.notify_all();
            } else {
                _cv.wait(lock, [this] { return _count == 0; });
            }
        }
    };

    /** Dstorm control region dynamic info, <em>located in shared memory</em>. */
    /** \struct ShormCtrl
     * Every process creates/attaches to /dev/ShmCtrl and creates a
     * \c ShmCtrl object using placement new.
     */
    struct ShormCtrl {
#if SHM_THREAD == SHM_IPC
        typedef boost::interprocess::interprocess_mutex Mutex;
        typedef shorm::ipc::barrier Barrier;
#elif SHM_THREAD == SHM_STD_THREAD
        typedef std::mutex Mutex;
        typedef StdBarrier Barrier;
#endif
        int32_t magic;          ///< must match code version
        Mutex mtx;              ///< lock during modifying ops
        /** deficiency: barrier can ONLY be initialized after nRanks
         * value has been (forever) finalized.  nRanks cannot change
         * after barrier is set (needs fixing).
         *
         * ISSUE: barrier has NO default constructor, so ShormCtrl cannot be
         *        default-constructed. Also barrier with count of zero is likely
         *        to throw (instead of being a no-op). */
        Barrier barrier;        ///< reconstructed whenever nRanks is modified
        struct ShormConf conf;
        struct ShormSegInfo segs[SHCNF_MAXSEG];
        orm_rank_t nRanks;    ///< max shm ranks this node will support.
        struct RankInfo {
            // XXX shm threads could be HIDDEN from orm transport,
            //     so rank same for all snRanks running on this node ??
            orm_rank_t rank;          ///< rank in overall Dstorm/orm framework
            /** consistency check: same as thread_local \c irank, which is
             * <B>s</b>ingle <B>n</b>ode's <B>sn</b>Rank index within
             * node-global ShormCtrl::ranks[]. */
            orm_rank_t snRank;
            char status;                ///< dead=0
            // etc. statistics?
        };
        struct RankInfo ranks[SHCNF_MAXRANKS];
    };

    /** any thread can quickly find its rank by looking at the irank'th
     * entry in ShormCtrl::ranks[irank]. We try to assign ranks to low
     * numbers, depending on \c ShormCtrl::ranks[*].status availability. */
    extern thread_local orm_rank_t irank;

    /** Open/create a named region with Dstorm control info. */
    struct Orm const* shorm_open( char const* const path
                                  , int const reattach
                                  , struct Orm const* const orm /*=nullptr*/
                                  //, ShormConf const* const shormConf, // at least create this
                                );


    /** Detach this thread (process?) from the shared memory,
     * \em without releasing any shared memory objects. */
    void shorm_close( struct Shorm** shorm );

    /** Remove a shared memory segment and all its child segments. */
    void shorm_remove( struct Shorm* shorm );

    /** count live entries (r/o except for the ShormCtrl mutex) */
    orm_rank_t shorm_countlive( struct Shorm const* const shorm );

    /** assigns an unused rank within ShormMgr's ShormCtrl shared memory,
     * marks it non-dead. \c rank is the global-orm rank it is associated
     * with, and the node-local rank is stored into \c irank. */

    void shorm_assignRank( struct Shorm*, orm_rank_t rank );

    /** mark any match to \c irank as dead */
    void shorm_removeRank( struct Shorm*, orm_rank_t rank );

    /** This is a 'C'-style iface, so put C++ objects behind the \c opaque ptr. */
    struct Shorm {
        /** @name Common funcs here
         * Because segments are best handled in C++ code, try to generate
         * a C++ lua interface directly <B>(not here)</B>. */
        //@{
        struct ShormConf const* (*getConf)(void);
        char const*             (*getPath)(void);
        uint_least16_t          (*nSegs)(void);
        struct ShormSeg const*  (*getSeg)(uint_least16_t const seg);
        //@}

        /** Hook to low-level Orm API is reproduced here.
         * \c orm always == \c *orm_shm, declared in \ref orm.h,
         * but later may include msg queue vs fixed-buffer impls.
         *
         * XXX BAD DESIGN
         *
         * \todo orm->obj       gives a Shorm
         *       Shorm::opqaue  gives a ShormMgr
         *       ShormMgr       SHOULD point back to the original orm (can keep the Shorm& member)
         */
        Orm const* const orm;        // actually const, but boost had problems
        bool const ownOrm;

        /** Hook to C++ \em ShormMgr impl is opaque (fwd-decl only) */
        void * const opaque;

        ShormConf const shormConf;
    };

#ifdef __cplusplus
}//extern "C"
#endif
/** \struct ShormSegInfo
 * - Shm \em segment may look like a header plus circular message queue.
 *   - or a more orm-like bunch of fixe-size buffers for
 *     hogwild or model-averaging algorithms
 * - Shm \em segment does not support the orm \em group concept.
 *   - functions can be no-op, nothrows
 * - \c shm_segment_create creates a \c ShormSegInfo entry within \c ShormCtrl
 */

/** \f shorm_open
 * - segment, node, network info, ... any global state
 * - if newly created, will have no segments defined.
 * - o/w also gives access to any previously-created segments
 * \p path          name of shared memory segment
 * \p reattach      0=create/reinitialize to no segments,
 *                  1=reattach to any existing segment regions
 *                  <B>only 0 supported for now</B>
 *
 * To restart a calculation, \c reattach=0 deletes any
 * segment regions to the O/S and reinitializes the control
 * region to freshly-created state.
 * 
 * To rejoin an ongoing calculation you might use \c reattach=1
 * This might be used to recover from a killed process (?)
 *
 * The return value <TT>struct Shorm* shorm=shorm_open(...)</TT>
 * is detachd from the shared memory object, freed (and set to NULL)
 * via \c shorm_close(&shorm).  Use \c shorm_remove(shorm) to
 * free the shared memory resources.
 *
 * This may be done automatically during the orm functions
 * \c Orm::proc_init and \c Orm::proc_term.
 */

/** /f shorm_close
 * - In principle could later shorm_open(path,reattach=1) to rejoin
 *   an existing Dstorm calculation.
 * - The shorm object is useless afterward, so we free the memory
 *   and set the pointer to NULL.
 * \post Shorm and ShormMgr destroyed, Orm::obj ptr set to NULL.
 */

/** \f shorm_remove
 * - Returns memory resources (hopefully for clean shutdown!)
 * - Will disrupt any process still using the shared memory.
 * - Might attempt to also cleanly shut down other attached
 *   processes first.
 * - <B>BUGGY until issues with barrier implementation can be sorted out</b>
 */ 
#endif //SHORMOPS_H
