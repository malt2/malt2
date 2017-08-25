/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#include "orm_shm.h"
#include "shm/shormOps.hpp"
#include "shm/shormMgr.hpp"
#include <assert.h>

#define TBD(...) do{orm->printf(orm,__VA_ARGS__);}while(0)

using shorm::ShormMgr;

static inline struct Shorm* toShorm( struct Orm* const orm ){
    assert( orm->obj != NULL );
    return static_cast<struct Shorm*>( orm->obj );
}
static inline ShormMgr* mgr( struct Shorm const* const shOrm ){
    return reinterpret_cast<ShormMgr*>( shOrm->opaque );
}
static inline ShormMgr const* cmgr( struct Shorm const* const shOrm ){
    return reinterpret_cast<ShormMgr const*>( shOrm->opaque );
}

extern "C" {
/** SHM segment create.
*
* Comments here apply to \b each segment created.
*
* - SHM segments should work differently from IB. \b Why?
*   - IB "most efficient" xfer is RDMA write[s], then reduce
*   - SHM "most efficient" xfer is plain read from shared memory
*
* -# Application publishes shm oBufs
*    - \b \em ScopedTryPush, write lock semantics, pushes shared memory
*      address onto:
*      -# global MPMC queue, for "any folks can grab this" \em RandRead policy
*      -# or individual MPSC queues for "targeted message" \em TargRead policy
*      - \em WildWrite usage (HogWild) would use a single oBuf, and
*        continue modifying irrespective of readers
*      - while \em SyncWrite<N> usage would make one copy into bounded queue
*        of N oBufs (historical \e multicopy used a single oBuf for \c store op).
*        - this copy could be implicit as a transformation step of a previous
*          result into a next one (ex. <em>apply gradient</em> to model parms).
*        - or a completely fresh dataset (ex. <em>new gradient</em>)
*        - you get a ptr the next oBuf, and fill it directly, then skip
*          the \c store and just \c push.)
*        - the \c push operation for SHM would be a (useless) memcpy into
*          particular rBufs of the segment of another rank (/thread).
*        .
* -# Application reduces from shm iBufs
*    - \b \em ScopedTryPop, retrieves one next shm oBuf, as an iBuf now, from another rank.
*      For SHM, this means you read from oBufs of other segments, and reduce into yours.
*      - Semantics:
*        -# fail (no shm input buffer available)
*        -# succeed, terminated by explicit "ignore" (do not increment read count)
*        -# succeed, with "release" semantics (explicit or out-of-scope)
*           - release semantics bumps up a read count, and if the read count is
*             equal to its target value...
*             -# RandRead (global data queue for this segment)
*                - removes pointer to shm data from global queue and marks the
*                  shm data buffer so the writer can re-use it.
*             -# TargRead always removes pointer to shm data from single-reader's
*                input queue,
*                - and bumps up a read count that writer can read to reclaim any
*                  fully-used oBufs
*                - [opt] use a LazyCounter (for many threads).
*                .
* -# Segment concept involves:
*   -# shm queue[s] of pointers to shm data buffers
*   -# shm data pointer and buffer lifetime management via <em>read count</em>
*   -# shm areas for data
*   -# ScopedPush and ScopedTryPop semantics
*   -# TargRead or RandRead implementation policy.
*
* - Shared memory segments required:
*\verbatim
*    "ShmCtrl" is the shm entry point:
*        - configuration, # threads, setup parameters
*        - transport type == SHM
*        - per segment: read count gc threshold,
*                       segment data item count, segment data size,
*                       segment buffer size, # buffers per segment
*                       thread_local pointer-to-segment named "sTTSS"
*        - note: unlike orm have NO rBuf receive buffers
*                              and NO iBuf input buffers
*        - shMpmc[seg]  array of { read_count (u8), thread # (u16), segment # (u8) }
*        NOTE: both RandRead and TargRead have same data in shMpmc queue so it
*              can be within the ShmCtrl memory region :)
*
*    "sTTSS" segments for TT = thread # = 0,1,2...nThreads are data segments with:
*                         SS = segment # = 0, [1, 2, ... SHCNF_MAXSEG]
*              - a number of fixed size oBufs, written by owning thread.
*\endverbatim
*
* RandRead data structure
*\verbatim
* RandRead:
*    ShmCtrl --->  ShMpmc[SS] - MPMC queues indicating oBufs that can be read for segment SS
*                        |    - with <em>read count</em> garbage collection semantics
*                        |      affecting ShMpmc and sTT shm regions.
*                        |    - each add_segment creates a new ShMpmc active queue
*                        |      up to SHCNF_MAXSEG
*                        |
*                        +---> read count, thread # & oBuf # within segment "sTT"
*                        |     - uint8_t,  uint16_t,  uint8_t
*                        |
*                        +--> sTTSS: segment SS of oBufs for thread TT
*                                    gc flag set when read count fulfilled
*             Operations:
*               ScopedTryPush( [this thread,] seg, oBuf# )  :  push onto ShMpmc[SS]
*               ScopedTryPop(seg) read shMpmc[seg SS] for "next available entry that is not from me",
*                                 rlock entry shMpmc[seg][r], read TT and obuf#
*                                 use data buffer obuf# in shm segment sTTSS for "reduce" operation.
*\endverbatim
*
* TargRead segment layout [original] : nice because ShMpmc part is "same" as for RandRead
*\verbatim
*    ShmCtrl --->  ShMpmc[SS] - MPMPC array of readable oBufs for segment SS
*                        |      + read count per oBuf, thread # & oBuf # within segment "sTTSS"
*                        |    - every oBuf "ptr" gets transferred to some number of sTT input queues
*                        |
*                        +---> read count, thread # & oBuf # within segment "sTT"
*         +--------------+     - uint8_t,  uint16_t,  uint8_t
*         |                
*         +--> sTTSS: ptrQueue[r] circular queue of ShMpmc[SS][r] atomic "pointers":
*                                 - seg # (u8) + index r (u8) within ShMpmc[seg][r]
*                                 - set of TT's oBuf buffers
*                                 -(perhaps a single iBuf targeted by each reduce call?)
*             Operations:
*               ScopedTryPush( [this thread,] seg, oBuf# )
*               :      push onto ShMpmc[SS]
*               :  AND put backptr into sTTSS
*               ScopedTryPop(seg) read sTTSS ptrQueue, dequeue next ptrQueue[] element
*               :                 read corresponding shMpmc[SS] entry,
*               :                      (rdCnt, PP=producer thread, oBuf#)
*               :                 determine sPPSS address of data buffer of PP, oBuf#
*               :                 reduce!
*               :                 update ShMpmc read count (atomic incr)
*\endverbatim
*
* - TargRead segment layout [optimization ONLY] :
*   - ShMpmc ops could well take 100s of ns, so could maybe get much faster
*     - use arrays bounded SPSC queues instead, for speed.
*     - NO read-modify-write ops should be needed in fast-path.
*     - all queues in shm still for GC purposes, GC means need to modify SPSC to be able to "ACK" somehow
*     - little data structure re-use compared with ShMpmc.
*\verbatim
*    ShmCtrl ---> XXX nothing extra
*                  ShSpsc[SS]
*                  - SS = segment, producer = producer's rank, consumer = consumer's rank
*                  - SPSC [bounded/circular] arrays each of size sz transmitting the
*                    (known) producer's oBuf #.
*                  - read count per oBuf, thread # & oBuf # within segment "sTTSS"
*                           - every oBuf "ptr" gets transferred to some number of sTT input queues
*              sTTSS: - array of ShSpsc[R](sz) SPSC circular queues of oBuf numbers.
*                         - producer is thread TT
*                         - consumer is individual rank \#, in range R = out-degree for node TT in IoNet
*                             - IoNet translates index 0..R-1 to 0..nThreads-1 "rank"
*                         - queue data is the oBuf #
*                         - sz is max number of outbound buffer numbers that can be simultaneously pushed
*                     - 0..sz-1 oBufs for data, eached sized according to add_segment instructions.
*                     - and augmented for a gc count "return path" (adds complexity)
*             Operations:
*               ScopedTryPush( [this thread,] seg, oBuf# )  :  set readCnt of oBuf# to (?) size of sendList {r}.
*                                 For each of {r} access the receivers' ShSpsc queues in sRRSS,
*                                 and push the sender's seg & oBuf# 
*               ScopedTryPop(seg) read sTTSS, for sender's seg & oBuf#. ++readCnt?. Repeat.
*                                 Reduce the sent buffers.
*                                 finally --(?) --readCnt of each sender buffer.
*\endverbatim
*
* - Note: Vyukov's MPMC queue sequence is of form GEN | MASK  (MASK is some bits == nThreads)
*   - sequence numbers in Vyukov / mstump implementations go like
*     - gen | 0 "free"
*     - gen | 1 in-use    <-- can extend to double as a read count
*     - gen+1 | 0
*  - since read count should never exceed \# of threads, using the sequence counter
*    to store read count seems like a good thing.
*  - \b BUT need to be careful about memory_order issues or do atomicInc, atomicFetch ops!
*/
    orm_return_t shm_segment_create( struct Orm* orm,
                                       const orm_segment_id_t segment_id,
                                       const orm_size_t size,
                                       const orm_group_t /*UNUSED: group*/,
                                       const orm_timeout_t /*UNUSED: timeout_ms*/,
                                       const orm_alloc_t /*UNUSED: alloc_policy*/
                                     )
    {
        // don't support sync for shm
        //orm_return_t ret = orm_sync_register( orm, segment_id, (orm_sync_t)ORM_SYNC_NONE );
        if( segment_id >= SHCNF_MAXSEG ){
            orm->printf(orm,"ERROR: segment_id too large for shared memory");
            return ORM_ERROR;
        }
        orm->printf(orm,"\nWIP: Beginning shm_segment_create( segment_id=%u, size=%u ...)\n",
                    (unsigned)segment_id, (unsigned)size );
        //uint32_t sz = static_cast<uint32_t>(size);

        struct Shorm* shorm = toShorm(orm);
        struct ShormCtrl* shCtrl = mgr(shorm)->getShCtrl();
        // shCtrl is shared memory
        ShormSegInfo& ss = shCtrl->segs[ segment_id ];

        { // XXX serialize this code block XXX
            // scoped seglock
            if( ss.state != 0 ){
                // segment already exists?
            }
            // open_or_create
        }

        return ORM_SUCCESS;
    }
    //  ~ segement_alloc + segment_register
    orm_return_t shm_segment_delete( struct Orm* orm,
                                       const orm_segment_id_t segment_id)
    {
        TBD("TBD %s", __func__);
        //orm_return_t ret = orm_sync_unregister( orm, segment_id ); if( ret == ORM_SUCCESS ){ ... }
        //struct Shorm* shOrm = toShorm(orm);
        return ORM_SUCCESS;
    }

    orm_return_t shm_segment_ptr( struct Orm* orm,
                                    const orm_segment_id_t segment_id,
                                    orm_pointer_t * ptr)
    {
        TBD("TBD %s", __func__);
        //struct Shorm* shOrm = toShorm(orm);
        *ptr = NULL;
        return ORM_SUCCESS;
    }

#ifdef __cplusplus
}//extern "C"
#endif
