/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef ORM_H
#define ORM_H

#include "orm_fwd.h"

#if ! WITH_LIBORM
#error "including orm.h but not WITH_LIBORM -- try orm_fwd.h instead?"
#endif

#if 0
// XXX There should probably be NO external includes here.
// The purpose of this header is to define the interface.
// Each implementation should include the headers required for liborm compile.
#if WITH_MPI
#include <mpi.h>
#endif

#if WITH_GPU
#include <mpi-ext.h> /* Needed for CUDA-aware check */
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#endif

#include "orm_types.h"


// We need various orm_XXX types to declare function prototypes for
// our Orm dispatch tables.

// ? static assert typeof orm functions == typeof orm functions ?
//   ( in case orm transport changes a prototype? )

#ifdef __cplusplus
extern "C" {
#endif

    // ---- typing shorthands ----
    /** \c src is address of one of the \c const library objects,
     * \c orm_[transport], \c orm_mpi, or \c orm_shm.
     *
     * \p src: \em Copy the library dispatch table and set the state
     * information pointer
     * \p config: \em NEW accept optional transport layer config info.
     *
     * - Originally, \c Dstorm always passed the address of a
     *   \c Transport<TransportENUM> object as \c config.
     *   - if dStorm::obj is nullptr, transport must fall back to MPI (FIXME).
     *   - otherwise, TransportEnum is \c dStorm::orm->obj.transport,
     *     and can compare with OMPI, GPU or SHM
     *     (defined in \ref dstorm_fwd.hpp)
     * - \b BADNESS: config is stored in \c obj, but later proc_init
     *   resets it to some per-segment info in a C++ struct derived
     *   from \c OrmConf (see \ref ormConf.hpp)
     */
    struct Orm * orm_construct (struct Orm const* const src, void const* const config);

    /** Your process copy of orm should be freed when done.  */
    void orm_destruct (struct Orm * const orm);
    // ---------------------------

#if 0
    /** Some items \b only require a pointer to an Orm::printf
     * function, so don't give them access to all the other
     * network functions.  (Even if you don't give them one,
     * they can always use vanilla printf to compile OK).
     *
     * Note: sig changed to return \c int to agree with ::printf
     *
     * \deprecated with orm supporting multiple transports, there is
     * \b NO unique printf that works for everybody.
     */
    typedef int (*OrmPrintf) (const char *fmt, ...);
#endif

    // ---- orm types (typedefed "same as" transport types -----
    // TBD

    /** Dispatch table mirroring the subset of orm calls used by dstorm.
     *
     *
     * To implement, you may need to also used the same constants
     * (ex. segment_ptr has some reserved orm bytes at beginning of
     *  actual segment OFFSET_BYTES (?))
     *
     * - <B>Threading / Processes</B>
     *   - \em orm_transport:
     *     - One \c Orm per process. I don't think all transport code is thread-safe at all.
     *   - \em orm_shm:
     *     - sudo(?) rm -f /dev/shm/ShormCtrl
     *     - Construct \c Orm once per process (master, creates /dev/shm/ShormCtrl)
     *     - spawned threads attach as slaves to segments described by ShormCtrl.
     *     - ... means that shm impls need to be thread-safe
     *       - \todo orm_shm routines must be made threadsafe</em>
     */
    struct Orm
    {
        /** Opaque state to support orm transport API implementations.
         *
         * - \c obj \b was \c NULL for \c orm_transport, where lib[transport] stores
         *   the specific transport's state info.
         * - \b NEW: orm now offers all transports the ability to set a
         *   synchronization mechanism,
         *   - so all 'obj' must support a mapping from segment to sync type
         * - non-IB Transports may need other Transport-specific opaque info.
         *   - E.g. for ORM_SHM, \c obj points to a \c Shorm storing state
         *     for shared memory,
         *   - and for ORM_MPI/ORM_GPU, MPI-related info is in \c Mormconf
         *
         * - \c orm_transport can be used directly (ignore \c obj)
         *
         * - Shared memory construction is a bit different for each transport:
         *   - proc_init will now take a <em> number of threads</em> that
         *     participate in Dstorm activity.
         *   - This number of threads is fixed, so \c barrier is easy
         *     to implement
         *   - Processes that don't need any \c barrier can still access
         *     the shared memory state and segment buffers.
         *
         * - \c orm_shm should only be accessed as a copy of the dispatch
         *   table attached to a \c Shorm* .
         */
        void* const obj;

        enum OrmTransport const transport;

        //----------------------------------------------------------------
        /// @name anytime functions
        /** Callable even when \c obj is NULL */
        //@{
        int (*printf) (struct Orm const* const orm,     // <-- really?
                       const char *fmt, ...);

        /** malloc a copy of \c src with \c obj=config (can be NULL).
         *
         * \p src is the address of one of the builtin dispatch tables.
         * Because  orm_shm are shared ibrary objects, they
         * are intentional const, and cannot modify their own state.
         * We provide \c construct and \c destruct to create/delete
         * \b copies of the library dispatch tables.  These copies
         * exist in process memory and CAN modify their \c obj state
         * information.
         *
         * \em Recommendation: use \c orm_construct instead, to pass
         * transport layer config info via \c this->obj, so that
         * \c proc_init can use it to complete the orm construction.
         *
         * Eventually, one of \c construct or \c proc_init can disappear.
         * \c proc_init remains only to closely match the original IB API.
         */
        struct Orm * (*construct) (struct Orm const* const src );

        /** free memory obtained with \c construct(struct Orm const* const) */
        void (*destruct) (struct Orm * const orm);
        //@}
        //----------------------------------------------------------------

        /// @name member functions
        /** Member functions only useful after \c shorm_open. */
        //@{
        /// \copydoc orm_proc_init
        orm_return_t (*proc_init) (struct Orm * const orm,
                                     const orm_timeout_t timeout_ms);
        /// \copydoc orm_proc_rank
        orm_return_t (*proc_rank) (struct Orm const* const orm,
                                     orm_rank_t * const rank);
        /// \copydoc orm_proc_num
        orm_return_t (*proc_num) (struct Orm const* const orm,
                                    orm_rank_t * const proc_num);
        /// \copydoc orm_proc_term
        orm_return_t (*proc_term) (struct Orm const* const orm,
                                     const orm_timeout_t timeout_ms);
        /// \copydoc orm_wait
        orm_return_t (*wait) (struct Orm const* const orm,
                                const orm_queue_id_t queue,
                                const orm_timeout_t timeout_ms);
        /// \copydoc orm_barrier
        orm_return_t (*barrier) (struct Orm const* const orm,
                                   const orm_group_t group,
                                   const orm_timeout_t timeout_ms);
        /// \copydoc orm_group_create
        orm_return_t (*group_create) (struct Orm const* const orm,
                                        orm_group_t * const group);
        orm_return_t (*group_create_mpi) (struct Orm const* const orm,
                                        const orm_segment_id_t segment_id,
                                        int* sendlist,
                                        size_t send_size,
                                        int* recvlist,
                                        size_t recv_size);
        orm_return_t (*group_delete_mpi) (struct Orm const* const orm,
                                            const orm_segment_id_t segment_id);
        orm_return_t (*win_post) (struct Orm const* const orm,
                                        const orm_segment_id_t segment_id);
        orm_return_t (*win_start) (struct Orm const* const orm,
                                         const orm_segment_id_t segment_id);
        orm_return_t (*win_complete) (struct Orm const* const orm,
                                            const orm_segment_id_t segment_id);
        orm_return_t (*win_wait) (struct Orm const* const orm,
                                        const orm_segment_id_t segment_id);
        /// \copydoc orm_group_add
        orm_return_t (*group_add) (struct Orm const* const orm,
                                     const orm_group_t group,
                                     const orm_rank_t rank);
        /// \copydoc orm_group_commit
        orm_return_t (*group_commit) (struct Orm const* const orm,
                                        const orm_group_t group,
                                        const orm_timeout_t timeout_ms);
        /// \copydoc orm_state_vec_get
        orm_return_t (*state_vec_get) (struct Orm const* const orm,
                                         orm_state_vector_t state_vector);
        //  ~ segement_alloc + segment_register
        /// \copydoc orm_segment_create
        orm_return_t (*segment_create) (struct Orm const* const orm,
                                          const orm_segment_id_t segment_id,
                                          const orm_size_t size,
                                          const orm_group_t group,
                                          const orm_timeout_t timeout_ms,
                                          const orm_alloc_t alloc_policy);
        /// \copydoc orm_segment_delete
        orm_return_t (*segment_delete) (struct Orm const* const orm,
                                          const orm_segment_id_t segment_id);
        /// \copydoc orm_segment_ptr
        orm_return_t (*segment_ptr) (struct Orm const* const orm,
                                       const orm_segment_id_t segment_id,
                                       orm_pointer_t * ptr);
        /// \copydoc orm_write
        orm_return_t (*write) (struct Orm const* const orm,
                                 const orm_segment_id_t segment_id_local,
                                 const orm_offset_t offset_local,
                                 const orm_rank_t rank,
                                 const orm_segment_id_t segment_id_remote,
                                 const orm_offset_t offset_remote,
                                 const orm_size_t size,
                                 const orm_queue_id_t queue,
                                 const orm_timeout_t timeout_ms);

        /** Set synchronization style for a segment.
         * - call this right after \c Orm::segment_create
         * - 0, or no synchronization, is always a valid \c segment_sync_type
         * - \return nonzero error code when
         *   - \c segment_id is nonexistent, or
         *   - any \c Orm::write calls have been issued, or
         *   - if \c segment_sync_type is unimplemented for this Orm's transport
         * - Non-blocking
         *
         * - <em>For now</em> assume that transports accept ANY sync value and translate
         *   it to their "closest implementation", perhaps 'async' if totally unrecognized.
         *   - this simplifies initial interface.
         *
         * \todo remove Orm::sync by adding extra sync arg to Orm::segment_create.
         */
        orm_return_t (*sync) (struct Orm const* const  orm,
                                const orm_segment_id_t segment_id,
                                const orm_sync_t         segment_sync_type);
        /** nonblocking retrieve synchronization type for a segment.
         * - return nonzero error code when
         *   - \c segment_id is nonexistent, or
         *   - if \c segment_sync_type is unimplemented for this Orm's transport
         * - Non-blocking
         * - If no errors, return ORM_SUCCESS and modify \c sync_type.
         */
        orm_return_t (*getsync) (struct Orm const* const  orm,
                                   const orm_segment_id_t segment_id,
                                   orm_sync_t *             segment_sync_type);
     

#if 0
        /// \name NOTIFY_ACK support
        /// Every segment can have up to [at least] 256 notifications, which are
        /// 4-byte integers.
        //@{
        orm_return_t (*notify_waitsome)(
                                           const orm_segment_id_t       segment_id,
                                           const orm_notification_id_t  ntBeg,
                                           const orm_number_t           num,
                                           const orm_notification_id_t* const first_id
                                           const orm_timeout_t          timeout_ms);
        orm_return_t (*notify_reset)( const orm_segment_id_t      segment_id,
                                        const orm_notification_id_t ntBeg,
                                        orm_notification_t*         oldNtVal);

        orm_return_t (*notify) ( const orm_segment_id_t       segment_id_remote,
                                   const orm_rank_t             rank,
                                   const orm_notification_id_t  notification_id,
                                   const orm_notification_t     notification_value,
                                   const orm_queue_id_t         queue,
                                   const orm_timeout_t          timeout_ms );
        //@}
#endif
    };

    /// \name template Orm objects
    /// These \c const <em>function dispatch tables</em> are used to initialize Orm
    /// objects during the Orm::construct(Orm* src_template) \e constructor calls.
    //@{
#if !defined(__CYGWIN__)                
#if WITH_MPI
    extern struct Orm const orm_mpi;
#endif
#if WITH_GPU
    extern struct Orm const orm_gpu;
#endif
#endif
    /** everybody should be able to compile/test with shared memory dstorm */
    extern struct Orm const orm_shm;
    //@}

#ifdef __cplusplus
}//extern "C"
#endif


#endif // ORM_H
