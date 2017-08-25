/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */

#include "orm.h"
#include <stddef.h>
#include <malloc.h>
#include <string.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

    // ----- fwd decl version of Orm.h prepending "shm_" toa function names
    int shm_printf (struct Orm const* const orm,
                    const char *fmt, ...);
    static struct Orm * shm_construct (struct Orm const* const src)
    {
        assert( src != NULL );
        assert( src == &orm_shm && "double-check, remove assertion if OK"!=NULL );
        struct Orm * ret = (struct Orm*)malloc(sizeof(struct Orm));
        memcpy( ret, src, sizeof(struct Orm) );
        *(void**)(&ret->obj) = NULL;
        return ret;
    }
    static void shm_destruct (struct Orm * const orm)
    {
        //orm->obj = NULL;        // XXX this MIGHT need to be overridden.
        if( orm->obj != NULL ){
            orm->printf(orm," ** WARNING ** shm_destruct freeing orm @ %p while orm->obj non-NULL\n", (void*)orm);
        }else{
            orm->printf(orm," shm_destruct freeing orm @ %p\n", (void*)orm);
        }
        free(orm);
    }

    static orm_return_t shm_group_create_mpi (struct Orm const* const orm,
                                    const orm_segment_id_t segment_id,
                                    int* sendlist,
                                    size_t send_size,
                                    int* recvlist,
                                    size_t recv_size)
    {
        return ORM_SUCCESS;
    }
    static orm_return_t shm_group_delete_mpi (struct Orm const* const orm,
                                    const orm_segment_id_t segment_id)
    {
        return ORM_SUCCESS;
    }
    
    static orm_return_t shm_win_post (struct Orm const* const orm,
                                    const orm_segment_id_t segment_id)
    {
        return ORM_SUCCESS;
    }
    static orm_return_t shm_win_start (struct Orm const* const orm,
                                     const orm_segment_id_t segment_id)
    {
        return ORM_SUCCESS;
    }
    static orm_return_t shm_win_complete (struct Orm const* const orm,
                                        const orm_segment_id_t segment_id)
    {
        return ORM_SUCCESS;
    }

    static orm_return_t shm_win_wait (struct Orm const* const orm,
                                    const orm_segment_id_t segment_id)
    {
        return ORM_SUCCESS;
    }


    orm_return_t shm_proc_init (struct Orm * const orm,
                                  const orm_timeout_t timeout_ms);
    orm_return_t shm_proc_rank (struct Orm const* const orm,
                                  orm_rank_t * const rank);
    orm_return_t shm_proc_num (struct Orm const* const orm,
                                 orm_rank_t * const proc_num);
    orm_return_t shm_proc_term (struct Orm const* const orm,
                                  const orm_timeout_t timeout_ms);
    orm_return_t shm_wait (struct Orm const* const orm,
                             const orm_queue_id_t queue,
                             const orm_timeout_t timeout_ms);
    orm_return_t shm_barrier (struct Orm const* const orm,
                                const orm_group_t group,
                                const orm_timeout_t timeout_ms);
    orm_return_t shm_group_create (struct Orm const* const orm,
                                     orm_group_t * const group);
    orm_return_t shm_group_add (struct Orm const* const orm,
                                  const orm_group_t group,
                                  const orm_rank_t rank);
    orm_return_t shm_group_commit (struct Orm const* const orm,
                                     const orm_group_t group,
                                     const orm_timeout_t timeout_ms);
    orm_return_t shm_state_vec_get (struct Orm const* const orm,
                                      orm_state_vector_t state_vector);
    //  ~ segement_alloc + segment_register
    orm_return_t shm_segment_create (struct Orm const* const orm,
                                       const orm_segment_id_t segment_id,
                                       const orm_size_t size,
                                       const orm_group_t group,
                                       const orm_timeout_t timeout_ms,
                                       const orm_alloc_t alloc_policy);
    orm_return_t shm_segment_delete (struct Orm const* const orm,
                                       const orm_segment_id_t segment_id);
    orm_return_t shm_segment_ptr (struct Orm const* const orm,
                                    const orm_segment_id_t segment_id,
                                    orm_pointer_t * ptr);
    orm_return_t shm_write (struct Orm const* const orm,
                              const orm_segment_id_t segment_id_local,
                              const orm_offset_t offset_local,
                              const orm_rank_t rank,
                              const orm_segment_id_t segment_id_remote,
                              const orm_offset_t offset_remote,
                              const orm_size_t size,
                              const orm_queue_id_t queue,
                              const orm_timeout_t timeout_ms);
    orm_return_t shm_sync (struct Orm const* const  orm,
                             const orm_segment_id_t segment_id,
                             const orm_sync_t         segment_sync_type);
    // ---------------------------
  

    /** Initial orm shim for shared memory.
     *
     * - Usage:
     *   - <TT>struct Orm const * orm = &orm_transport;</TT>
     *   - use \c orm->printf("hello") instead of orm_printf("hello")
     *   - etc.   \c transport_FOO(args) becomes \c orm->FOO(args)
     *
     * - \em Warning: functions of Orm (\ref orm.h) are NULL if undefined,
     *                so add them to \ref shm_TBD.c to avoid segfaults.
     */
    struct Orm const orm_shm =
    {
        .obj                = NULL,
        .transport          = ORM_SHM,
        .printf             = &shm_printf,
        .construct          = &shm_construct,
        .destruct           = &shm_destruct,
        //--------------------------------------
        .proc_init          = &shm_proc_init,
        .proc_rank          = &shm_proc_rank,
        .proc_num           = &shm_proc_num,
        .proc_term          = &shm_proc_term,
        .wait               = &shm_wait,
        .barrier            = &shm_barrier,
        .group_create       = &shm_group_create,
        .group_create_mpi   = &shm_group_create_mpi,
        .group_delete_mpi   = &shm_group_delete_mpi,
        .win_post           = &shm_win_post,
        .win_start          = &shm_win_start,
        .win_complete       = &shm_win_complete,
        .win_wait           = &shm_win_wait,
        .group_add          = &shm_group_add,
        .group_commit       = &shm_group_commit,
        .state_vec_get      = &shm_state_vec_get,
        .segment_create     = &shm_segment_create,
        .segment_delete     = &shm_segment_delete,
        .segment_ptr        = &shm_segment_ptr,
        .write              = &shm_write,

        // etc.
    };

#ifdef __cplusplus
}//extern "C"
#endif
