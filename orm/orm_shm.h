/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef ORM_SHM_H
#define ORM_SHM_H
#include "orm_types.h"

// We need various orm_XXX types to declare function prototypes for
// our Orm dispatch tables.

#ifdef __cplusplus
extern "C" {
#endif

    // only need fwd decl for this header.
    struct Orm;

    // copy from orm.h and change func names
    //    :%s/(\*\([a-zA-Z_]*)/shm_\1/
    int shm_printf (struct Orm* orm,
                    const char *fmt, ...);
    orm_return_t shm_proc_init (struct Orm* orm,
                                  const orm_timeout_t timeout_ms);
    orm_return_t shm_proc_rank (struct Orm* orm,
                                  orm_rank_t * const rank);
    orm_return_t shm_proc_num (struct Orm* orm,
                                 orm_rank_t * const proc_num);
    orm_return_t shm_proc_term (struct Orm* orm,
                                  const orm_timeout_t timeout_ms);
    orm_return_t shm_wait (struct Orm* orm,
                             const orm_queue_id_t queue,
                             const orm_timeout_t timeout_ms);
    orm_return_t shm_barrier (struct Orm* orm,
                                const orm_group_t group,
                                const orm_timeout_t timeout_ms);
    orm_return_t shm_group_create (struct Orm* orm,
                                     orm_group_t * const group);
    //orm_return_t orm_group_delete (struct Orm* orm,
    //const orm_group_t group);
    orm_return_t shm_group_add (struct Orm* orm,
                                  const orm_group_t group,
                                  const orm_rank_t rank);
    orm_return_t shm_group_commit (struct Orm* orm,
                                     const orm_group_t group,
                                     const orm_timeout_t timeout_ms);
    orm_return_t shm_state_vec_get (struct Orm* orm,
                                      orm_state_vector_t state_vector);
    //  ~ segement_alloc + segment_register
    orm_return_t shm_segment_create (struct Orm* orm,
                                       const orm_segment_id_t segment_id,
                                       const orm_size_t size,
                                       const orm_group_t group,
                                       const orm_timeout_t timeout_ms,
                                       const orm_alloc_t alloc_policy);
    orm_return_t shm_segment_delete (struct Orm* orm,
                                       const orm_segment_id_t segment_id);
    orm_return_t shm_segment_ptr (struct Orm* orm,
                                    const orm_segment_id_t segment_id,
                                    orm_pointer_t * ptr);
    orm_return_t shm_write (struct Orm* orm,
                              const orm_segment_id_t segment_id_local,
                              const orm_offset_t offset_local,
                              const orm_rank_t rank,
                              const orm_segment_id_t segment_id_remote,
                              const orm_offset_t offset_remote,
                              const orm_size_t size,
                              const orm_queue_id_t queue,
                              const orm_timeout_t timeout_ms);
    // ---------------------------
    // orm_error_message ?

#ifdef __cplusplus
}//extern "C"
#endif
#endif // ORM_SHM_H
