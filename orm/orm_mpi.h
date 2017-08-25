/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */

#include "orm_fwd.h"
#include "orm_types.h"
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

    int mpi_printf(struct Orm const* const orm, char const* fmt, ...);
    struct Orm * mpi_construct (struct Orm const* const src);
    void mpi_destruct (struct Orm * const orm);
    orm_return_t mpi_proc_init (struct Orm * const orm, 
                                  const orm_timeout_t timeout_ms);
    orm_return_t mpi_proc_rank (struct Orm const* const orm, 
                                  orm_rank_t * const rank);
    orm_return_t mpi_proc_num (struct Orm const* const orm, 
                                 orm_rank_t * const size);
    orm_return_t mpi_proc_term(struct Orm const* const orm,
                                 const orm_timeout_t timeout_ms);
    orm_return_t mpi_wait(struct Orm const* const orm,
                            const orm_queue_id_t queue,
                            const orm_timeout_t timeout_ms);
    orm_return_t mpi_barrier(struct Orm const* const orm,
                               const orm_group_t group,
                               const orm_timeout_t timeout_ms);
    orm_return_t mpi_group_create(struct Orm const* const orm,
                                    orm_group_t *group);
    orm_return_t mpi_group_add(struct Orm const* const orm,
                                 const orm_group_t group,
                                 const orm_rank_t rank);
    orm_return_t mpi_group_commit (struct Orm const* const orm,
                                     const orm_group_t group,
                                     const orm_timeout_t timeout_ms);
    orm_return_t mpi_state_vec_get (struct Orm const* const orm,
                                      orm_state_vector_t state_vector);
    orm_return_t mpi_win_post (struct Orm const* const orm,
                                 const orm_segment_id_t segment_id);
    orm_return_t mpi_win_start (struct Orm const* const orm,
                                  const orm_segment_id_t segment_id);
    orm_return_t mpi_win_complete(struct Orm const* const orm,
                                    const orm_segment_id_t segment_id);
    orm_return_t mpi_win_wait(struct Orm const* const orm,
                                const orm_segment_id_t segment_id);
    orm_return_t mpi_group_create_mpi (struct Orm const* const orm,
                                         const orm_segment_id_t segment_id,
                                         int* sendlist,
                                         size_t send_size,
                                         int* recvlist,
                                         size_t recv_size);
    orm_return_t mpi_group_delete_mpi (struct Orm const* const orm,
                                         const orm_segment_id_t segment_id);
    orm_return_t mpi_segment_create (struct Orm const* const orm,
                                       const orm_segment_id_t segment_id,
                                       const orm_size_t size,
                                       const orm_group_t group,
                                       const orm_timeout_t timeout_ms,
                                       const orm_alloc_t alloc_policy);
    orm_return_t mpi_segment_delete (struct Orm const* const orm,
                                       const orm_segment_id_t segment_id);
    orm_return_t mpi_segment_ptr (struct Orm const* const orm,
                                    const orm_segment_id_t segment_id,
                                    orm_pointer_t * ptr);
    orm_return_t  mpi_write(struct Orm const* const orm,
                              const orm_segment_id_t segment_id_local,
                              const orm_offset_t offset_local,
                              const orm_rank_t rank,
                              const orm_segment_id_t segment_id_remote,
                              const orm_offset_t offset_remote,
                              const orm_size_t size,
                              const orm_queue_id_t queue,
                              const orm_timeout_t timeout_ms);
#ifdef __cplusplus
}//extern "C"
#endif

