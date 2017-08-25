/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#include "orm_shm.h"
#include "orm.h"
#include <stddef.h>      // NULL
#include <stdio.h>

#ifdef __cplusplus
//extern "C" {
#error "Please compile with gcc -- g++ may have dropped support for designated initializers"
#endif

//#define TBD(X) do{;}while(0)
#define TBD(...) do{orm->printf(orm,__VA_ARGS__);}while(0)

// copy from orm.h and change func names
//    :%s/(\*\([a-zA-Z_]*)/shm_\1/
//void shm_printf (const char *fmt, ...);
//orm_return_t shm_proc_init (struct Orm* orm,
//const orm_timeout_t timeout_ms);
orm_return_t shm_wait (struct Orm* orm,
                         const orm_queue_id_t queue,
                         const orm_timeout_t timeout_ms)
{ TBD("TBD %s", __func__);
    return ORM_SUCCESS; }
orm_return_t shm_group_create (struct Orm* orm,
                                 orm_group_t * const group)
{ TBD("TBD %s", __func__);
    return ORM_SUCCESS; }
//orm_return_t orm_group_delete (struct Orm* orm,
//const orm_group_t group);
orm_return_t shm_group_add (struct Orm* orm,
                              const orm_group_t group,
                              const orm_rank_t rank)
{ TBD("TBD %s", __func__);
    return ORM_SUCCESS; }
orm_return_t shm_group_commit (struct Orm* orm,
                                 const orm_group_t group,
                                 const orm_timeout_t timeout_ms)
{ TBD("TBD %s", __func__);
    return ORM_SUCCESS; }
orm_return_t shm_write (struct Orm* orm,
                          const orm_segment_id_t segment_id_local,
                          const orm_offset_t offset_local,
                          const orm_rank_t rank,
                          const orm_segment_id_t segment_id_remote,
                          const orm_offset_t offset_remote,
                          const orm_size_t size,
                          const orm_queue_id_t queue,
                          const orm_timeout_t timeout_ms)
{ TBD("TBD %s", __func__);
    //orm_sync_freeze(orm, segment_id_local); // for proper Orm::sync(..) error msg (remove when segment_create gets sync_type parameter)
    return ORM_SUCCESS;
}

