/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef ORM_TYPES_H
#define ORM_TYPES_H
// All orm types go here
//
#include <stdint.h>

typedef void*          orm_pointer_t;
typedef uint_least16_t orm_rank_t;
typedef unsigned char  orm_segment_id_t;
typedef unsigned short orm_notification_id_t;
typedef unsigned int   orm_number_t;

typedef enum 
{
    ORM_ERROR = -1,
    ORM_SUCCESS = 0,
    ORM_TIMEOUT = 1,
} orm_return_t;

typedef unsigned long orm_timeout_t;
typedef unsigned char orm_group_t;
typedef unsigned long orm_size_t;
typedef unsigned long orm_offset_t;
typedef unsigned long orm_alloc_t;
typedef unsigned char orm_queue_id_t;
typedef unsigned char orm_group_t; 
typedef unsigned char *orm_state_vector_t;
typedef unsigned int orm_notification_t;
typedef unsigned long orm_cycles_t;
static const orm_rank_t ORM_GROUP_ALL=0;
static const orm_timeout_t ORM_TEST = 0x0;

static const orm_timeout_t ORM_BLOCK = 0xffffffffffffffff;

enum orm_alloc_policy_flags
{
    ORM_MEM_UNINITIALIZED = 0, /**< Memory will not be initialized */
    ORM_MEM_INITIALIZED = 1,     /**< Memory will be initialized (zero-ed) */
    ORM_MEM_GPU = 2
};


/**
 * * State of queue.
 * *
 * */
typedef enum
{
    ORM_STATE_HEALTHY = 0,
    ORM_STATE_CORRUPT = 1
} orm_vec_state_t;
#endif
