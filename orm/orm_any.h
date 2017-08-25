/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef ORM_ANY_H
#define ORM_ANY_H
/** @file
 * Boilerplate ormConf.hpp 'C' interface.
 */

#include "orm.h"

#ifdef __cplusplus
extern "C" {
#endif
    /// @name transport-agnostic \c Orm::obj support
    //@{
    /** will disappear when Orm::segment_create gets a 'sync_type' parameter */
    orm_return_t orm_sync (struct Orm const* const  orm,
                             const orm_segment_id_t segment_id,
                             const orm_sync_t         segment_sync_type);

    /** retrive segment's sync type -- perhaps implement post/wait start/complete
     * as switch statements based on \c segment_sync_type. */
    orm_return_t orm_getsync(struct Orm const* const  orm,
                               const orm_segment_id_t segment_id,
                               orm_sync_t *             segment_sync_type); 
    //@}
    /// @name internal helpers
    //@{
    /** Orm::write locks in a segments orm_sync setting.
     * \deprecated remove when orm_sync merges into segment_create */
    void orm_sync_freeze( struct Orm const* const orm,
                          const orm_segment_id_t segment_id );

    /** during Orm::segment_create, create a fresh entry in OrmConf */
    orm_return_t orm_sync_register( struct Orm const* const orm,
                                      const orm_segment_id_t segment_id,
                                      const orm_sync_t         orm_sync_type);

    /** during Orm::segment_create, remove an existing entry from OrmConf */
    orm_return_t orm_sync_unregister( struct Orm const* const  orm,
                                        const orm_segment_id_t segment_id);
    //@}
#ifdef __cplusplus
}//extern "C"
#endif
#endif // ORM_ANY_H
