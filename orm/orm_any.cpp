/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */

#include "orm_any.h"
#include "orm.h"
#include "ormConf.hpp"
#include <assert.h>

using namespace orm;

/** return \e just the OrmConf base class version of the
 * transport-specific \c Orm::obj */
static inline struct OrmConf* conf(struct Orm const* const orm) {
    assert (orm != nullptr);
    assert (orm->obj != nullptr);
    return static_cast<struct OrmConf*> (orm->obj);
}

extern "C" {

    /** \deprecated */
    void orm_sync_freeze( struct Orm const* const orm,
                          const orm_segment_id_t segment_id )
    {
        assert( orm != nullptr );
        assert (orm->obj != nullptr);
        auto& segsyncs = conf(orm)->segSyncs;
        auto found = segsyncs.find(segment_id); // look up segment_id
        if( found != segsyncs.end() ){
            auto& syncInfo = found->second;
            syncInfo.frozen = true;            // set frozen field to 'true'
        }
    }

    orm_return_t orm_sync (struct Orm const* const  orm,
                             const orm_segment_id_t segment_id,
                             const orm_sync_t         segment_sync_type)
    {
        assert( orm != nullptr );
        assert (orm->obj != nullptr);
        auto& segsyncs = conf(orm)->segSyncs;
        auto found = segsyncs.find(segment_id);
        if( found == segsyncs.end() ){
            return ORM_ERROR;
        }

        auto& syncInfo = found->second;
        if( syncInfo.frozen ){
            return ORM_ERROR; // illegal to change sync method once frozen
        }

        // TODO:  add orm_transport to ALL *Conf constructors
        // --- test for sync_type valid for window transport type HERE ---
        syncInfo.syncType = segment_sync_type;
        // OR assume that transports accept ANY sync value and translate it to
        // their "closest implementation".
        return ORM_SUCCESS;
    }

    orm_return_t orm_getsync (struct Orm const* const  orm,
                                const orm_segment_id_t segment_id,
                                orm_sync_t *             segment_sync_type)
    {
        assert( orm != nullptr );
        assert( orm->obj != nullptr );
        auto& segsyncs = conf(orm)->segSyncs;
        auto found = segsyncs.find(segment_id);
        if( found == segsyncs.end() )
            return ORM_ERROR;
        *segment_sync_type = found->second.syncType;
        return ORM_SUCCESS;
    }

    orm_return_t orm_sync_register( struct Orm const* const  orm,
                                      const orm_segment_id_t segment_id,
                                      const orm_sync_t         orm_sync_type
                                    )
    {
        assert( orm != nullptr );
        assert( orm->obj != nullptr );
        auto& segsyncs = conf(orm)->segSyncs;
        // insert if new key ...
        auto const ins = segsyncs.insert
            ( std::make_pair( segment_id,
                              OrmConf::SyncInfo(false,orm_sync_type)));

        if( ins.second != true )
            return ORM_ERROR; // (probably already there)

        // we successfully inserted a new key-value pair
        return ORM_SUCCESS;
    }
    orm_return_t orm_sync_unregister( struct Orm const* const orm,
                                        const orm_segment_id_t segment_id )
    {
        assert( orm != nullptr );
        assert( orm->obj != nullptr );
        auto& segsyncs = conf(orm)->segSyncs;
        auto found = segsyncs.find(segment_id);
        if( found == segsyncs.end() )
            return ORM_ERROR;         // it MUST be an existing entry

        segsyncs.erase( found );
        return ORM_SUCCESS;
    }
}//extern "C"

