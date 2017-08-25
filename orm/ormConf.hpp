/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef ORMCONF_HPP
#define ORMCONF_HPP
/** @file
 * opaque 'C++' info for ORM_MPI transport.
 */

#include "orm_fwd.h"
#include "orm_types.h"
#include <map>

namespace orm {

    /** Base type for Orm::obj for all transport types.
     * \b ALL Orm::obj must derive from this type.
     */
    struct OrmConf {
        OrmConf() :
            //(int orm_transport) : orm_transport(orm_transport),
            segSyncs()
        {}

        // complicated... for now simply assume that all
        // transports accept ANY sync value, no complaints.
        // transports map it to some "closest" impl.
        //
        // Perhps unknown impls get mapped to Async and
        // print a warning when first used?
        //
        //int orm_transport;

        struct SyncInfo {
            SyncInfo()
                : frozen(false)
                  , syncType(0)   // async
            {}
            SyncInfo( bool const       frozen,
                      orm_sync_t const initial_sync_type = orm_sync_t(ORM_SYNC_NONE) )
                : frozen(frozen)
                  , syncType( initial_sync_type )
            {}
            /** false after \c Orm::create_segment->true,
             * but true after Orm::write,
             * meaning calling \c Orm::sync is illegal. */
            bool frozen;
            orm_sync_t syncType;
        };

        /** keys added/removed by Orm::create_/delete_segment */
        std::map<orm_segment_id_t, SyncInfo> segSyncs;
    };

}//orm::
#endif //ORMCONF_HPP

