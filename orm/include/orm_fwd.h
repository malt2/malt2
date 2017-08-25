/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef ORM_FWD_H
#define ORM_FWD_H

#ifndef WITH_MPI
#define WITH_MPI 0
#endif
#ifndef WITH_GPU
#define WITH_GPU 0
#endif

#include <stdint.h>
#include <stddef.h>     // 'C' header for size_t

#ifdef __cplusplus
extern "C" {
#endif

    struct Orm;                 // fwd
    struct Shorm;               ///<'C' access to Dstorm shared memory object internal info

    /** transport enums */
    enum OrmTransport {
        ORM_OMPI=1,
        ORM_GPU=2,
        ORM_SHM=3,
    };

    /** OrmTransport --> C string */
    char const* ormTransport2cstr( enum OrmTransport const ot );

    /// @name orm typedefs
    /// All types are in orm_types.h
    //@{
    typedef int8_t orm_sync_t;
    //@}

    /** Orm capability enums */
    enum {
            ORM_WITH_OMPI =
#if WITH_OMPI
            1
#else
            0
#endif
            , ORM_WITH_GPU =
#if WITH_OMPI
            1
#else
            0
#endif
        , ORM_WITH_SHM = 1 ///< always build this \em crippled transport.
    };

    /** Orm sync options (internally mapped to "closest available"
     * impl for the chosen transport) */
    enum OrmSync {
        ORM_SYNC_NONE = 0
            , ORM_SYNC_NOTIFY = 1
            , ORM_SYNC_NOTIFY_ACK = 2
            , ORM_SYNC_BARRIER = 3
            // add ones that don't cleanly map to above two...
            // maybe ORM_SYNC_BARRIER, for debug runs ???
    };

    /** OrmSync --> C string */
    char const* ormSync2cstr( orm_sync_t const ormsync );

#ifdef __cplusplus
}//extern "C"
#endif

#endif // ORM_FWD_H
