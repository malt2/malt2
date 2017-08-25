/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef SHORMMGR_HPP
#define SHORMMGR_HPP
#include "shm/shormOps.hpp"       // Shorm must be a complete type
#include <assert.h>
#include <cstring>
#include <boost/interprocess/mapped_region.hpp>
namespace shorm
{
#define SHORM( ORMPTR, FUNCTION, ARGS... ) ORMPTR->FUNCTION( (ORMPTR), ARGS )
    class ShormMgr
    {
    public:
        /** ShormMgr controls the \em master ShormCtrl shared region.
         *
         * - Processes needing read access to Dstorm segments need
         *   need [at least] to construct a ShormMgr.
         * - R/W segment access is for now only for \em threads of a
         *   single process
         *   - this set of threads are the only ones allowed to
         *     participate in dStorm barrier() calls.
         * - access from Orm is Orm --> Shorm --> ShormMgr
         *
         * Each thread of a process that partakes in Dstorm semantics
         * is assigned a thread-local rank after calling ShormMgr::assignRank.
         *
         * \pre sh.orm is valid but doesn't yet point to us. When our
         * member functions get called, \c sh.orm.obj==this holds.
         *
         * throw on error, \post shCtrl non-NULL.
         *
         * \p reattach==0 forces full reinitialization of the shared memory
         * "ShormCtrl" segment.
         *
         * ShormMgr does NOT handle ranks or adding segments: it is an opaque
         * helper for struct \c Shorm.
         */
        ShormMgr( struct Shorm& sh, char const* const path, int const reattach )
            : sh(sh)
              , mappedRegion(nullptr)
              , shCtrl(nullptr)
        {
            //assert( sh != nullptr );
            assert( path != nullptr );
            assert( strlen(path) > 0 );
            //SHORM(sh.orm, printf, "Create shm region TBD\n");
            this->attach_shared_memory( path, reattach );
        }

        /** If we "register" when this thread/process attaches to
         * shm, we would "unregister" in the destructor.
         * A No-op for now. */
        ~ShormMgr()
        {
            if( shCtrl ) { shCtrl = nullptr; }
            if( mappedRegion ) { delete mappedRegion; mappedRegion=nullptr; }
        }
        void delete_shared_memory();
        struct ShormCtrl *getShCtrl() const { return shCtrl; }
    private:
        /** initialize easy things in ShormCtrl.
         * - uninitialized:
         *   - conf     done in \ref shorm_open
         *   - barrier  done in \ref shorm_open
         *     - unusable until threads are actually spawned !
         *   - segments by calling add_segment
         */
        void attach_shared_memory( char const* const path, int const reattach );

        struct Shorm& sh;        ///< the 'C' frontend to \c Orm \em orm funcs

        /** Shared memory control region. Note: when this object is destroyed,
         * the memory mapping for this process goes away (SEGFAULT).  So we
         * keep the mappedRegion alive.
         * this->shCtrl is then \em equivalent to mappedRegion.get_address(); */
        boost::interprocess::mapped_region * mappedRegion;
        /** Shared memory control region, mapped into our address space */
        struct ShormCtrl *shCtrl;
    };
}//shorm::
#endif // SHORMMGR_HPP
