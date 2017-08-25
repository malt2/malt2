/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#include "shm/shormMgr.hpp"

#include <stdexcept>

#if 1
#include <boost/version.hpp>
#if BOOST_VERSION < 105400
#error "Please use boost version >= 1.54"
#endif
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#endif

#define ORM_OK( STUFF ) do{ \
    if( (STUFF) != ORM_SUCCESS ){ \
        throw std::runtime_error("ERROR: need " #STUFF ); \
    } \
}while(0)


namespace shorm
{
    namespace bip = boost::interprocess;

    /**NOTE: cacheMaster had a "force" parameter instead of "reattach"
     *       Decide on proper semantics later.
     * - By supplying an external non-null allocator function to orm, it
     * should be possible to also access all orm segments via shared
     * memory. \sa _______ (func name?).
     * - In principle, any thread could fold in reduction results from:
     *   1. shared-memory <em>lockfree queue</em> of pushed \em update messages
     *   1. network <em>single message</em> preallocated segments
     */
    void ShormMgr::attach_shared_memory( char const* const path, int const reattach )
    {
        if( this->shCtrl != nullptr ){
            throw std::runtime_error("duplicate ShormMgr::attach_shared_memory call");
        }
        if( path == nullptr || path[0]=='\0' ){
            throw std::runtime_error("Bad shared memory path");
        }
        if( reattach == 0
            && bip::shared_memory_object::remove( path ) )
        {
            SHORM(sh.orm,printf,"WARNING: removing old shared memory %s\n", path );
            // Safer: if reattach and open_only OK, try to delete child segments too
        }
        size_t const bytes = sizeof(struct ShormCtrl);
        bool reinit = true;
        if( reattach ) {
            try {
                //SHORM(sh.orm,printf, "reattach (shm open_only) memory region %s...\n", path);
                bip::shared_memory_object ctrl( bip::open_only
                                                , path
                                                , bip::read_write );
                reinit = false;
                if( this->mappedRegion == nullptr ){
                    this->mappedRegion = new bip::mapped_region(ctrl, bip::read_write, 0, bytes);
                    SHORM(sh.orm,printf, "  new mappedRegion @ %p", (void*)this->mappedRegion);
                }
                this->shCtrl = reinterpret_cast<ShormCtrl*>( mappedRegion->get_address() );

                // NOTE: if we are a NEW rank, then assignRank must run, but this can foobar the barrier
                //SHORM(sh.orm,printf, "shared memory reattach INCOMPLETE (rank not assigned, barrier not reset)\n" );
            }catch( std::exception& e ){
                SHORM(sh.orm,printf, "shared memory %s reattach failed (will recreate)", path );
            }
        }
        if( reinit ) { // either failed attach or explict reattach==0
            try {
                SHORM(sh.orm,printf, "%s create_only...\n", path);
                bip::shared_memory_object ctrl( bip::create_only
                                                , path
                                                , bip::read_write );
                ctrl.truncate(0);     // newk old one...
                ctrl.truncate(bytes); // completely fresh area now.
                // map entire ctrl area into our address space
                this->mappedRegion = new bip::mapped_region(ctrl, bip::read_write, 0, bytes);

                // Placement new our control structure there
                // this->shCtrl = new (mappedRegion->get_address()) ShormCtrl;
                //Above would invoke default constructors, but Barrier is NOT default-construcible
                this->shCtrl = reinterpret_cast<ShormCtrl*>( mappedRegion->get_address() );

                new (&shCtrl->mtx) bip::interprocess_mutex();
                {
                    //SHORM(sh.orm,printf, "%s created: scoped_lock for init...\n", path);
                    bip::scoped_lock<ShormCtrl::Mutex>( shCtrl->mtx );

                    memset(shCtrl->segs, 0, sizeof(shCtrl->segs) );
                    for( uint32_t i=0U; i<SHCNF_MAXSEG; ++i ){
                        ShormSegInfo& ss = shCtrl->segs[i];
                        snprintf( ss.segName, SHCNF_MAXNAM, "Seg%03u", (unsigned)i );
                    }
                    shCtrl->nRanks = 0U;
                    memset(shCtrl->ranks, 0, sizeof(shCtrl->ranks) );
                }
                SHORM(sh.orm,printf, "%s reinit done, shCtrl @ %p\n", path, shCtrl);
            }catch( std::exception& e ){
                SHORM(sh.orm,printf, "Could not create shared memory segment %s");
                throw e;
            }
        }
        //else{
        //    bip::scoped_lock<ShormCtrl::Mutex>( shCtrl->mtx );
        //    //++shCtrl->nRanks;   // ??
        //}
        assert( this->shCtrl != nullptr );

        // EVERY rank attaches to the shared global item
        // now we have to install OUR rank into shCtrl RankInfo
        //{
            //SHORM(sh.orm,printf, "retrying scoped lock for shCtrl @ %p...\n", shCtrl);
            //bip::scoped_lock<ShormCtrl::Mutex>( shCtrl->mtx );
            //SHORM(sh.orm,printf, " shCtrl->nRanks = %u\n", shCtrl->nRanks);
            // as each shm thread/process attaches it gets its own rank
            // -- may wish to separate this step to a separate call --
            // (later, from shm_ssignRank)
        //}
        //SHORM(sh.orm,printf, "Good, things seem to be working\n\n");
    }

    /** Should delete not only control region but also any shared
     * \em segment regions.  Should we try to nicely signal other
     * processes to shut down?  \sa shorm_remove. */
    void ShormMgr::delete_shared_memory()
    {
        SHORM(sh.orm,printf, "ShormMgr::delete_shared_memory TBD");
    }
}//shorm::
