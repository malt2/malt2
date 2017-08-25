/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#include "shm/shormOps.hpp"
#include "shm/shormMgr.hpp"

#include <iostream>

using namespace std;

#ifdef __cplusplus
extern "C" {
#endif

    /** Initially, we thought shared memory would be one per PROCESS, but
     * perhaps we should allow pthreads...
     * This means that coding requires support concurrency (not just re-entrancy)
     *
     * Even if shm_proc_init were called once from the master and spawned threads,
     * SOME of our functions should be threadsafe.  I.e. two trheads each thinking
     * they are master is likely to be bad.
     */
    //static pthread_mutex_t mtx_shm_init;

    orm_return_t shm_proc_init ( struct Orm * const orm_,
                                   const orm_timeout_t /*timeout_ms*/ )
                                   // XXX uint32_t nThreads
    {
        //std::cout<<" shm_proc_init..."<<std::endl;
        orm_->printf(orm_," orm_ shm_proc_init ...\n");

        ShormConf const* config = nullptr;
        bool master = false;
        int reattach = 1;
        if( orm_ != nullptr && orm_->obj != nullptr ){
            config = reinterpret_cast<ShormConf const*>(const_cast<void const*>(orm_->obj));
            if( config->nThreads > 0U ){
                orm_->printf(orm_, " orm_ shm_proc_init MASTER ... %u threads", (unsigned)config->nThreads);
                master = true;
                reattach = 0;
            }
        }
        if( !master) orm_->printf(orm_," orm_ shm_proc_init SLAVE ...\n");
            
        Orm const* orm = shorm_open("ShormCtrl", reattach, orm_ );      // BUG: valgrind says this leaks XXX
        orm->printf(orm," shm_proc_init: shorm_open ret ptr %p\n", (void*)orm );

        // For SHM, orm->obj is a Shorm*
        //   and orm->obj->opaque is a ShormMgr*
        //     that has a real pointer to the ShormCtrl info
        //     (in shared memory) with info about Dstorm segments.

        Shorm * const shOrm = reinterpret_cast<Shorm*>( orm->obj );
        if(master){
            shorm_assignRank( shOrm, 0 ); // for now don't care about orm rank '0'
            orm->printf(orm," master thread associated local irank=%u with orm rank 0\n", (unsigned)irank);
        }else{
            // shCtrl->barrier is now initialized in shorm_open
            shorm_assignRank( shOrm, 0 ); // for now don't care about orm rank '0'
            orm->printf(orm," slave thread associated local irank=%u with orm rank 0\n", (unsigned)irank);
        }

        return ORM_SUCCESS;
    }
    /** reentrant.  If orm->obj is non-null, free its memory via \c shorm_close. */
    orm_return_t shm_proc_term (struct Orm* orm,
                                  const orm_timeout_t timeout_ms)
    {
        orm->printf(orm," shm_proc_term: orm ptr %p\n", (void*)orm );
        //assert(orm != nullptr); // FAILS for > 1 thread (each will call this function) assert(orm->obj != nullptr);
        if( orm->obj != nullptr ) {
            struct Shorm* shorm = reinterpret_cast<Shorm*>( orm->obj );
            orm->printf(orm," shm_proc_term-->shorm_close for irank=%u, orm @ %p, shorm @ %p\n",
                        (unsigned)irank, (void*)orm, (void*)shorm );
            shorm_close( &shorm );

            orm->printf(orm," shm_proc_term: orm ptr %p\n", (void*)orm );
            assert( shorm == nullptr );     // postcondition
            assert( orm != nullptr );
            assert( orm->obj == nullptr );  // perhaps not ?
            //free( orm );  // later get invalid free, lots of vg errors
        }
        return ORM_SUCCESS;
    }

    orm_return_t shm_proc_num (struct Orm* orm,
                                 orm_rank_t * const proc_num)
    {
        assert(orm != NULL );
        assert(orm->printf != NULL);
        Shorm const* const shOrm = reinterpret_cast<Shorm*>( orm->obj );
        if( shOrm == nullptr ) { // maybe we got called before shm_proc_init or after shm_proc_term?
            *proc_num = 999U;   // if so, return something "impossible"
        }else{
            assert( shOrm != nullptr );
            //orm->printf(orm," shOrm->opaque = %p\n", (void*)(shOrm->opaque) );
            ShormCtrl const* const shCtrl = reinterpret_cast<shorm::ShormMgr const*>
                (shOrm->opaque)->getShCtrl();
            {
                //bip::scoped_lock< bip::interprocess_mutex >( shCtrl->mtx );
                // pretend atomic read of nRanks is ok
                *proc_num = shCtrl->nRanks;
                //orm->printf(orm,"shm_proc_num nRanks=%u\n",*proc_num);
            }
            if( *proc_num == 0 || *proc_num >= SHCNF_MAXRANKS ){
                orm->printf(orm," Strange: shm_proc_num is %u\n", *proc_num);
            }
        }
        return ORM_SUCCESS;
    }

    orm_return_t shm_proc_rank (struct Orm* orm,
                                  orm_rank_t * const rank)
    { 
        // DO NOT use orm->printf (it would recurse) TBD("TBD %s", __func__);
        // unless you set RECURSE_GAURD for shm_printf
        *rank = irank;
        //orm->printf(orm,"shm_proc_rank irank=%u\n", (unsigned)irank);
        // based on irank (see ShormOps.h), look up ShormCtrl.ranks[irank],
        // doublecheck consistency, return ShormCtrl.ranks[irank].rank
        return ORM_SUCCESS;
    }


    /** dead thread detection is NOT AVAILABLE */
    orm_return_t shm_state_vec_get (struct Orm* orm,
                                      orm_state_vector_t state_vector)
    {
        Shorm const* const shorm = reinterpret_cast<Shorm*>( orm->obj );
        ShormCtrl const* const shCtrl = reinterpret_cast<shorm::ShormMgr const*>
            (shorm->opaque)->getShCtrl();
        //XXX still need to firm up whether we return state for this node only,
        //XXX or within some global orm transport
        //    FOR NOW, use the orm-specific "nProc"
        uint32_t nLive = 0U;
        for(orm_rank_t i=0U; i<shCtrl->nRanks; ++i)
        //for(orm_rank_t i=0U; i<SHCNF_MAXRANKS; ++i)
        {
            ShormCtrl::RankInfo const& ri = shCtrl->ranks[i];
            unsigned char state = ORM_STATE_CORRUPT;

            if( ri.status != 0 ){ // not dead
                //orm->printf(orm, "live shm: i=%u rank=%u snRank=%u irank=%u\n",
                //            (unsigned)i, (unsigned)ri.rank, (unsigned)ri.snRank, (unsigned)irank );
                //if( i == ri.snRank ){
                state = ORM_STATE_HEALTHY;
                ++nLive;
                //}
                assert( ri.snRank == i || ri.snRank == (orm_rank_t)-1U );   // sanity check
            }

            state_vector[ i ] = state;
        }
        orm->printf(orm, "shm_stat_vec_get: nLive = %u", (unsigned)nLive );
        return ORM_SUCCESS;
    }

    orm_return_t shm_barrier (struct Orm* orm,
                                const orm_group_t /*group*/,
                                const orm_timeout_t /*timeout_ms*/)
    {
        Shorm const* const shorm = reinterpret_cast<Shorm*>( orm->obj );
        ShormCtrl * const shCtrl = reinterpret_cast<shorm::ShormMgr const*>
            (shorm->opaque)->getShCtrl();
        shCtrl->barrier.wait();
        return ORM_SUCCESS;
    }

#ifdef __cplusplus
}//extern "C"
#endif
