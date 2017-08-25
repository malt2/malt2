/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#include "shm/shormOps.hpp"
#include "shm/shormMgr.hpp"
#include <stdlib.h>             // malloc, free for 'C' stuff ?

#include <iostream>
#include <sstream>

using namespace std;

#if SHM_THREAD == SHM_IPC
#include <boost/interprocess/sync/scoped_lock.hpp>
namespace bip = boost::interprocess;
#define LOCK_GAURD( shCtrl ) bip::scoped_lock< bip::interprocess_mutex >( shCtrl->mtx )

#elif SHM_THREAD == SHM_STD_THREAD
#define LOCK_GUARD( shCtrl ) std::lock_guard<std::mutex>( shCtrl->mtx )

#endif

extern "C" {


    using shorm::ShormMgr;

    thread_local orm_rank_t irank;

    static inline ShormMgr* mgr( struct Shorm const* const shOrm ){
        return reinterpret_cast<ShormMgr*>( shOrm->opaque );
    }
    static inline ShormMgr const* cmgr( struct Shorm const* const shOrm ){
        return reinterpret_cast<ShormMgr const*>( shOrm->opaque );
    }

    /** This is a 'C'-style constructor for struct \c Shorm.
     * not all dispatch functions are set up yet! Add them as
     * the lua interface develops (if ever). */
    struct Orm const* shorm_open( char const* const path
                                  , int const reattach
                                  , struct Orm const* const orm /*=nullptr*/
                                )
    {
        std::ostringstream oss;
        // This is what we are initializing in this function:
        // There is one of these PER THREAD if shm_proc_init is called from each thread !
        struct Shorm* shOrm = (struct Shorm*)malloc(sizeof(Shorm));
        oss<<" shorm_open t"<<(void*)(uintptr_t)pthread_self()<<" malloced Shorm* @ "<<(void*)shOrm<<"\n";
        memset( shOrm, 0, sizeof(Shorm) );

        assert( orm != nullptr ); // until we double-check this use-case
        ShormConf const* config;             // remember configuration pointer (if any)
        // link Shorm to Orm dispatch table
        if( orm == nullptr ){
            const_cast<Orm*&>(shOrm->orm) = (struct Orm*)malloc(sizeof(struct Orm));
            oss<<" Orm malloc @ "<<(void*)shOrm->orm;
            const_cast<bool&>(shOrm->ownOrm) = true;
            memcpy( const_cast<Orm*>(shOrm->orm), &orm_shm, sizeof(struct Orm) );
            config = nullptr;
        }else{
            const_cast<Orm const*&>(shOrm->orm) = orm;
            const_cast<bool&>(shOrm->ownOrm) = false;
            config = reinterpret_cast<ShormConf const *>(const_cast<void const*>(orm->obj));
        }
        if( config != nullptr ) oss<<" config: nThreads = "<<config->nThreads;
        else /*              */ oss<<" config: none "<<(reattach?" reattach":"")<<endl;
        oss<<(reattach?" reattach":"");
        // link Orm ---::obj--> Shorm
        const_cast<void*&>(shOrm->orm->obj) = shOrm;

        // create & install the opaque manager object
        ShormMgr* shormMgr = new ShormMgr( *shOrm, path, reattach );
        oss<<" ShormMgr @ @ "<<(void*)shormMgr;
        const_cast<void*&>(shOrm->opaque) = (void*)shormMgr;
        //cout<<" shormMgr opaque --> "<<(void*)shOrm->opaque<<endl;
        struct ShormCtrl* shCtrl = shormMgr->getShCtrl();
        oss<<" shCtrl @ "<<(void*)shCtrl;
        //cout<<" ShormMgr shCtrl --> "<<(void*)shormMgr->getShCtrl()<<" nThreads="<<shCtrl->conf.nThreads<<endl;

        // ShormMgr constructor has created shared region with name 'path'
        // ... finally we can stick our config directly into shared mem ...
        if( config  && config->nThreads > 0U ){
            // configure shared memory ShormCtrl things JUST ONCE
            if( reattach ){
                std::runtime_error e(" shorm_open: nThreads>0 ~ MASTER, reattach illegal\n");
                oss<<" THREAD t"<<(void*)(uintptr_t)pthread_self()<<" ERROR !\n\t"<<e.what();
                shOrm->orm->printf(shOrm->orm,"%s",oss.str().c_str());
                throw(e);
            }
            memcpy( (void*)&shCtrl->conf, config, sizeof(struct ShormConf) );

            // SHM master pretends all nodes "alive" even if threads have not yet been spawned.
            oss<<"\n\tMASTER: set ri.status=1 but ri.snRank=-1 for "<<shCtrl->conf.nThreads<<" threads, awaiting assignRank";
            const_cast<uint_least16_t&>(shCtrl->conf.nThreads) = config->nThreads;
            for( uint32_t i=0U; i< config->nThreads; ++i ){
                ShormCtrl::RankInfo &ri = shCtrl->ranks[i];
                ri.status = 1U;                 // alive
                ri.snRank = orm_rank_t(-1U);  // awaiting assignRank call (even master thread)
            }
            shCtrl->nRanks = shCtrl->conf.nThreads; // awaiting shm_assignRank ...

            // we now know how many threads the barrier operation intends to coordinate
            new (&shCtrl->barrier) ShormCtrl::Barrier( (unsigned int)shCtrl->conf.nThreads );
            // XXX shCtrl->barrier is not freed, but this is in /dev/shm/ShormCtrl, so NOT a true leak
            //     ... unless we are removing the shared memory entirely
            oss<<"\n\tshorm_open NEW BARRIER( "<<shCtrl->conf.nThreads<<")";
        }else{
            oss<<"\n\tshorm_open slave, nThreads = "<<shCtrl->conf.nThreads;
        }

        shOrm->orm->printf(shOrm->orm,"%s\n",oss.str().c_str());

        assert( (void*)(shOrm->orm->obj) == (void*)(shOrm) );
        return shOrm->orm;
    }

    void shorm_close( struct Shorm** shorm )
    {
        assert( shorm != nullptr );
        struct Shorm* sh = *shorm;
        sh->orm->printf(sh->orm,"shorm_close!!!!!!\n");
        {
            assert( sh != nullptr );
            // free subobjects, make them unusable
            const_cast<void*&>(sh->orm->obj) = nullptr;
            //
            ShormMgr* m = mgr(sh);      // from sh->opaque
            sh->orm->printf(sh->orm, " deleting ShormMgr @ %p", (void*)m );
            delete m;
            const_cast<void*&>(sh->opaque) /*ShormMgr*/ = nullptr;
            //
            cout<<" shorm_close, ownOrm="<<sh->ownOrm<<endl;
            if( sh->ownOrm ){
                sh->orm->printf(sh->orm, " freeing sh->orm @ %p", (void*)sh->orm);
                free( const_cast<Orm*&>(sh->orm) );
                const_cast<Orm*&>(sh->orm) = nullptr;
            }
        }
        cout<<" shorm_close freeing Shorm @ "<<(void*)sh;
        free(sh);
        *shorm = nullptr;
    }

    orm_rank_t shorm_countlive( struct Shorm const* const shorm )
    {
        ShormCtrl * const shCtrl = cmgr(shorm)->getShCtrl();
        orm_rank_t live = 0U;
        {
            LOCK_GUARD( shCtrl );
            for( orm_rank_t i=0U; i<SHCNF_MAXRANKS; ++i ){
                ShormCtrl::RankInfo& ri = shCtrl->ranks[i];
                if( ri.status != 0 ){ // ri.status != 0 means "assigned"
                    if( ri.snRank != i ){
                        throw std::runtime_error(" ri.snRank != i");
                    }
                    if( i >= shCtrl->nRanks ){
                        throw std::runtime_error(" i >= shCtrl->nRanks");
                    }
                    ++live;
                }
            }
            return live;
        }
    }
    /** side-effect: thread_local irank is set as index into
     * ShormCtrl::ranks[irank] entry.
     *
     * \note Unsafe to call in multi-process because of issues with
     *       barrier re-initialization.
     *
     * - Current workaround is to ONLY call shorm_assignRank from
     *   shm_proc_init, which must:
     *   1. master sets nThreads/nRanks to desired value and pretends
     *      all theads already alive.
     *   1. barrier initialized, all RankInfo status != 0
     *   1. slaves assignRank and grab entries with:
     *      - status != 0 and smRank == -1U
     */
    void shorm_assignRank( struct Shorm* shorm, orm_rank_t rank )
    {
        irank = orm_rank_t(-1U);
        ShormCtrl * shCtrl = cmgr(shorm)->getShCtrl();
        {
            LOCK_GUARD( shCtrl );
            orm_rank_t firstfree = orm_rank_t(-1U);
            orm_rank_t lastused = orm_rank_t(-1U);
            // 
            for( orm_rank_t i=0U; i<SHCNF_MAXRANKS; ++i ){
                ShormCtrl::RankInfo& ri = shCtrl->ranks[i];
                if( ri.status == 1 && ri.snRank == orm_rank_t(-1U) ){
                    if( firstfree == orm_rank_t(-1U) ){
                        firstfree = i;
                        //cout<<" firstfree (init) = "<<firstfree<<endl;
                    }
                }else if( ri.status == 0 ){
                    if( firstfree == orm_rank_t(-1U) ){
                        firstfree = i;
                        //cout<<" firstfree (init) = "<<firstfree<<endl;
                    }
                }else{ // ri.status != 0 means "assigned"
                    assert( ri.status == 1 );
                    assert( ri.snRank != orm_rank_t(-1U) );
                    //cout<<" i = "<<i<<" live: snRank="<<ri.snRank<<" irank="<<irank<<endl;
                    if( ri.snRank != i ){
                        throw std::runtime_error("shorm_assignRank: rank improperly assigned?");
                    }
                    lastused = i;
                }
            }
            shorm->orm->printf(shorm->orm,"assignRank: firstfree=%d lastused=%d SHCNF_MAXRANKS=%u\n",
                               (int)firstfree, (int)lastused, (unsigned)SHCNF_MAXRANKS );
            if( firstfree == orm_rank_t(-1U) )
                throw std::runtime_error(" shorm_assignRank: no ranks available" );
            irank = firstfree;
            if( irank < 0  || irank >= SHCNF_MAXRANKS )
                throw std::runtime_error(" irank out of range");
            ShormCtrl::RankInfo& ri = shCtrl->ranks[irank];
            ri.rank = rank;
            ri.snRank = irank;
            ri.status = 1U;
            if( lastused == (orm_rank_t)(-1U) || irank > lastused ) lastused = irank;
            if( lastused >= shCtrl->nRanks ){
                uint32_t old = shCtrl->nRanks;
                shCtrl->nRanks = lastused + 1U;
                shorm->orm->printf(shorm->orm," nRanks was %u,  max increased to %u\n", (unsigned)old, (unsigned)(shCtrl->nRanks) );
                // XXX deabtable ...
            }
            assert( shCtrl->nRanks > 0 && shCtrl->nRanks <= SHCNF_MAXRANKS );
        }
    }
    void shorm_remove( struct Shorm* sh )
    {
        assert( sh != nullptr );
        using namespace shorm;
        ShormMgr* mgr = reinterpret_cast<ShormMgr*>( sh->opaque );
        mgr->delete_shared_memory();
    }
}//extern "C"

