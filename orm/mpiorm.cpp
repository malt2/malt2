/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#include "mpiorm.hpp"
#include "orm_any.h"
#include "orm_mpi.h"
#include<iostream>
#include<assert.h>

using namespace std;
using namespace orm;

static inline struct MormConf* conf(struct Orm const* const orm) {
    assert (orm->obj != NULL);
    return static_cast<struct MormConf*> (orm->obj);
}

#ifndef __cplusplus
#error "this must be compiled with C++"
#endif

extern "C" {

    orm_return_t mpi_segment_create (struct Orm const* const orm,
                                       const orm_segment_id_t segment_id,
                                       const orm_size_t size,
                                       const orm_group_t group,
                                       const orm_timeout_t timeout_ms,
                                       const orm_alloc_t alloc_policy)
    {
        orm_return_t ret = orm_sync_register( orm, segment_id, (orm_sync_t)ORM_SYNC_NONE );
        if( ret == ORM_SUCCESS ){
            orm->printf(orm, "\nBeginning MPI_Segment_create (segment_id=%u, size=%u ...)\n",
                        (unsigned)segment_id, (unsigned) size);
            struct MormConf* config = conf(orm);
            MPI_Win win;
            float *ptr;
            MPI_Alloc_mem(size, MPI_INFO_NULL, &ptr);
            MPI_Win_create(ptr, size, sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
            config->g2ptr[segment_id] = ptr;
            config->g2mpi[segment_id] = win;
        }
        return ret;
    }

    orm_return_t mpi_segment_delete (struct Orm const* const orm,
                                       const orm_segment_id_t segment_id)
    {
        orm_return_t ret = orm_sync_unregister( orm, segment_id );
        if( ret == ORM_SUCCESS ){
            struct MormConf* config = conf(orm);

            auto found = config->g2mpi.find(segment_id);
            assert( found != config->g2mpi.end() );
            MPI_Win_free( &found->second );
            config->g2mpi.erase(found);

            auto ptr = config->g2ptr.find(segment_id);
            assert( ptr != config->g2ptr.end());
            MPI_Free_mem(ptr->second);
            config->g2ptr.erase(ptr);
        }
        return ret;
    }

    orm_return_t mpi_write(struct Orm const* const orm,
                             const orm_segment_id_t segment_id_local,
                             const orm_offset_t     offset_local,
                             const orm_rank_t       rank,
                             const orm_segment_id_t segment_id_remote,
                             const orm_offset_t     offset_remote,
                             const orm_size_t       size,
                             const orm_queue_id_t   queue,
                             const orm_timeout_t    timeout_ms)

    {
        orm_sync_freeze(orm, segment_id_local);
        // for proper Orm::sync(..) error msg (remove when segment_create gets sync_type parameter)

        struct MormConf* config = conf(orm);
        MPI_Win win = config->g2mpi[segment_id_local];

        orm_sync_t sync_type;
        orm_getsync(orm, segment_id_local, &sync_type);
        if(sync_type==ORM_SYNC_BARRIER)
            MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, win);

        // NO GOOD: this is INSIDE a for(..){ write(*); } loop !!!
        // MPI_Win_fence(0, config->g2mpi[segment_id_remote]);
        //        MPI_Request request;

        MPI_Put(config->g2ptr[segment_id_local] + offset_local/sizeof(float),
                size/sizeof(float), MPI_FLOAT, rank,
                offset_remote/sizeof(float),
                size/sizeof(float), MPI_FLOAT, config->g2mpi[segment_id_remote]);
        //        config->q2request[queue].push_back(request); // put write request into queue

        // NO GOOD: MPI_Win_fence(0, config->g2mpi[segment_id_remote]);

        //MPI_Win_flush(rank,win);
        if(sync_type==ORM_SYNC_BARRIER)
            MPI_Win_unlock(rank, win);
        return ORM_SUCCESS;
    }

    orm_return_t mpi_wait(struct Orm const* const orm,
                            const orm_queue_id_t queue,
                            const orm_timeout_t timeout_ms)
    {
        //  orm->printf(orm, "\n Beginning MPI_Wait (queue_id=%u ...)\n", queue);
        //struct MormConf* config = conf(orm);
        //        cout << "========request  queue size is " << config->q2request[queue].size() << " ==========" <<endl;
        // MPI_Waitall(config->q2request[queue].size(), (config->q2request[queue].data()), MPI_STATUSES_IGNORE);
        //config->q2request[queue].clear();
        //        cout << "========wait finished request  queue size is " << config->q2request[queue].size() << " ==========" <<endl;
        return ORM_SUCCESS;
    }

    orm_return_t mpi_segment_ptr (struct Orm const* const orm,
                                    const orm_segment_id_t segment_id,
                                    orm_pointer_t *ptr)
    {
        struct MormConf* config = conf(orm);
        *ptr = config->g2ptr[segment_id];
        return ORM_SUCCESS;
    }

    orm_return_t mpi_proc_init (struct Orm * const orm,
                                  const orm_timeout_t timeout_ms)
    {
        assert( orm != nullptr);
        MPI_Init(NULL,NULL);
        struct MormConf* conf = new MormConf();
        const_cast<void*&>(orm->obj) = conf;
        return ORM_SUCCESS;
    }

    orm_return_t mpi_proc_term(struct Orm const* const orm,
                                 const orm_timeout_t timeout_ms)
    {
        orm->printf(orm," Beginning _MPI_Finalize: orm ptr %p\n", (void*)orm);
        if(orm->obj != nullptr)
        {
            struct MormConf* config = reinterpret_cast<MormConf*>(orm->obj);
            assert( config->g2mpi.size() == 0 );
            assert( config->g2ptr.size() == 0 );
            assert( config->groups.size() == 0 );
            //assert( orm == config->orm );
            //const_cast<Orm const*&>(config->orm) = nullptr;
            delete config;
            const_cast<void*&>(orm->obj) = nullptr;

            //assert(config==nullptr);
            assert(orm != nullptr);
            assert(orm->obj == nullptr);
        }

        MPI_Finalize();
        return ORM_SUCCESS;
    }

    orm_return_t mpi_group_create_mpi (struct Orm const* const orm,
                                         const orm_segment_id_t segment_id,
                                         int* sendlist,
                                         size_t send_size,
                                         int* recvlist,
                                         size_t recv_size)
    {
        int const verbose=1;
        orm_sync_t sync_type;
        orm_getsync(orm, segment_id, &sync_type);
        if(verbose){
            orm_rank_t rank;
            orm->proc_rank(orm, &rank);
            if(rank==0){
                printf("----mpi_group_create---- sync_type: %d %s -------------------\n",
                       (int)sync_type, ormSync2cstr( sync_type ));
            }
        }
        if(sync_type == ORM_SYNC_NOTIFY || sync_type == ORM_SYNC_NOTIFY_ACK )
        {
            struct MormConf* config = conf(orm);
            MPI_Group fromgroup, togroup, groupworld;
            MPI_Comm_group(MPI_COMM_WORLD, &groupworld);
            MPI_Group_incl(groupworld, send_size, (int*)recvlist, &fromgroup);
            MPI_Group_incl(groupworld, recv_size, (int*)sendlist, &togroup);
            config->groups[segment_id].push_back(groupworld);
            config->groups[segment_id].push_back(fromgroup);
            config->groups[segment_id].push_back(togroup);
        }
        return ORM_SUCCESS;
    }

    orm_return_t mpi_group_delete_mpi (struct Orm const* const orm,
                                         const orm_segment_id_t segment_id)
    {
        orm_sync_t sync_type;
        orm_getsync(orm, segment_id, &sync_type);
        if(sync_type == ORM_SYNC_NOTIFY || sync_type == ORM_SYNC_NOTIFY_ACK )
        {
            struct MormConf* config = conf(orm);
            auto found = config->groups.find(segment_id);
            assert(found != config->groups.end());
            assert( found->second.size() == 3 );
            MPI_Group_free(&(found->second[0]));
            MPI_Group_free(&(found->second[1]));
            MPI_Group_free(&(found->second[2]));
            config->groups.erase( found );
        }
        return ORM_SUCCESS;
    }

    orm_return_t mpi_win_post (struct Orm const* const orm,
                                 const orm_segment_id_t segment_id)
    {
        orm_sync_t sync_type;
        orm_getsync(orm, segment_id, &sync_type);
        if(sync_type == ORM_SYNC_NOTIFY || sync_type == ORM_SYNC_NOTIFY_ACK )
        {
            struct MormConf* config = conf(orm);
            MPI_Win_post(config->groups[segment_id][1], 0, config->g2mpi[segment_id]);
        }
        return ORM_SUCCESS;
    }

    orm_return_t mpi_win_start (struct Orm const* const orm,
                                  const orm_segment_id_t segment_id)
    {
        orm_sync_t sync_type;
        orm_getsync(orm, segment_id, &sync_type);
        if(sync_type == ORM_SYNC_NOTIFY || sync_type == ORM_SYNC_NOTIFY_ACK )
        {
            struct MormConf* config = conf(orm);
            MPI_Win_start(config->groups[segment_id][2], 0, config->g2mpi[segment_id]);
        }
        return ORM_SUCCESS;
    }

    orm_return_t mpi_win_complete(struct Orm const* const orm,
                                    const orm_segment_id_t segment_id)
    {
        orm_sync_t sync_type;
        orm_getsync(orm, segment_id, &sync_type);
        if(sync_type == ORM_SYNC_NOTIFY || sync_type == ORM_SYNC_NOTIFY_ACK )
        {
            struct MormConf* config = conf(orm);
            MPI_Win_complete(config->g2mpi[segment_id]);
        }
        return ORM_SUCCESS;
    }

    orm_return_t mpi_win_wait(struct Orm const* const orm,
                                const orm_segment_id_t segment_id)
    {
        orm_sync_t sync_type;
        orm_getsync(orm, segment_id, &sync_type);
        if(sync_type == ORM_SYNC_NOTIFY || sync_type == ORM_SYNC_NOTIFY_ACK )
        {
            struct MormConf* config = conf(orm);
            MPI_Win_wait(config->g2mpi[segment_id]);
        }
        return ORM_SUCCESS;
    }

}//extern "C"

