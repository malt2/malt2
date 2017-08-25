/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */

#include "orm_mpi.h"
#include "orm_any.h"
#include <stdarg.h>     // ...
#include <stdio.h>      // vsnprintf
#include <stdlib.h>     //atoi
#include <unistd.h>
#include <memory.h>
#include <malloc.h>
#include <assert.h>
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

    int mpi_printf(struct Orm const* const orm, char const* fmt, ...)
    {
        va_list args;
        va_start( args, fmt );
        char line[1024U];
        int nchars;
        nchars = vsnprintf( line, 1020U, fmt, args );
        va_end(args);
        line[nchars] = '\0';

        // FIX: Orm::printf must be usable even if the Orm object is not yet initialized
        //      init sets internal data Orm::obj to not-nullptr
        //      so check this -- if non-NULL, then we have called _MPI_Init
        //      (this is in mpiorm.cpp)
        // OPTIMIZATION:
        //      hostname is fixed for an initialized Orm, so it really should
        //      be cached inside the Orm::obj
        if( orm!=NULL && orm->obj!=NULL){
            char hostname[256]; //={'\0'};
            int rank = atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
            //int rank; //=0;
            //MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            gethostname(hostname, 255);
            fprintf(stdout, "[%s:   %02u] %s", hostname, rank, line);
        }else{
            fprintf(stdout, "[??:   ??] %s", line);
        }
        fflush(stdout);
        return 0;
    }

    struct Orm * mpi_construct (struct Orm const* const src)
    {
        assert( src != NULL );
        //assert( (src == &orm_mpi || src == &orm_gpu, "double-check, remove assertion if OK"!=NULL );
        struct Orm * ret = (struct Orm*)malloc(sizeof(struct Orm));
        memcpy( ret, src, sizeof(struct Orm) );
        *(void**)(&ret->obj) = NULL;
        return ret;
    }

    void mpi_destruct (struct Orm * const orm)
    {
        if( orm->obj != NULL ){
            orm->printf(orm," ** WARNING ** MPI_Destruct while orm->obj non-NULL\n");
        }
        free(orm);
    }
    
    orm_return_t mpi_proc_rank (struct Orm const* const orm, 
                                  orm_rank_t * const rank)
    {
        (void)orm;
        int mpirank;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
        *rank = (orm_rank_t)(mpirank);
        return ORM_SUCCESS;
    }

    orm_return_t mpi_proc_num (struct Orm const* const orm, 
                                 orm_rank_t * const size)
    {
        (void)orm;
        int mpisize;
        MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
        *size = (orm_rank_t)mpisize;
        return ORM_SUCCESS;
    }

    orm_return_t mpi_barrier(struct Orm const* const orm,
                               const orm_group_t group,
                               const orm_timeout_t timeout_ms)
    {
        (void)orm;
        MPI_Barrier(MPI_COMM_WORLD);
        return ORM_SUCCESS;
    }

    orm_return_t mpi_group_create(struct Orm const* const orm,
                                    orm_group_t *group)
    {   
        (void)orm;
        return ORM_SUCCESS;
        //return MPI_Comm_group(comm, (MPI_Group*)group);
    }

    orm_return_t mpi_group_add(struct Orm const* const orm,
                                 const orm_group_t group,
                                 const orm_rank_t rank)
    {
        (void)orm;
        MPI_Group world_group;
        MPI_Comm_group(MPI_COMM_WORLD, &world_group);
        //MPI_Group prime_group;
        // MPI_Group is a structure doesn't work with orm_rank_t
        // MPI_Group_incl(world_group, 1, (const int) &rank, &prime_group);
        // MPI_Group_union((MPI_Group)group, prime_group, (MPI_Group) &group);
        return ORM_SUCCESS;
    }

    orm_return_t mpi_group_commit (struct Orm const* const orm,
                                     const orm_group_t group,
                                     const orm_timeout_t timeout_ms)
    {
        (void)orm;
        return ORM_SUCCESS;
    }

    orm_return_t mpi_state_vec_get (struct Orm const* const orm,
                                      orm_state_vector_t state_vector)
    {
        (void)orm;
        int nProc;
        MPI_Comm_size(MPI_COMM_WORLD, &nProc);
        int i;
        for(i=0; i<nProc; i++) state_vector[i] = 0;

        return ORM_SUCCESS;
    }

    /** Initial orm shim for mpi.
     *
     * - Usage:
     *   - <TT>struct Orm const * orm = & orm_mpi;</TT>
     *   - use \c orm->printf("hello") instead of orm_printf("hello")
     *   - etc.   \c orm_FOO(args) becomes \c orm->FOO(args)
     */
    struct Orm const orm_mpi =
    {
        .obj                = NULL,
        .transport          = ORM_OMPI,
        .printf             = &mpi_printf,
        .construct          = &mpi_construct,
        .destruct           = &mpi_destruct,
        //--------------------------------------
        .proc_init          = &mpi_proc_init,
        .proc_rank          = &mpi_proc_rank,
        .proc_num           = &mpi_proc_num,
        .proc_term          = &mpi_proc_term,
        .wait               = &mpi_wait,
        .barrier            = &mpi_barrier,
        .group_create       = &mpi_group_create,
        .group_create_mpi   = &mpi_group_create_mpi,
        .group_delete_mpi   = &mpi_group_delete_mpi,
        .group_add          = &mpi_group_add,
        .group_commit       = &mpi_group_commit,
        .state_vec_get      = &mpi_state_vec_get,
        .segment_create     = &mpi_segment_create,
        .segment_delete     = &mpi_segment_delete,
        .segment_ptr        = &mpi_segment_ptr,
        .write              = &mpi_write,
        .win_post           = &mpi_win_post,
        .win_start          = &mpi_win_start,
        .win_complete       = &mpi_win_complete,
        .win_wait           = &mpi_win_wait,
        // etc.
        .sync               = &orm_sync,
        .getsync            = &orm_getsync,
    };



#ifdef __cplusplus
}//extern "C"
#endif



