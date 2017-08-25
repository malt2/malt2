/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */

#include "orm_any.h"
#include "orm_mpi.h"
#include <stddef.h>
#include <stdarg.h>     // ...
#include <stdio.h>      // vsnprintf
#include <unistd.h>
#include <memory.h>
#include <malloc.h>
#include <assert.h>
#include <mpi.h>
//#include <helper_cuda.h>
//#include <helper_functions.h>


#ifdef __cplusplus
extern "C" {
#endif


#if 0
    int gpu_printf(struct Orm const* const orm, char const* fmt, ...)
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
            int rank;
            char hostname[256];
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            gethostname(hostname, 255);
            fprintf(stdout, "[%s:   %02u] %s", hostname, rank, line);
        }else{
            fprintf(stdout, "[??:   ??] %s", line);
        }
        fflush(stdout);
        return 0;
    }
#endif

#if 0
    static struct Orm * gpu_construct (struct Orm const* const src)
    {
        assert( src != NULL );
        assert( src == &orm_gpu && "double-check, remove assertion if OK"!=NULL );
        const size_t buf_size = sizeof(struct Orm);
        struct Orm * ret = (struct Orm*)malloc(buf_size);
        memcpy( ret, src, buf_size );
        *(void**)(&ret->obj) = NULL;
        return ret;
    }
#endif

#if 0
    static void gpu_destruct (struct Orm * const orm)
    {
        if( orm->obj != NULL ){
            orm->printf(orm," ** WARNING ** MPI_Destruct while orm->obj non-NULL\n");
        }
        free(orm);
    }
#endif

    orm_return_t gpu_proc_init (struct Orm * const orm, 
                                  const orm_timeout_t timeout_ms);

#if 0
    orm_return_t gpu_proc_rank (struct Orm const* const orm, 
                                  orm_rank_t * const rank)
    {
        (void)orm;
        int mpirank;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
        *rank = (orm_rank_t)mpirank;
        return ORM_SUCCESS;
    }
#endif

#if 0
    orm_return_t gpu_proc_num (struct Orm const* const orm, 
                                 orm_rank_t * const size)
    {
        (void)orm;
        int mpisize;
        MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
        *size = (orm_rank_t)mpisize;
        return ORM_SUCCESS;
    }
#endif

    orm_return_t gpu_proc_term(struct Orm const* const orm,
                                 const orm_timeout_t timeout_ms);
    orm_return_t gpu_wait(struct Orm const* const orm,
                            const orm_queue_id_t queue,
                            const orm_timeout_t timeout_ms);
    //TODO: not working, no corresponding cuda-aware barrier in mpi 
    orm_return_t gpu_barrier(struct Orm const* const orm,
                               const orm_group_t group,
                               const orm_timeout_t timeout_ms)

    {
        (void)orm;
        MPI_Barrier(MPI_COMM_WORLD);
        return ORM_SUCCESS;
    }

#if 0
    orm_return_t gpu_group_add(struct Orm const* const orm,
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
#endif

#if 0
    orm_return_t gpu_group_commit (struct Orm const* const orm,
                                     const orm_group_t group,
                                     const orm_timeout_t timeout_ms)
    {
        (void)orm;
        return ORM_SUCCESS;
    }
#endif

#if 0
    orm_return_t gpu_state_vec_get (struct Orm const* const orm,
                                      orm_state_vector_t state_vector)
    {
        (void)orm;
        int nProc;
        MPI_Comm_size(MPI_COMM_WORLD, &nProc);
        int i;
        for(i=0; i<nProc; i++) state_vector[i] = 0;
        return ORM_SUCCESS;
    }
#endif

    orm_return_t gpu_segment_create (struct Orm const* const orm,
                                       const orm_segment_id_t segment_id,
                                       const orm_size_t size,
                                       const orm_group_t group,
                                       const orm_timeout_t timeout_ms,
                                       const orm_alloc_t alloc_policy);
    orm_return_t gpu_segment_delete (struct Orm const* const orm,
                                       const orm_segment_id_t segment_id);
    orm_return_t gpu_group_create_mpi (struct Orm const* const orm,
                                         const orm_segment_id_t segment_id,
                                         int* sendlist,
                                         size_t send_size,
                                         int* recvlist,
                                         size_t recv_size);
    orm_return_t gpu_group_delete_mpi (struct Orm const* const orm,
                                         const orm_segment_id_t segment_id);
    orm_return_t gpu_segment_ptr (struct Orm const* const orm,
                                    const orm_segment_id_t segment_id,
                                    orm_pointer_t * ptr);
    orm_return_t  gpu_write(struct Orm const* const orm,
                              const orm_segment_id_t segment_id_local,
                              const orm_offset_t offset_local,
                              const orm_rank_t rank,
                              const orm_segment_id_t segment_id_remote,
                              const orm_offset_t offset_remote,
                              const orm_size_t size,
                              const orm_queue_id_t queue,
                              const orm_timeout_t timeout_ms);

    orm_return_t gpu_win_post (struct Orm const* const orm,
                                 const orm_segment_id_t segment_id);
    orm_return_t gpu_win_start (struct Orm const* const orm,
                                  const orm_segment_id_t segment_id);
    orm_return_t gpu_win_complete (struct Orm const* const orm,
                                     const orm_segment_id_t segment_id);
    orm_return_t gpu_win_wait (struct Orm const* const orm,
                                 const orm_segment_id_t segment_id);
    /** Initial orm shim for gpu.
     *
     * - Usage:
     *   - <TT>struct Orm const * orm = & orm_gpu;</TT>
     *   - use \c orm->printf("hello") instead of orm_printf("hello")
     *   - etc.   \c orm_FOO(args) becomes \c orm->FOO(args)
     */
    struct Orm const orm_gpu =
    {
        .obj                = NULL,
        .transport          = ORM_GPU,
        .printf             = &mpi_printf,              //&gpu_printf,
        .construct          = &mpi_construct,           //&gpu_construct,
        .destruct           = &mpi_destruct,            //&gpu_destruct,
        //--------------------------------------
        .proc_init          = &mpi_proc_init,
        .proc_rank          = &mpi_proc_rank,           //&gpu_proc_rank,
        .proc_num           = &mpi_proc_num,            //&gpu_proc_num,
        .proc_term          = &mpi_proc_term,
        .wait               = &gpu_wait,
        .barrier            = &gpu_barrier,
        .group_create       = NULL,
        .group_create_mpi   = &mpi_group_create_mpi,    //&gpu_group_create_mpi,
        .group_delete_mpi   = &mpi_group_delete_mpi,    //&gpu_group_delete_mpi,
        .win_post           = &mpi_win_post,            //&gpu_win_post,
        .win_start          = &mpi_win_start,           //&gpu_win_start,
        .win_complete       = &mpi_win_complete,        //&gpu_win_complete,
        .win_wait           = &mpi_win_wait,            //&gpu_win_wait,
        .group_add          = &mpi_group_add,           //&gpu_group_add,
        .group_commit       = &mpi_group_commit,        //&gpu_group_commit,
        .state_vec_get      = &mpi_state_vec_get,       //&gpu_state_vec_get,
        .segment_create     = &gpu_segment_create,
        .segment_delete     = &gpu_segment_delete,
        .segment_ptr        = &gpu_segment_ptr,
        .write              = &gpu_write,
        // etc.
        .sync               = &orm_sync,
        .getsync            = &orm_getsync,
    };



#ifdef __cplusplus
}//extern "C"
#endif



