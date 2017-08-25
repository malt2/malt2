/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef MPIORM_HPP
#define MPIORM_HPP
/** @file
 * opaque 'C++' info for ORM_MPI transport.
 */

#if WITH_MPI==0
#error "mpiorm.hpp, but not WITH_MPI?"
#endif

#include "orm.h"          // struct Orm dispatch table
#include "ormConf.hpp"    // base config class
#include <map>
#include <vector>
#include <mpi.h>

namespace orm {

    struct MormConf : public OrmConf {
        MormConf() : OrmConf()
                     //, segsyncs()
                     //, orm(nullptr)
                     , g2mpi()
                     , g2ptr()
                     , groups()
        {}
        //std::map<orm_segment_id_t, int> segsyncs;
        std::map<orm_segment_id_t, MPI_Win>     g2mpi;
        /** Note: There are <em>device memory</em> pointers.
         * - OMPI segment pointers are from MPI_Alloc_mem
         *   - (CPU RAM, maybe even shared/pinned memory?)
         * - GPU segment pointers are from cudaMalloc
         *   - (memory on GPU)
         */
        std::map<orm_segment_id_t, float*>      g2ptr;
        std::map<orm_segment_id_t, std::vector<MPI_Group> > groups;// groupworld fromgroup togroup
        //std::map<orm_queue_id_t,   std::vector<MPI_Request> > q2request;
        //Orm const* const orm;
        //MPI_Comm shmcomm;
        //const_cast<void*&>(this->orm->obj) =  MormConf;
    };

}//orm::
#endif //MPI_ORM_HPP

