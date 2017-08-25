/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#include "orm.h"
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

    /** orm_construct has bad type-safety issues (perhaps should not be accessible?) */
    struct Orm * orm_construct (struct Orm const* const src, void const* const config)
    {
        struct Orm * orm = src->construct( src );
        if( config != NULL ){
            // HACK to avoid changing the proc_init API (historically matching IB)
            //*(void**)( &orm->obj ) = (void*)config;    // C equiv of const_cast'ing
            // FIXME: this **might** break shm version, but fixes MPI,GPU versions
            //        that want to call MPI_Comm_rank during printf anytime
            //        obj is non-NULL.
            //
            // I expect this will break the foobar shm stuff...
            (void)config;
            *(void**)( &orm->obj ) = NULL;    // C equiv of const_cast'ing
        }
        if(1){
            printf(" orm_construct fresh orm @ %p with config ptr orm->obj @ %p\n"
                   , (void const*)orm, (void const*)(orm->obj) );
            // We now build only the exactly required subset into liborm ...
            printf(" src orm @ %p, src printf func @ %p, orm proc_init func @ %p\n"
                   , (void const*)src, (void const*)src->printf, (void const*)orm->proc_init );
        }
        return orm;
    }
    void orm_destruct (struct Orm * const orm)
    {
        assert( orm != NULL ); // improper destructor events can trigger this
        if( orm != NULL ){
            orm->destruct(orm);     // redirect to orm_destruct or shm_destruct or ...
        }
    }


#ifdef __cplusplus
}//extern "C"
#endif
