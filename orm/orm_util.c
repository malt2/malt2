/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */

#include "orm_fwd.h"
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

    /** OrmTransport --> C string */
    char const* ormTransport2cstr( enum OrmTransport const ot )
    {
        char const* ret = NULL;
        switch(ot){
          case(ORM_OMPI ): ret="ORM_OMPI" ; break;
          case(ORM_GPU  ): ret="ORM_GPU"  ; break;
          case(ORM_SHM  ): ret="ORM_SHM"  ; break;
        }
        assert( ret != NULL );
        return ret;
    }


    char const* ormSync2cstr( orm_sync_t const ormsync )
    {
        char const* ret = NULL;
        switch(ormsync){
          case(ORM_SYNC_NONE      ): ret="ORM_SYNC_NONE"      ; break;
          case(ORM_SYNC_NOTIFY    ): ret="ORM_SYNC_NOTIFY"    ; break;
          case(ORM_SYNC_NOTIFY_ACK): ret="ORM_SYNC_NOTIFY_ACK"; break;
          case(ORM_SYNC_BARRIER   ): ret="ORM_SYNC_BARRIER"   ; break;
        }
        assert( ret != NULL );
        return ret;
    }

#ifdef __cplusplus
}//extern "C"
#endif

