/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef DSTORM_COMMON_H
#define DSTORM_COMMON_H

#include <cstdint>


#ifndef WITH_LIBORM
#define WITH_LIBORM 0
#endif
#ifndef WITH_MPI
#define WITH_MPI 1 
#endif
#ifndef WITH_GPU
#define WITH_GPU 1 
#endif

#if WITH_LIBORM && ! WITH_GPU
#define WITH_SHM 1
#else
#define WITH_SHM 0
#endif

#include <orm_types.h>
#include <orm_fwd.h>

/** some things, like MKL, may require data vectors to have longer alignment.
 * 16 should be OK for SSE instruction set. If using a vec_header before your
 * data, the size of that struct should also be aligned.
 */
#define DSTORM_DATA_ALIGN 16
#define DSTORM_ALIGN_UP(x) (((x)+(DSTORM_DATA_ALIGN-1U)) / DSTORM_DATA_ALIGN * DSTORM_DATA_ALIGN)

/** assert for ORM_SUCCESS: \c throw_orm_error is declared in \ref dstorm_fwd.hpp */
#define NEED(X) do{ orm_return_t orm_return_code = (X); \
    if( orm_return_code != ORM_SUCCESS ) { \
        dStorm::throw_orm_error( orm_return_code, __FILE__, __LINE__, #X ); \
    } \
}while(0)

#endif // DSTORM_COMMON_H
