/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */

#include "orm_shm.h"

#include <stdarg.h>
//#include <sstream>
//#include <iostream>
//#include <iomanip>
#include <stdio.h>      // vsnprintf
#include <stdlib.h>     // exit

#ifdef __cplusplus
#error "Must compile shm_printf.c with gcc"
extern "C" {
#endif

    static void shm_stdout( struct Orm* orm, char const* msg )
    {
        orm_rank_t rank;
        if( shm_proc_rank( orm, &rank ) != ORM_SUCCESS ){
            fprintf(stdout,"ERROR determining shm_proc_rank");
            exit(-1);
        }
        fprintf(stdout,"s%02u: %s",rank,msg);
        fflush(stdout);
    }

#define RECURSE_GAURD 1
#if RECURSE_GAURD
    // Avoid this (globals inside libraries CAN bite you unexpectedly)
    static volatile unsigned shm_printf_recurse=0U;
#endif

    /** print shm msg as "S<rank>: msg" using plain \c cout.
     *
     * Makes a small attempt to have strings emitted in single op.
     * Doesn't yet serialize with an interprocess lock
     *
     * For distributed shm + orm system, think about adding a
     * 'host' prefix and perhaps piggybacking on orm_printf
     * (or mpi printf for other network fabrics).
     */
    int shm_printf (struct Orm* orm,
                    const char *fmt, ...)
    {
#if RECURSE_GAURD
        if( ++shm_printf_recurse == 1U )
        {
#endif
            // stack corrutpion if called recursively
            va_list args;
            va_start( args, fmt );
            char line[1024U];
            int nchars;
            nchars = vsnprintf( line, 1020U, fmt, args );
            va_end(args);
            line[nchars] = '\0';

            shm_stdout( orm, &line[0] );     // should not call back here
#if RECURSE_GAURD
        }
        --shm_printf_recurse;
#endif
        return ORM_SUCCESS;
    }

#ifdef __cplusplus
}//extern "C"
#endif
