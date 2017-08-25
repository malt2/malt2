/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef DSTORM_PUSH_IPC
#define DSTORM_PUSH_IPC
/** \file
 * define dStorm::detail::DstormIPC helper class for push_impl_host.
 */

#include "dstorm_fwd.hpp"       // orm_rank_t

#include <boost/interprocess/sync/named_mutex.hpp> // the easiest one to work with
#include <boost/interprocess/sync/scoped_lock.hpp>

namespace dStorm {
    namespace detail {
        namespace bip = boost::interprocess;

        struct dstorm_push_mutex_remover
        {
            dstorm_push_mutex_remover() {
                bip::named_mutex::remove("dstorm_push_mutex");
            }
            ~dstorm_push_mutex_remover() {
                bip::named_mutex::remove("dstorm_push_mutex");
            }
        }; // for classes, destruction is UNSAFE.

        struct DstormIPC
            : private dstorm_push_mutex_remover
        {
            //struct mutex_remove {
            //    mutex_remove() {named_mutex::remove("dstorm_push_mutex");}
            //    ~mutex_remove() {named_mutex::remove("dstorm_push_mutex");}
            //};
            DstormIPC( orm_rank_t /*iProc*/ )
                : dstorm_push_mutex(bip::open_or_create,"dstorm_push_mutex")
            {
                // XXX eventually should "register" self in a shared-memory segment
                // XXX should have count of registrations
                // XXX should have destructor removing shm upon decrement-to-zero
            }
            bip::named_mutex dstorm_push_mutex;
        };

    }//detail::
}//dstorm::

#endif // DSTORM_PUSH_IPC
