/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef DSTORM_TRANSPORT_HPP
#define DSTORM_TRANSPORT_HPP
#include "dstorm_fwd.hpp"

#if WITH_SHM
#include "shm/shormOps.hpp"
#endif

namespace dStorm {

    /** MPI config for Dstorm transport layer */
    template<> class Transport<OMPI>
    {
    public:
        static constexpr TransportEnum const transport = OMPI;
        Transport() {}
    private: // no configurables for mpi
        // mpi config done with mpi_run* scripts and a "machines" file.
    };

    /** MPI-GPU config for Dstorm transport layer */
    template<> class Transport<GPU>
    {
    public:
        static constexpr TransportEnum const transport = GPU;
        Transport() {}
    private: // no configurables for mpi
        // mpi config done with mpi_run* scripts and a "machines" file.
    };


#if WITH_SHM
    /** Shared-memory config for Dstorm transport layer */
    template<> class Transport<SHM>
    {
    public:
        static uint_least16_t const default_nThreads = 4U;
        static constexpr TransportEnum const transport = SHM;
        Transport( uint_least16_t nThreads = default_nThreads )
            : shormConf({nThreads})
        {}
        uint_least16_t getNthreads() const { return this->shormConf.nThreads; }
        /** ShormConf::nThreads is \# of threads to spawn. */
        ShormConf const shormConf;
    };

#endif


}//dStorm::
#endif// DSTORM_TRANSPORT_HPP
