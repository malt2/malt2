/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef DSTORM_ANY_HH_
#define DSTORM_ANY_HH_
/** \file
 * Dstorm inlines common to BOTH cpu and gpu impls.
 * - Simple functions, only needing to include \ref segInfo.hh
 * - simple "host" utils for CPU/GPU
 */
#include "dstorm.hpp"
#include "segInfo.hh"


#if defined(DSTORM_LIBRARY_COMPILE)
// library compile wants to produce ALL stuff
//     (there may be a way to force this with flags?)
#define DSTORM_INLINE inline
#else
// normally, one prefers to use inline
#define DSTORM_INLINE inline
#endif

/** throw on error (so don't need an Orm for printf) */
#define SUCCESS_OR_DIE( STMT, VALUE ) do{ \
    if( (STMT) != (VALUE) )  { \
        throw(std::runtime_error( "(" #STMT ") != (" #VALUE ")")); \
    } \
} while(0)

#define ORM_NEED(X) SUCCESS_OR_DIE(X, true )

/** enable for extra NOTIFY_ACK debug messages */
#define NOTACK_DBG 0

namespace dStorm {

    namespace detail {

        /** Dstorm::net point to this "liveness" concept.
         *
         * todo? When LiveBase is ready, pair it with a MkIonet
         *        impl by deriving from dStorm::IoNetBase.
         *
         *  The MkIoNet impl encapsulates a name() and a function call
         *  mkSend(node).  IoNetBase expands to provide to provide lookups
         *  in form of table lookups for sendLists and recvLists(), as
         *  well as adding the liveness-related update() to update the
         *  send/recv lists culling dead (send?) destinations.
         *
         *   \sa liveness.hpp
         *   \sa mkIonet.hpp
         *
         * - Where does IoNet_t get used in init sequence?
         * - <B>Not here</b>, because IoNet_t is for \c \b add_segment !
         * - This is very different from original milde_malt.
         */
        class DstormLiveness
        {
        public:
            /** DstormCommInit is a pre-constructed base of Dstorm and
             * orm is used to access the correct barrier function.
             *
             * \c dBase gives us \c dBase.nProc during construction,
             * but \c orm must remain valid for our whole lifetime. */
            DstormLiveness( DstormCommInit const& dBase );
            ~DstormLiveness();

            /** add a barrier for all \em live transport processes (ranks).
             * Safety issue: perhaps users should NOT even be allowed
             * to call \c dstorm->orm->barrier() !!!
             * Why?
             *  orm->barrier() might not deal with dead nodes properly XXX
             *  (check your transport's code !!!)
             *
             * \p timeout_ms \b NEW: transport (MPI) will hang ALL processes if
             *               using ORM_BLOCK, so require this,
             *
             * and return the error status.
             *
             * \todo investigate whether barrier timeout can magically
             *       recover a set of live nodes and continue.
             */
            orm_return_t barrier( orm_timeout_t const timeout_ms ) const;

            orm_return_t  recover();  ///< wait for pending i/o to complete

            /** exactly mirror transport queue health state vector */
            void set_dead() { state_update(true); }

            /** from transport queue health state vector, add any new deaths
             * (never changes existing dead nodes to live) */
            void add_dead() { state_update(false); }

            /** normally setLive is not required since death is a response to
             * a transport error of some sort.
             *
             * Tricky: CLIENT should redo build_lists too
             */
            void setLive( orm_rank_t const r, bool const isAlive );

            bool live( orm_rank_t const r ) const {
                assert(r < avoid_list.size());
                return avoid_list[r] == 0U;
            }
            bool dead( orm_rank_t const r ) const {
                return ! live(r);
            }
            orm_rank_t nDead() const { return fProc; } // good if do not modify avoid_list yourself
            orm_rank_t nDeadCount();

#if WITH_LIBORM
        private: // could be public
            struct Orm const& orm;
#endif
        public:// ? really ?
            /** This reflects the orm_state_vec_get and any injected failures.
             * NOTE: vector<bool> is not best choice here.  If you inject a failure,
             * this is NOT reflected in fProc (need private avoid_list and setter
             * mods, as well as a const getter XXX) */
            std::vector<unsigned> avoid_list;
        private:
            /** orm_state_vector_t is a ptr to unsigned bytes, where each byte results
             * from OR'ing the states of each transport queue.  I.e. vec[i] != 0 means that
             * some queue for rank has an error */
            orm_state_vector_t vec;

            /** if some are dead, then \c barrier() should operate on a smaller
             * subset of ranks \b (TBD) */
            orm_group_t   survivors;

            /** if any avoid_list, create a orm_group_t \c survivors for \c barrier */
            void setSurvivors();

            /** add/set deaths from transport queue state vector.
             * - Since MPI doesn't support processes going away,
             *   \c avoid_list[iProc] are all pegged at 'false'.
             * - See \ref dstorm.cpp to re-enable transport dead-node handling.
             * - Better: implement Orm::state_vec_get even for OMPI, and
             *           perhaps stop the program nicely if somebody is 'dead'.
             */
            void state_update( bool const exactMirror );

            orm_rank_t    fProc;      ///< count failed Processes (from \c transport_state_vec_get)
        };

    }//detail::

    /// \defgroup NetFunc Network DstormLiveness
    //@{
    DSTORM_INLINE orm_rank_t    Dstorm::nDeadCount() const {
        return net->nDeadCount();
    }
    DSTORM_INLINE orm_rank_t    Dstorm::nDead() const {
        return net->nDead();
    }
    DSTORM_INLINE void
        Dstorm::barrier( orm_timeout_t const timeout_ms /*= 60000U*/ ) const {
        orm_return_t ret = net->barrier( timeout_ms );
if( ret != ORM_SUCCESS ){
            std::ostringstream oss;
            oss<<" r"<<iProc<<" Error: barrier( timeout="<<timeout_ms<<" ms ), returned "<<name(ret);
            ORM_COUT(this->orm, oss.str());
            throw ErrNetwork( oss.str() );
        }
        return; // void
    }
    //@}

    namespace detail {

#if defined(__CYGWIN__)
#if ! WITH_LIBORM
#error "Cygwin does not support trasnport compile (can try WITH_LIBORM and used Transport<SHM>, perhaps)"
#endif

#else


#if WITH_GPU
        template<> DSTORM_INLINE
            DstormCommInit::DstormCommInit( Transport<GPU> const& tr )
            : transport(GPU)
              , iProc(0U), nProc(0U)
              , orm( orm_construct( &orm_gpu, const_cast<Transport<GPU>*>(&tr)  ))
            {
                {
                    std::ostringstream oss;
                    oss<<" DstormComminit: GPU orm @ "<<(void*)orm<<"\n";
                    ORM_COUT(this->orm, oss.str());
                }
                if(orm==nullptr)                // liborm is REQUIRED for GPU transport
                    throw("null GPU orm?");
                this->finalize();
            }
#endif
#if WITH_MPI
        template<> DSTORM_INLINE
            DstormCommInit::DstormCommInit( Transport<OMPI> const& tr )
            : transport(OMPI)
              , iProc(0U), nProc(0U)
              , orm( orm_construct( &orm_mpi, const_cast<Transport<OMPI>*>(&tr)  ))
            {
                {
                    std::ostringstream oss;
                    oss<<" DstormComminit: OMPI orm @ "<<(void*)orm<<"\n";
                    ORM_COUT(this->orm, oss.str());
                }
                if(orm==nullptr)                // liborm is REQUIRED for MPI transport
                    throw("null MPI orm?");
                this->finalize();
            }
#endif
#if WITH_SHM
        static inline struct ShormConf mkShormConf( Transport<SHM> const& tr )
        {
            ShormConf ret{ tr.getNthreads() };
            return ret;
        }
        // only supported constructor for cygwin compile...
        template<> DSTORM_INLINE
            DstormCommInit::DstormCommInit( Transport<SHM> const& tr )
            : transport(SHM)
              , iProc(0U), nProc(0U)
              , orm( orm_construct( &orm_shm, &tr.shormConf ))
                  // NOTE: passing ShormConf (not Transport<SHM>) into orm
                  //       avoids liborm depending on dStorm library.
            {
                {
                    std::ostringstream oss;
                    oss<<" DstormComminit: SHM orm @ "<<(void*)orm<<"\n";
                    ORM_COUT(this->orm, oss.str());
                }
                if(orm==nullptr)                // liborm is REQUIRED for MPI transport
                    throw("null SHM orm?");
                finalize(/*SHM*/);
            }
#endif

    }//detail::
#endif
    /** We \em may have Orm* orm, but we maintain copies of iProc, nProc (ease of use) */
    inline int Dstorm::get_iProc()
    {
        return iProc;
    }

    inline int Dstorm::get_nProc()
    {
        return nProc;
    }

    /** send obuf to \e \b all sendlist ranks (the frequent use case) */
    inline ssize_t Dstorm::push( SegNum const s )
    {
        uint32_t const which = -1U;
        return Dstorm::push_impl( s, which );
    }
#if 1
    inline orm_pointer_t Dstorm::ptrOdata( SegNum const s ) const {
        return mem_as<orm_pointer_t>( validSegInfo(s,__func__) . ptrOdata() );
    }
    inline orm_pointer_t Dstorm::ptrIdata( SegNum const s ) const {
        return mem_as<orm_pointer_t>( validSegInfo(s,__func__) . ptrIdata() );
    }
    inline orm_pointer_t Dstorm::ptrRdata( SegNum const s,
                                          orm_rank_t const recv_index //0..recv_list.size()
                                        ) const {
        SegInfo & sInfo = validSegInfo(s,__func__);
        //assert( recv_index < this->get_recv_list(sInfo.ionet).size() );  // FULL ionet recv list size
        assert( recv_index < this->segRecvVec(s).size() );  // FULL ionet recv list size
        orm_rank_t rbufnum = sInfo.rbufBeg() + recv_index;
        ORM_NEED( rbufnum < sInfo.rbufEnd() );

        return mem_as<orm_pointer_t>( sInfo.ptrData( rbufnum ));
    }
#else // __CUDA_ARCH__ ??
    inline void* Dstorm::ptrOdata( SegNum const s ) const {
        return mem_as<void*>( validSegInfo(s,__func__) . ptrOdata() );
    }
    inline void* Dstorm::ptrIdata( SegNum const s ) const {
        return mem_as<void*>( validSegInfo(s,__func__) . ptrIdata() );
    }
    inline void* Dstorm::ptrRdata( SegNum const s,
                                          unsigned short const recv_index //0..recv_list.size()
                                        ) const {
        SegInfoGpu & sInfo = validSegInfo(s,__func__);
        //assert( recv_index < this->get_recv_list(sInfo.ionet).size() );  // FULL ionet recv list size
        assert( recv_index < this->segRecvVec(s).size() );  // FULL ionet recv list size
        unsigned short rbufnum = sInfo.rbufBeg() + recv_index;
        ORM_NEED( rbufnum < sInfo.rbufEnd() );

        return mem_as<void*>( sInfo.ptrData( rbufnum ));
    }
#endif

    inline SegInfo const& Dstorm::getSegInfo( SegNum const s ) const {
        //static_assert( is_segment_format<FMT>::value,
        //               "ERROR: require is_segment_format<FMT>");
        SegInfoMapType::const_iterator found = this->seginfos.find(s);
        if( found == this->seginfos.end() ){
            throw std::runtime_error("Dstorm::getSegInfo() inactive segment");
        }
        return *found->second;
    }

    inline SegInfo & Dstorm::validSegInfo( SegNum const s, char const* funcname ) const {
        SegInfoMapType::const_iterator found = this->seginfos.find(s);
        if( found == this->seginfos.end()
            || ! found->second->valid )
        {
            throw std::runtime_error(std::string("Dstorm::getSegInfo() inactive segment: ")
                                     .append(funcname));
        }
        return *found->second;
    }

    template<typename FMT> inline
        uint32_t segKey( SegNum const s ){
            static_assert( is_segment_format<FMT>::value,
                           "ERROR: require is_segment_format<FMT>");
            static_assert( sizeof(SegNum) == 2, "Please adjust segKey function" );
            //return ((Dstorm::SegInfoMapType::key_type)(unsigned char)(FMT::value) << 16U)
            //    + s;
            return (Dstorm::SegInfoMapType::key_type) s;
            // NEW:    IGNORE FMT completely, storing FMT::value (unsigned char)
            //         now within SegInfo.
            // WHY?    Allow lua to call non-templated Dstorm::xFoo(SegNum)
            //         instead of templated (original) Dstorm::Foo<FMT>(SegNum)
            // (C++ code can still use the Foo<FMT> version if convenient)
        }


    /** After orm_write we can force some delay.  With no delay, there is the
     * possibility that the client overwrites data that has not yet been sent out.
     *
     * - QPAUSE_BARRIER is a 'debug' setting,
     * - QPAUSE_WAIT was originally always done after push, but is better before the push.
     * - QPAUSE_NONE is a no-op (hogwild) we may frequently send mixed-version oBuf
     */
    inline void Dstorm::wait( Qpause const qpause,
                       orm_queue_id_t const queue_id,
                       orm_timeout_t const timeout_ms )
    {
        switch(qpause)
        {
          case(QPAUSE_NONE):
              break;
          case(QPAUSE_WAIT_HALF):

          case(QPAUSE_WAIT):
              NEED( orm->wait(orm, queue_id, timeout_ms));
              break;
          case(QPAUSE_BARRIER): // oops, this doesn't have a timeout_ms setting
              this->barrier(500000);  // most heavy-handed, most reproducible
              break;
          default: ; // milde compile complained about missing, but all cases are here ?
        }
    }

    inline IoNet_t Dstorm::add_ionet( std::unique_ptr<mm2::UserIoNet>&& ptr ) {
        assert( ptr );
        IoNet_t ret = this->iographs.push_back( this->iProc, std::move(ptr) );
        //assert( ret > IONET_MAX );// never conflict with a 'builtin' tag, 7 predefined and IONET_MAX == 8
        build_lists(); // NB
        return ret;
    }

    inline std::vector<orm_rank_t> const& Dstorm::netSendVec( IoNet_t const ionet ) const {
        assert( ionet < this->iographs.size() );
        return this->iographs[ ionet ].send();
    }
    inline std::vector<orm_rank_t> const& Dstorm::netRecvVec( IoNet_t const ionet ) const {
        assert( ionet < this->iographs.size() );
        return this->iographs[ ionet ].recv();
    }

    inline std::vector<Tnode> const& Dstorm::segSendVec(SegNum seg) const {
        try {
            return iographs[ getSegInfo(seg).ionet ].send();
        }catch( std::exception& e ){
            // iographs code is independent of transport code (like older transports)
            ORM_COUT(this->orm, e.what());
            throw;
        }
    }
    // ??? seginfos:segInfoMapType on GPU?
    inline std::vector<Tnode> const& Dstorm::segRecvVec(SegNum seg) const {
        try{
            //return iographs[ getSegInfo(seg).ionet ].recv();
            return iographs[ getSegInfo(seg).ionet ].recv();
        }catch( std::exception& e ){ // iographs doesn't know how we want to print...
            ORM_COUT(this->orm, e.what());
            throw;
        }
    }
}//dStorm::
#endif // DSTORM_ANY_HH_

