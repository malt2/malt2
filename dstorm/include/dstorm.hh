/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef DSTORM_HH_
#define DSTORM_HH_

#include "dstorm_any.hh"
#include "dstorm_any2.hh"       // need dstorm_any + segImpl.hh

#include "dstorm_msg.hh"        // for store, push, reduce
#include "segInfo.hh"
#include "segImpl.hh"
#if WITH_SHM
#include "shm/shormOps.hpp"     // for SHM config structure, ShormConf
#endif

#include "demangle.hpp"         // just for debug

extern "C" {
    /** dstorm catches many error signals (NOT calling previous ones) */
    void dstorm_terminate(int signum);
}

namespace dStorm {

    namespace detail {

#if WITH_NOTIFYACK
        DSTORM_INLINE SegSenderAcks::SegSenderAcks( SegInfo const& s )
            : state( s.outdegree_max, ACKED )
              , nSending(0U)
              , seg_id( s.seg_id )
              , ntBeg( s.rbufNotifyAckBeg() )
              , ntNum( s.rbufNotifyAckEnd() - ntBeg )
              , t_ack(0U)
        {
            assert( ntNum == s.outdegree_max );
        }
#endif

    }//detail::

#if WITH_NOTIFYACK // need transport_notify for this
    DSTORM_INLINE ssize_t
        Dstorm::push( SegNum const s, NotifyEnum const done, uint32_t const snd /*= -1U*/ )
        {
            if( done == NTF_SELECT ){ // NEW push semantics to allow push via "only ONE out-edge"
                return this->push_impl(s,snd);
            }
            else if( done == NTF_RUNNING ){
                SegInfo & sInfo = validSegInfo(s,__func__); // throw if non-existent
                sInfo.getReduceState().reset();
                DST_COUT(" push("<<s<<",NTF_RUNNING):stream-reset");    // for now, rarely used.
                assert( streamCheckEof( sInfo ) == false );
                return 0U;
            }
            else if( done != NTF_DONE ){
                throw std::runtime_error(" push(SegNum, NTF_huh?, snd)");
            }
            {
                // original "push" extension only existed for the 'end-of-stream' handshake
                SegInfo & sInfo = validSegInfo(s,__func__); // throw if non-existent
#ifndef NDEBUG
                SegPolicy const reduce = sInfo.policy & REDUCE_OP_MASK;
                assert( reduce == REDUCE_STREAM );
#endif
                std::vector<Tnode> const& send_list = iographs.send(sInfo.ionet); //send_lists[ sInfo.ionet ];
                assert( snd < send_list.size() || snd == -1U );
                std::vector<orm_rank_t>& send_bufnum = send_bufnums[ sInfo.ionet ];
                // What is send_bufnums[] ?              ^^^^^^^^^^^^
                //  "r" such that for dst = src:send_list[s], dst::recv_list[r] == src
                //  i.e. the remote rbuf to which we write

                // What span of sendlist entries are now DONE?
                uint32_t sndbeg = snd;
                uint32_t sndend = snd+1U;
                if( snd == -1U ){
                    sndbeg = 0U;
                    sndend = send_list.size();
                }else{
                    assert( snd < send_list.size() );
                }

                for(uint32_t xsnd=sndbeg; xsnd<sndend; ++xsnd){
                    if( ! this->net->live(send_list[xsnd]) ) {    // skip dead recipients
                        throw std::runtime_error(" TODO: REDUCE_STREAM lacks code to handle EOF with dead nodes");    // TODO
                    }
                    assert( xsnd < send_list.size() );
                    // XXX If easy to do, also throw if it looks as if store has been called without a real push,
                    //     because we do not want to shut down with "real data" in the oBuf.
                    //     This MIGHT be chackable in oBuf hdr ??? Not sure.
                    orm_notification_id_t ntRemote = sInfo.rbufNotifyBeg() + send_bufnum[xsnd];
                    if(NOTACK_DBG) DST_COUT(" push-done: r"<<iProc<<" seg "<<s<<" xsnd="<<xsnd<<" ntRemote="<<ntRemote<<" sending("<<xsnd<<")");
                    NEED(orm_wait(GQUEUE_write,ORM_BLOCK));
                    orm_number_t qsz;     NEED( orm_queue_size( 0, &qsz ));
                    assert( qsz == 0U );  // seems to hold true
                    NEED(orm_notify( sInfo.seg_id, send_list[xsnd],
                                       ntRemote, orm_notification_t(done), // nonzero notification value
                                       GQUEUE_write, ORM_BLOCK ));
                    NEED(orm_wait(GQUEUE_write,ORM_BLOCK));
                    //if(NOTACK_DBG) DST_COUT(" push-done: r"<<iProc<<" seg "<<s<<" ntRemote="<<ntRemote<<" sending("<<xsnd<<")");
                    this->segacks[s]->sending(xsnd);     // <--- register: 'xsnd' has a non-ACKed write
                }
                return 0U;
            }
        }
    DSTORM_INLINE detail::SegSenderAcks&
        Dstorm::validSegSenderAcks( SegNum const s ) const
        {
            SegInfo & sInfo = validSegInfo(s,__func__); // throw if non-existent
            SegPolicy const segsync = sInfo.policy & SEGSYNC_MASK;
            if( segsync != SEGSYNC_NOTIFY_ACK )
                throw std::domain_error("Dstorm::wait(Segnum,timeout) requires SEGSYNC_NOTIFY_ACK");
            auto found = this->segacks.find(s);
            if( found == this->segacks.end() ) throw std::runtime_error("expected SegSenderAcks NOT FOUND");
            if( found->second == nullptr ) throw std::runtime_error("expected SegSenderAcks was a nullptr");
            return *found->second;
        }
#endif
#if !WITH_GPU && WITH_NOTIFYACK
    DSTORM_INLINE uint32_t
        Dstorm::wait( SegNum const s, orm_timeout_t const timeout_ms )
        {
            detail::SegSenderAcks & acks = validSegSenderAcks(s);
            if(NOTACK_DBG) DST_COUT( "NotAck: Dstorm::wait(SegNum="<<s<<", timout="<<timeout_ms<<") awaiting acks...");
            uint32_t const nSending = acks.wait(timeout_ms);
            return nSending;
        }
#endif
#if WITH_NOTIFYACK
    DSTORM_INLINE orm_cycles_t
        Dstorm::ackTicks( SegNum const s )
        {
            detail::SegSenderAcks & acks = validSegSenderAcks(s);
            return acks.ackTicks();
        }
#endif

    DSTORM_INLINE ssize_t
        Dstorm::push_impl( SegNum const s, uint32_t const which )
        {
#if WITH_GPU
            if( this->transport == GPU )
                return this->push_impl_gpu(s, which);
            else
#endif
                return this->push_impl_host(s,which);
        }
}//dStorm::
//#endif // not using dstorm.cuh
#endif // DSTORM_HH_
