/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
/****************************************************************
 * The one-sided RDMA programming interface. The number of
 * workers are decided during run-time with a config-file.
 *
 *
 *
 * **************************************************************/
#include "dstorm.hh"            // api AND support structures

#include "detail/dstormIPC.hpp"

using namespace std;

namespace dStorm {

    ssize_t Dstorm::push_impl_host( SegNum const s, uint32_t const which )
    {
        int const verbose=0;

        if(verbose>1) ORM_COUT(this->orm, " Entering Dstorm::push( segnum="<<s<<" )");

#if WITH_LIBORM
        if( orm == &orm_shm ){
            throw std::runtime_error("Hmmm. do i even do anything for shm Dstorm push?");
        }
#else
        detail::SegSenderAcks *acks = nullptr;
#endif

        SegInfo & sInfo = validSegInfo(s,__func__); // throw if non-existent
        SegPolicy const segsync = sInfo.policy & SEGSYNC_MASK;
        {// Sanity checks, and set 'acks' if we need it
#if WITH_LIBORM
            if( !( segsync==SEGSYNC_NONE || segsync==SEGSYNC_NOTIFY ))  // remaps NOTIFY to full barrier, NOTIFY_ACK not supported
                throw std::domain_error("Unsupported SEGSYNC_XXX value (WITH_LIBORM only supports SEGSYNC_NONE)");
#else
            if( !( segsync==SEGSYNC_NONE || segsync==SEGSYNC_NOTIFY || segsync==SEGSYNC_NOTIFY_ACK) )
                throw std::domain_error("Unsupported SEGSYNC_XXX value");
            assert( this->segacks.find(s) != this->segacks.end() );
            if( segsync == SEGSYNC_NOTIFY_ACK ){ // throw if segacks looks bad
                auto found = this->segacks.find(s);
                if( found == this->segacks.end() ) throw std::runtime_error("expected SegSenderAcks NOT FOUND");
                if( found->second == nullptr ) throw std::runtime_error("expected SegSenderAcks was a nullptr");
                acks = found->second;
                assert( acks != nullptr );
                if(NOTACK_DBG) DST_COUT( "NotAck: push initial state "<<acks->str() );
            }else{
                assert( this->segacks[s] == nullptr );
            }
#endif
        }

        // TODO if we are the unique rank on this machine
        //  AND if we are single-threaded (ORM_NUM_THREADS, perhaps, someday?)
        //  THEN we can skip this lock
        //  ... OR we can do orm_wait a lot and intersperse the writes from various ranks?
        // New: this lock is only needed for older transports, so create with defer_lock, then lock if nec.
        namespace bip = boost::interprocess;
        assert( this->ipc != nullptr );
        bip::scoped_lock< bip::named_mutex > push_one_at_a_time(this->ipc->dstorm_push_mutex, bip::defer_lock);
        // - OMPI there should be issues (hopefully)
        // - GPU version has disabled the IPC lock
        //   each think there's enough room on the queue, and then collectively overflow it.
        //

        auto obufHdr = mem_as<MsgHeader<seg::Dummy>*>( sInfo.ptrObuf() );
        if(verbose>1) ORM_COUT(this->orm, "Dstorm::push("<<demangle(typeid(+sInfo).name())<<" s="<<s<<")"
                                 //FMT::name<<">()"
                                 <<", iter="<<obufHdr->hdr.a.iter
                                 <<", bytes="<<obufHdr->hdr.a.bytes
                                 <<", fmt="<<(unsigned)obufHdr->hdr.a.fmt
                                 <<", pushes="<<(unsigned)obufHdr->hdr.a.pushes
                                );
        if( obufHdr->hdr.a.pushes > 0U ){
            ORM_COUT(this->orm, " duplicate Dstorm::push");
            return ORM_SUCCESS;
        }
        if(verbose>2) ORM_COUT(this->orm,  "trace 000");

        uint32_t nWrite = 0U;
        std::vector<Tnode> const& send_list = iographs.send(sInfo.ionet); //send_lists[ sInfo.ionet ];
        std::vector<orm_rank_t>& send_bufnum = send_bufnums[ sInfo.ionet ];
        // What is send_bufnums[] ?              ^^^^^^^^^^^^
        //  "r" such that for dst = src:send_list[s], dst::recv_list[r] == src
        uint32_t nErr = 0U;
        std::ostringstream oss;
        // If send_list size is "too big" (cf.  max queue size)
        // then we will insertwait's into the following write loop:

        assert( which==-1U || which < send_list.size() );
        uint32_t const sndBeg = (which==-1U? uint32_t{0U}                           : which);
        uint32_t const sndEnd = (which==-1U? static_cast<uint32_t>(send_list.size()): which+1U);
#if WITH_LIBORM
        // NEW
        orm->win_start( orm, sInfo.seg_id );
        //
#endif
        for (size_t snd = sndBeg; snd < sndEnd; ++nWrite, ++snd) {
            if(verbose>1) ORM_COUT( this->orm, "trace 003 send_list["<<snd<<"] = "<<send_list[snd]);
            if( ! this->net->live(send_list[snd]) ) {    // skip dead recipients
                continue;
            }
            uint32_t remotebuf = sInfo.rbufBeg()     // in SegBase, common to all nodes
                /*            */ + send_bufnum[snd];   // OUR index in DEST recv_list
            assert( remotebuf >= sInfo.rbufBeg() );
            if(verbose>1) ORM_COUT( this->orm, "trace 003 with remotebuf "<<remotebuf
                                      <<" push send_bufnum[snd]="<<send_bufnum[snd]<<", so obuf --> remotebuf "
                                      <<remotebuf<<" offset 0x"<<std::hex<<remotebuf*sInfo.segBytes<<std::dec);
            { // 1. check queue free entries
#if WITH_LIBORM
                NEED(orm->wait(orm,GQUEUE_write,ORM_BLOCK));
#endif
            }
            // 2. write (or write_notify) our oBuf --> remote rBuf
            if( WITH_LIBORM || segsync == SEGSYNC_NONE ){
                try{
                    if(verbose>1) oss<<(snd>0?"\n\t\t":"  ")<<"write seg "<<(unsigned)sInfo.seg_id<<" r"<<iProc
                        <<" --> remote r"<<send_list[snd]<<" rbuf "<<remotebuf;
#if WITH_LIBORM
#define OrmWrite(...) orm->write(orm, __VA_ARGS__ )
#endif
                    NEED( OrmWrite( sInfo.seg_id, sInfo.obufnum() * sInfo.bufBytes,
                                      send_list[snd],
                                      sInfo.seg_id, remotebuf * sInfo.bufBytes,
                                      obufHdr->hdr.a.bytes,
                                      GQUEUE_write, ORM_BLOCK ));
#undef OrmWrite
                }catch(orm_error &e){
                    ORM_COUT( this->orm, " **WARNING** Dstorm::push write failure\n" );
                    ++nErr; // failed write is non-fatal, trigger netRecover after write loop
                }
            }
        } // loop INITIATING the sending of oBuf to remote ranks
#if WITH_LIBORM
        // NEW
        orm->win_complete(orm, sInfo.seg_id);
        //
#endif
        //
        // At this point it is possible to reduce the receipt of mixed-version
        // buffers at out-vertices by delaying the sender before he begins to
        // modify his obuf.
        if(0){
            // If settings above QPAUSE_NONE are necessary, you probably have a bug
            // somewhere else.
            //
            // Note: even with full barrier, you still get mixed-version vectors (rarely)
            // To reduce them a lot, you can try SEGSYNC_NOTIFY, or better, SEGSYNC_NOTIFY_ACK.
            // TBD: SEGSYNC_NOTIFY_ACK2 to ensure 100% squash of mixed-version
            if( WITH_LIBORM || segsync == SEGSYNC_NONE ){
                // SEGSYNC_NONE should always be a no-op here (hogwild-ish)
                this->wait( QPAUSE_NONE, GQUEUE_write, ORM_BLOCK );
            }
#if ! WITH_LIBORM
            else if( segsync == SEGSYNC_NOTIFY ){
                this->wait( QPAUSE_NONE, GQUEUE_write, ORM_BLOCK );
            }else if( segsync == SEGSYNC_NOTIFY_ACK ){
                this->wait( QPAUSE_NONE, GQUEUE_write, ORM_BLOCK );
            }
#endif
        }

        if(verbose>0) ORM_COUT(this->orm, oss.str());
        if(verbose>2) ORM_COUT(this->orm, "trace 004");
        // receiver: DO NOT WORRY if "pushes" field does not
        // match value in MsgTrailer,
        obufHdr->hdr.a.pushes += (1U + (nWrite>0U?1U:0U) ); // a bool would have been ok
        if( nErr ){
            ORM_COUT(this->orm, " push: transport write FAILED, nErr = "<<nErr);
        }else{
            if(verbose>1) ORM_COUT(this->orm, " push: orm_write OK");
            // ... oops, forgot Liveness detection ...
            //   orm_wait seems like a bad idea -- it waits for data of
            //   all ported write requests have arrived at the remote side
            //   (but not necessarily all the data), It means write to local
            //   source won't affect data placed in the remote target location.
            //   INSTEAD of here, it seems this should occur BEFORE we [transport]_write.

        }

        if(verbose>2) ORM_COUT( this->orm, "trace 005");
        if(nErr) this->netRecover();
        // Hmmm. perhaps keep this. Might be useful for stats:
        // (Note; at this point, do NOT know whether the data
        //        has been transfered or not.  We never get to know that.
        //        Even if queue size is zero, NIC may still be xfering!
        if(verbose>2) ORM_COUT( this->orm, "trace 007");
        return nErr
            ?  -(ssize_t)nErr
            :  (ssize_t)obufHdr->hdr.a.bytes * (nWrite /*-nErr*/ );
    }

}//dStorm::
