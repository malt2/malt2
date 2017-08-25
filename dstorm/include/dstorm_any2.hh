/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef DSTORM_ANY2_HH
#define DSTORM_ANY2_HH

#include "dstorm_any.hh"

// we additionally have access to SegImpl, FMT, ... in this header
#include "segImpl.hh"
#include "demangle.hpp"

#include <unistd.h>             // sleep

namespace dStorm {

    template< typename FMT, typename... Args >
        void Dstorm::add_segment( SegNum const s,
                                  IoNet_t const ionet,
                                  SegPolicy const policy,
                                  uint32_t const cnt,
                                  Args&&... args )
        {
            int const verbose=0;
            static_assert( is_segment_format<FMT>::value,
                           "ERROR: require is_segment_format<FMT>");
            using detail::SegImpl;
            {
                // XXX scoped lock if multithreaded
                ORM_NEED( seginfos.find( segKey<FMT>(s) ) == seginfos.end() );
                if(verbose){
                    if(verbose>1 || (verbose==1 && iProc==0)){
                        SegPolicyIO policyIO{policy};
                        ORM_COUT(this->orm, "Dstorm::add_segment(SegNum="<<(unsigned)s<<", ionet="<<(unsigned short)ionet
                                   <<":"<<(ionet < this->iographs.size()? this->iographs[ionet].shortname(): std::string("HUH?"))
                                   <<", policy="<<dStorm::name(policyIO)<<", cnt="<<cnt<<",...)");
                    }
                    //this->barrier(300000);
                    ORM_COUT(this->orm, " r"<<iProc<<" seg"<<s<<" sendList "<<this->print_sync_list( ionet ));
                }

                if(verbose>1) ORM_COUT(this->orm, "Dstorm::add_segment -- new SegImpl...");
                // Oh. we call SEGIMPL constructor AND SEGIMPL::add_segment.
                // So who gets the arg pack?    A: SegImpl constructor
                SegImpl<FMT>* segimpl = new SegImpl<FMT>( std::forward<Args>(args)... );
                assert( segimpl != nullptr );
                SegInfo *seginfo = segimpl;   // cast all-the-way-down
                // These depend on FMT, and were moved from segimpl_add_segment
                // XXX Can segimpl_add_segment, whose code is now indpt of FMT,
                //     become a non-templated Dstorm helper function ??? XXX
                seginfo->sizeofMsgHeader = sizeof( MsgHeader<FMT> );
                //seginfo->fmtValue = FMT::value; // <-- this const value is set during 'add_segment'

                if(verbose>1) ORM_COUT(this->orm, "Dstorm::add_segment -- segimpl->add_segment(s,this,ionet,cnt) type "<<typeid(seginfo).name());
                // next call invokes SegVec...::setSegInfo, finishes SegInfo setup, and invokes SegVec...:add_segment
                segimpl->segimpl_add_segment( s, this, ionet, policy, cnt );

                this->barrier();        // give time for internal messages to propagate
 
                { // throw if SegNum s already there
                    SegInfoMapType::iterator found = this->seginfos.find(s);
                    if( found != seginfos.end() ){
                        ORM_COUT(this->orm, " ERROR: add_segment( "<<s<<" ): segment already exists");
                        throw std::runtime_error("add_segment: segment already exists");
                    }
                }
                this->seginfos[ s ] = seginfo;          // point SegNum s at its SegInfo base class
#if WITH_NOTIFYACK
                // support structure for SEGSYNC_NOTIFY_ACK
                this->segacks[(SegNum)s] = (((policy&SEGSYNC_MASK) == SEGSYNC_NOTIFY_ACK )
                    ? new detail::SegSenderAcks( *seginfo )
                    : (detail::SegSenderAcks*)nullptr );
                if( (policy&REDUCE_OP_MASK) == REDUCE_STREAM ){
                    // we will count NTF_DONE notificiations for the set of rbufs.
                    seginfo->reduceState = new detail::AwaitCountedSet( seginfo->rbufNotifyEnd() - seginfo->rbufNotifyBeg() );
                }
#endif
            }
            {
                // testing...
                SegInfo *sInfo = this->seginfos[ s ];
#if !defined(__CUDA_ARCH__)
                if( verbose>1 || (verbose==1 && iProc==0U))
                    ORM_COUT(this->orm, " TEST: +(*seginfo) --> "<<demangle(typeid(+(*sInfo)).name()) );
#endif
                if(verbose>1 && iProc==0U) this->print_seg_info( s );
            }
#if WITH_LIBORM // --> run-time, if this->transport==OMPI or GPU ??
            std::vector<Tnode> send_list = segSendVec(s); 
            std::vector<Tnode> recv_list = segRecvVec(s); 
            Tnode* send_list1 = &send_list[0];
            Tnode* recv_list1 = &recv_list[0];
            int* sendlist = new int[send_list.size()];
            int* recvlist = new int[recv_list.size()];
            std::copy(send_list1, send_list1+send_list.size(), sendlist);
            std::copy(recv_list1, recv_list1+recv_list.size(), recvlist);
	    if(verbose) ORM_COUT(this->orm, "group_create_mpi...");
            orm->group_create_mpi( orm, s, &sendlist[0], send_list.size(), &recvlist[0], recv_list.size()); 
	    if(verbose) ORM_COUT(this->orm, "win_post...");
            orm->win_post(orm, s);
            if(sendlist) delete[] sendlist;
            if(recvlist) delete[] recvlist;
#endif
            // moved to Seg_VecGpu<T>::add_segment (Impl::add_segment call)
            //if( this->transport == GPU ){
                // 1. cudaMalloc and set segInfo->segInfoGpu
                // 2. copy sInfoPod from host to device
            //}
            this->barrier();    // add_/delete_segment are rare, and must NEVER overlap

        }//Dstorm::add_segment<FMT,Args...>

    inline void Dstorm::delete_segment( SegNum const s ) // non-const because of seginfos erase
    {
        this->barrier();    // add_/delete_segment are rare, and must NEVER overlap
        int const verbose=1;
        SegInfo & sInfo = validSegInfo(s,__func__);
        if(verbose>1 || (verbose==1 && iProc==0))
            ORM_COUT(this->orm, "Dstorm::delete_segment("<<s<<"), type "<<demangle(typeid(+sInfo).name()));

#if WITH_LIBORM
        orm->group_delete_mpi(orm, sInfo.seg_id);
        orm->segment_delete(orm, sInfo.seg_id );       
#endif
        sInfo.delete_segment();         // SegVecFoo cleanup
        //                                 equiv. (+sInfo).delete_segment();
        // Seg_VecGpu::delete_segment will delete gpu version of SegInfoPod

        delete &sInfo;                 // DESTROY our top-level segment object
        seginfos.erase(s);           // and remove our pointer to it's SegInfo
        this->barrier();

    }//Dstorm::delete_segment

    template< typename FMT, typename IN_CONST_ITER > inline
        void Dstorm::store( SegNum const s,
                            IN_CONST_ITER iter, uint32_t cnt,
                            uint32_t offset/*=0U*/, double const wgt/*=1.0*/ ) const
    {
        static_assert( is_segment_format<FMT>::value,
                       "ERROR: require is_segment_format<FMT>");
        // ASSUME iter valid for cnt + offset items
        // iter::value_type must be convertible to SegTag<FMT>::type's data type
        // TODO check above assumptions every time

        int const verbose=0;
        typedef typename std::iterator_traits<IN_CONST_ITER>::value_type T;
        static_assert( std::is_arithmetic<T>::value || std::is_pod<T>::value,
            "Dstorm::store IN_CONST_ITER::value_type must be POD" );
        // gcc-4.8.2 still has not implemented this type trait
        //static_assert( std::is_trivially_copyable<T>()::value,
        //    "Dstorm::store IN_CONST_ITER::value_type must be trivially copyable" );
        // so require something more strict (above)
        //static_assert( std::is_const<T>::value == true,
        //    "Dstorm::store IN_CONST_ITER::value_type must be const (paranoia)" );
        if(verbose) ORM_COUT(this->orm, " Dstorm::store<"<<FMT::name<<">"
                               <<"( "<<type_str<IN_CONST_ITER>()<<" iter"
                               <<", cnt="<<cnt<<", offset="<<offset<<" wgt="<<wgt<<" )" );
        SegInfo & sInfo = validSegInfo(s,__func__); // throw if non-existent or invalid
#if WITH_NOTIFYACK
        // NOTIFY_ACK support
        //    NOTIFY only done during 'push'
        //    REDUCE_AVG_RBUF_OBUF modifies oBuf, during reduce, so await acks there.
        //    Others should await here, in 'store' write to oBuf, if not earlier (what function call?)
        //    Should be safe to always await acks here "just in case" ...
        SegPolicy const segsync = (sInfo.policy & SEGSYNC_MASK);
        // SegPolicy const red = sInfo.policy & REDUCE_OP_MASK;
        if( // red==REDUCE_AVG_RBUF_OBUF &&
            segsync==SEGSYNC_NOTIFY_ACK )
        {
            auto found = this->segacks.find(s);
            if( found == this->segacks.end() ) throw std::runtime_error("expected SegSenderAcks NOT FOUND");
            if( found->second == nullptr ) throw std::runtime_error("expected SegSenderAcks was a nullptr");
            detail::SegSenderAcks *acks = found->second;
            if(NOTACK_DBG) DST_COUT( "NotAck: store state "<<acks->str()<<", segacks["<<s<<"]->wait()...");
            // await ACKs for whole SET of successful <tt>write_notify</tt>s
            uint32_t const w = acks->wait(ORM_BLOCK);

            if( w > 0U ){       // w == 0 means all sends acked
                throw std::runtime_error("store acks->wait failure");
            }
        }
#endif
        auto *sTop = dynamic_cast< detail::SegImpl<FMT>* >(&sInfo);
        //assert( sTop != nullptr ); // give a better message...
        if( sTop == nullptr ){
            std::ostringstream oss;
            oss<<" store<FMT>(SegNum "<<s<<",iter,cnt "<<cnt<<",offset "<<offset<<") wrong FMT for Segnum?";
            throw std::runtime_error(oss.str());
        }
        // sTop can shunt us down to Impl if it has nothing to do (likely)
        sTop->store( iter, cnt, offset, wgt );
    }//Dstorm::store

    template< typename FMT, typename IN_CONST_ITER >
        inline void
        Dstorm::store( SegNum const s, IN_CONST_ITER iter, IN_CONST_ITER const end,
                       uint32_t const offset/*=0U*/, double const wgt/*=1.0*/ ) const
    {
        // ASSUME iter valid for cnt + offset items
        // iter::value_type must be convertible to SegTag<FMT>::type's data type

        int const verbose=0;
        //typedef typename IN_CONST_ITER::value_type T;
        typedef typename std::iterator_traits<IN_CONST_ITER>::value_type T;
        static_assert( std::is_pod<T>::value == true,
            "Dstorm::store IN_CONST_ITER::value_type must be POD" );
        // gcc-4.8.2 still has not implemented this type trait
        //static_assert( std::is_trivially_copyable<T>()::value,
        //    "Dstorm::store IN_CONST_ITER::value_type must be trivially copyable" );
        // so require something more strict (above)
        //static_assert( std::is_const<T>::value == true,
        //    "Dstorm::store IN_CONST_ITER::value_type must be const (paranoia)" );
        if(verbose) ORM_COUT(this->orm, " Dstorm::store<"<<FMT::name<<">"
                               <<"( "<<type_str<IN_CONST_ITER>()<<" iter beg,end"
                               ", offset="<<offset<<", wgt="<<wgt<<" ) NB: CHECK buf ovflw in SegImpl?" );
        //SegInfo * sInfo = this->seginfos.at(segKey<FMT>(s)); // throw if non-existent
        //assert( sInfo != nullptr );
        //assert( sInfo->valid == true );
        SegInfo & sInfo = validSegInfo(s,__func__); // throw on err
        assert( sInfo.valid == true );
        //auto *sTop = sInfo->getSegImpl<FMT>();
        auto *sTop = dynamic_cast< detail::SegImpl<FMT>* >(&sInfo);
        assert( sTop != nullptr );
        // sTop can shunt us down to Impl if it has nothing to do (likely)
        sTop->store( iter, end, offset, wgt );
    }//Dstorm::store

#if 1 //dstorm_any2

    DSTORM_INLINE uint32_t
        Dstorm::reduce( SegNum const s ) const
        {
            int const verbose=0;
            SegInfo & sInfo = validSegInfo(s,__func__);
            SegPolicy const segsync = sInfo.policy & SEGSYNC_MASK;
            SegPolicy const red = sInfo.policy & REDUCE_OP_MASK;
            uint32_t nReduce = 0U;

            auto ackToIter = this->recv_src.end();
            if( segsync == SEGSYNC_NOTIFY_ACK /*|| red == REDUCE_STREAM*/ ){
                // reduce only uses the send queue for ACKs
                ackToIter = this->recv_src.find( sInfo.ionet );
                if( ackToIter == this->recv_src.end() ) throw std::runtime_error("recv_src incomplete");
            }
            assert( red != REDUCE_STREAM );

#if WITH_LIBORM
            //
            // Generic "wait for rdma data"
            //
            // How? According to segsync setting (Set by Orm::segment_create, Orm::sync)
            // TBD copy above code blocks into liborm
            //
            NEED( orm->win_wait(orm, sInfo.seg_id) );
            //
            //
            //
#elif WITH_NOTIFYACK
            /** 0 : increase overlap by sending ack "perhaps a wee tad too soon".
             * - This really separates SEGSYNC_NOTIFY_ACK into two distinct flavors
             * - SAFER_ACK==1 is the "safe" version
             * - SAFER_ACK==0 sends the ACK while the reduction is ongoing
             *   - should \b usually be OK if the calculation
             *     takes longer than the reduction  (perhaps this could be measured,
             *     from time to time, within liborm and auto-adapted)
             */
#define SAFER_ACK 0
            if( segsync==SEGSYNC_NOTIFY || segsync==SEGSYNC_NOTIFY_ACK ){
                // XXX Q: SegSenderAcks::wait_write_notify_all() ?
                // blocking notify_waitsome for ALL rBufs to have arrived
                orm_notification_id_t const ntBeg = sInfo.rbufNotifyBeg();
                orm_number_t          const ntNum = recv.size(); // was sInfo.rbufNotifyEnd() - sInfo.rbufNotifyBeg();
                assert( (unsigned)(sInfo.rbufNotifyEnd() - sInfo.rbufNotifyBeg()) == recv.size() ); // used to hold for dstsgd5
                if(verbose>1) DST_COUT(" r"<<iProc<<" waiting for "<<" ntNum="<<ntNum<<" notifications at ntId "<<ntBeg<<".."<<ntBeg+ntNum-1);
                if( ntNum <= 0U )
                    throw(" Ooops, is this SEGSYNC_NOTIFY with no receive buffers valid? SEG_ONE support might need some code mods!");
                if( this->nDead() ) throw std::runtime_error("reduce: write_notify dead node support TBD");

                std::ostringstream oss;
                // many-to-one receipt of notifications

                for( orm_number_t ntCnt = 0; ntCnt<ntNum; ++ntCnt)
                {// block on ALL write notifications (and certainly their writes) having arrived
                    // NOTE: could also think of sequential reduce, so that the ACK gets sent back ASAP
                    orm_notification_id_t id;         // which notification did we get? in range 0..rcvList.size()-1
                    if( orm_notify_waitsome( sInfo.seg_id, ntBeg, ntNum, &id, ORM_BLOCK ) != ORM_SUCCESS )
                        throw( "SEGSYNC_NOTIFY waitsome failure during reduce" );
                    if(verbose) oss<<(ntCnt>0?"\n\t\t":"  ")<<"r"<<iProc<<" ntCnt "<<ntCnt<<" of "<<ntNum<<" notification id "<<id
                                             <<" (rbuf "<<sInfo.rbufBeg() + (id - sInfo.rbufNotifyBeg())<<")"
                                             <<" wait "<<ntCnt+1U<<"/"<<ntNum;
                    orm_notification_t val = 0;
                    if( orm_notify_reset( sInfo.seg_id, id, &val ) != ORM_SUCCESS )
                        throw( "SEGSYNC_NOTIFY reset failure during reduce" );
                    assert( val == NTF_RUNNING );
                    if( !SAFER_ACK && segsync==SEGSYNC_NOTIFY_ACK )
                    { // immediately send back ACK meaning "write data received"
                        // this is a weak guarantee: modifying obuf after getting the ack
                        // MIGHT still cause mixed-version issues!
                        orm_rank_t const ackTo = recv[id];               // rank ackTo notified us.
                        SenderInfo const& sender = ackToIter->second[id];  // more sender info, about rank ackTo
                        // all ranks use recvlist_size write_notifies, followed by sendlist_size ack-notifies
                        // sendlist_index : which of ackTo's sendlist items point at us.
                        orm_notification_id_t ackId  = sender.recvlist_size + sender.sendlist_index;
                        orm_notification_t    ackVal = orm_notification_t(NTF_ACK);

                        if(NOTACK_DBG) DST_COUT( "NotAck: B reduce orm_notify( seg_id="<<(unsigned)sInfo.seg_id<<" ackTo="<<(unsigned)ackTo
                                  <<" ackId="<<(unsigned)ackId<<" ackVal="<<(unsigned)ackVal
                                  <<" q=0, ORM_BLOCK )");
                        WAIT_GQUEUE_ack_NOT_TOO_FULL;
                        NEED( orm_notify( sInfo.seg_id, ackTo, ackId, ackVal, GQUEUE_ack, ORM_BLOCK ));
                    }
                }
                if(verbose>1) DST_COUT(oss.str());
                else if(verbose==1) DST_COUT(" r"<<iProc<<" "<<ntNum<<" notifications received");
            }
#else
#warning "Warning: using a barrier in Dstorm::reduce(Segnum)"
            this->barrier();
#endif
//#endif
            if(1) // ( WITH_LIBORM || red != REDUCE_STREAM )
            { // Perform the [multiple-rbuf, nonstreaming] reduce:
                if(verbose>1) ORM_COUT(this->orm, "Dstorm::reduce( s="<<s<<" ["<<demangle(typeid(+sInfo).name())<<"])");
                SegPolicy const lay = sInfo.policy & SEG_LAYOUT_MASK;
                if( SEG_FULL == lay ){
                    nReduce = sInfo.segimpl_reduce();
                    if(verbose>1) ORM_COUT(this->orm, "Dstorm::reduce( s="<<s<<") returning nReduce = "<<nReduce);
                }else if( SEG_ONE == lay ){
                    if( red != REDUCE_NOP )
                        throw std::runtime_error("SEG_ONE layout does not support a delayed reduce operation");
                    if(verbose>1) ORM_COUT(this->orm, "Dstorm::reduce( s="<<s<<") REDUCE_NOP with SEG_ONE (no-op) is acceptable");
                }else{
                    throw std::runtime_error("unsupported layout");
                }
            }
#if WITH_LIBORM
            // FIXME: dstorm.cuh had at THIS POINT:
            //
            // if( segsync == SEGSYNC_NOTIFY_ACK /*|| red == REDUCE_STREAM*/ ){
            //    // reduce only uses the send queue for ACKs
            //    ackToIter = this->recv_src.find( sInfo.ionet );
            //    if( ackToIter == this->recv_src.end() ) throw std::runtime_error("recv_src incomplete");
            //}

            //
            // Generic "my rbufs are ready to receive"
            NEED( orm->win_post(orm, sInfo.seg_id) );
            //
#elif WITH_NOTIFYACK
            // ACK ** after ** reduce [above] is MUCH PREFERED (strong guarantee: you can modify your oBuf)
            if( SAFER_ACK && segsync==SEGSYNC_NOTIFY_ACK )
            {
                assert( recv.size() > 1 ); // only if needed should we have delayed ACKing
                for( uint32_t id=0; id<recv.size(); ++id )
                { // delayed send back ACK meaning "write data received AND you can push more to me"
                    orm_rank_t const ackTo = recv[id];               // rank ackTo notified us.
                    SenderInfo const& sender = ackToIter->second[id];  // more sender info, about rank ackTo
                    // all ranks use recvlist_size write_notifies, followed by sendlist_size ack-notifies
                    // sendlist_index : which of ackTo's sendlist items point at us.
                    orm_notification_id_t ackId  = sender.recvlist_size + sender.sendlist_index;
                    orm_notification_t    ackVal = orm_notification_t(NTF_ACK);
                    if(NOTACK_DBG) DST_COUT( "NotAck: C reduce orm_notify( seg_id="<<(unsigned)sInfo.seg_id<<" ackTo="<<(unsigned)ackTo
                                             <<" ackId="<<(unsigned)ackId<<" ackVal="<<(unsigned)ackVal
                                             <<" q=0, ORM_BLOCK )");
                    WAIT_GQUEUE_ack_NOT_TOO_FULL;
                    NEED( orm_notify( sInfo.seg_id, ackTo, ackId, ackVal, GQUEUE_ack, ORM_BLOCK ));
                }
            }
#else
#warning "Warning: using a barrier in dstorm.hh, function Dstorm::reduce"
            this->barrier();
#endif
#undef SAFER_ACK
            return nReduce;
        }//Dstorm::reduce
#undef WAIT_GQUEUE_ack_NOT_TOO_FULL

#endif // dstorm_any2
}//dStorm::
#endif // DSTORM_ANY2_HH
