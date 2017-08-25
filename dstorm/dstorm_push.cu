/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#include "dstorm.hpp"           // cpu/gpu now agree on a common Dstorm API
#if !WITH_GPU
#error "WITH_GPU compile flag must be set, so Dstorm:: gpu functions get declared"
#endif
#include "dstorm_any.hh"        // temporary inline header code common to both cuda and host-side gpu
#include "dstorm_any2.hh"       // need dstorm_any + segImpl.hh
#include "segImpl.hh"           // SegImpl + SegInfo + segVecGpu inlines and template fns

#include "demangle.hpp"         // just for debug messages (is this even ok for cuda ??)

namespace dStorm {

    ssize_t Dstorm::push_impl_gpu( SegNum const s, uint32_t const which )
    {
        int const verbose=0;
        if(verbose>1) ORM_COUT(this->orm, " Entering Dstorm::push_impl_gpu( segnum="<<s<<", which="<<which<<" )");
        SegInfo & sInfo = validSegInfo(s,__func__); // throw if non-existent
        SegPolicy const segsync = sInfo.policy & SEGSYNC_MASK;

        {// Sanity checks, and set 'acks' if we need it
            assert( this->ipc == nullptr );
            if( transport != GPU ){
                throw std::runtime_error("push_impl_gpu called without transport == GPU!");
            }
            if( !( segsync==SEGSYNC_NONE || segsync==SEGSYNC_NOTIFY ))
                // remaps NOTIFY to full barrier, NOTIFY_ACK not supported
                throw std::domain_error("GPU push supports only SEGSYNC_NONE or SEGSYNC_NOTIFY ???");
            if (verbose>1) {
                if( this->iProc == 0U ){
                    this->print_seg_info(s);
                }
            }
        }

        uint_least32_t result;
        uint_least8_t pushes;
        auto obufHdr = mem_as<MsgHeader<seg::Dummy>*>( sInfo.ptrObuf() );
        {
            uint_least32_t *d_result;
            uint_least8_t *d_pushes;
            CudaCheckError();
            checkCudaErrors(cudaMalloc((void**)&d_result, sizeof(uint_least32_t)));
            checkCudaErrors(cudaMalloc((void**)&d_pushes, sizeof(uint_least8_t)));

            //ORM_COUT(this->orm, "About to call push_init " << obufHdr << s << d_result << d_pushes);
            //ORM_COUT(this->orm, "About to call push_init ");
            //push_init<<<1,1>>>(obufHdr, s, d_result, d_pushes);
            push_init<<<16, 16>>>(obufHdr, s, d_result, d_pushes);
           
            CudaCheckError();
            checkCudaErrors(cudaMemcpy(&result, d_result, sizeof(uint_least32_t), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(&pushes, d_pushes, sizeof(uint_least8_t), cudaMemcpyDeviceToHost));
            //  OHOH: d_result ... were never freed -- they should really be allocated JUST ONCE XXX
            cudaFree(d_result);
            cudaFree(d_pushes);
        }
        if( pushes > 0U ){
            ORM_COUT(this->orm, " duplicate Dstorm::push");
            return 0;
        }
        if(verbose>2) ORM_COUT(this->orm,  "trace 000");

        uint32_t nWrite = 0U;
        std::vector<Tnode> send_list = iographs.send(sInfo.ionet); //send_lists[ sInfo.ionet ];
        std::vector<unsigned short> send_bufnum = send_bufnums[ sInfo.ionet ];
        uint32_t nErr = 0U;
        std::ostringstream oss;
        assert( which==-1U || which < send_list.size() );
        uint32_t const sndBeg = (which==-1U? (uint32_t)0U : which);
        uint32_t const sndEnd = (which==-1U? static_cast<uint32_t>(send_list.size()): which+1U);
        //ORM_COUT(this->orm, "send list begin: "<<sndBeg << ", send list end: " << sndEnd << "\n");
        orm->win_start(orm, sInfo.seg_id);
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
                orm->wait(orm, GQUEUE_write,ORM_BLOCK);
            }
            // 2. write (or write_notify) our oBuf --> remote rBuf
            if( WITH_LIBORM || segsync == SEGSYNC_NONE ){
                try{
                    if(verbose>1) ORM_COUT(this->orm, (snd>0?"\n\t\t":"  ")<<"write seg "<<(unsigned)sInfo.seg_id<<" r"<<iProc
                                             <<" --> remote r"<<send_list[snd]<<" rbuf "<<remotebuf << " with bufBytes: " << result << " local offset: "<< sInfo.obufnum()*sInfo.bufBytes << "remote offset: " << remotebuf * sInfo.bufBytes  << " obufnum:" << sInfo.obufnum() <<" bytes per buf:" << sInfo.bufBytes);
                    // MPI_FLOAT undefined ??? << " size of MPI_FLOAT: " << sizeof(MPI_FLOAT));
#define OrmWrite(...) orm->write(orm, __VA_ARGS__ )
                    OrmWrite( sInfo.seg_id, sInfo.obufnum() * sInfo.bufBytes,
                                send_list[snd],
                                sInfo.seg_id, remotebuf * sInfo.bufBytes,
                                result,//obufHdr->hdr.a.bytes,
                                GQUEUE_write, ORM_BLOCK );
#undef OrmWrite
                }catch(orm_error &e){
                    ORM_COUT( this->orm, " **WARNING** Dstorm::push write failure\n" );
                    ++nErr; // failed write is non-fatal, trigger netRecover after write loop
                }
            }
        } // loop INITIATING the sending of oBuf to remote ranks
        orm->win_complete(orm, sInfo.seg_id);
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

        }

        if(verbose>0) ORM_COUT(this->orm, oss.str());
        if(verbose>2) ORM_COUT(this->orm, "trace 004");
        // receiver: DO NOT WORRY if "pushes" field does not
        // match value in MsgTrailer,

        push_con<<<1,1>>>(obufHdr,nWrite);

        //obufHdr->hdr.a.pushes += (1U + (nWrite>0U?1U:0U) ); // a bool would have been ok
        if( nErr ){
            ORM_COUT(this->orm, " push: mpi_write FAILED, nErr = "<<nErr);
        }else{
            if(verbose>1) ORM_COUT(this->orm, " push: mpi_write OK");
        }

        if(nErr) this->netRecover();
        // Hmmm. perhaps keep this. Might be useful for stats:
        // (Note; at this point, do NOT know whether the data
        //        has been transfered or not.  We never get to know that.
        //        Even if queue size is zero, NIC may still be xfering!
        if(verbose>2) ORM_COUT( this->orm, "trace 007");

        return nErr
            ?  -(ssize_t)nErr
            :  result * nWrite ;
        //:  (ssize_t)obufHdr->hdr.a.bytes * (nWrite /*-nErr*/ );
    } //push

}//dStorm::
