/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
/****************************************************************
 * The one-sided RDMA programming interface. The number
 * of workers are decided during run-time with a config-file.
 *
 *
 *
 * **************************************************************/
#include "dstorm.hh"            // api AND support structures
#include "dstorm_net.hpp"       // also pull in globIoNet, for test/debug helpers
#include "detail/dstormIPC.hpp" // serialize orm push across whole system UN-NECESSARY?

#include <cstdarg>
#include <iostream>
#include <iomanip>
#include <cstring>      // memcpy, memset
#include <utility>
#include <type_traits>
#include <algorithm>
#include <iterator>

#include <assert.h>

/** stats msgs were getting a bit too long... */
#define VERBOSE_DESTRUCTION 1

/** used within Dstorm, so this->orm can be used for error msg */
#define SIGACTION( SIG, ACT, OLDACT ) do{ \
    if( sigaction( (SIG), (ACT), (OLDACT) ) == -1 ){ \
        ORM_COUT(this->orm, "Error installing sigaction for signal "<<(SIG)); \
    } \
}while(0)

using namespace std;

/** supposed to be in C++11, but apparently got delayed to C++14? */
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

//struct sigaction oldSigStkFlt;

/** dstorm installs a termination handler so output can be seen
 * from any mpi_run machine
 */
void dstorm_terminate(int signum)
{
    //if( signum == SIGSTKFLT && oldSigStkFlt.sa_handler != nullptr) {
    //    // pass-through SIGSTKFLT (other valid ORM uses?)
    //    (*oldSigStkFlt.sa_handler)(signum);
    //}
    // NOTE. output functions are **unsafe**, and if interrupted
    // during an unsafe function, program behavior is UNDEFINED
// FIXME remove segment below
#if 0 
    cout<<" ORM VERSION "<<ORM_MAJOR_VERSION
        <<"."<<ORM_MINOR_VERSION<<"."<<ORM_REVISION
        <<"fault detected. signum = "<<signum
#if defined(_GNU_SOURCE)
        <<" ("<<strsignal(signum)<<")"
#endif
        <<" . stopping."<<endl;
    //pthread_exit(NULL);
#endif
    exit(0);
}

/// \defgroup Typedefs typedefs related to ORM types
//@{
typedef unsigned char orm_state_vector_element;
static_assert( std::is_same< orm_state_vector_element*, orm_state_vector_t >()
               ,"ohoh: orm_state_vector_t may have changed" );
//@}

namespace dStorm {

    IoNet_t ScalGraphs::push_back( Tnode const node, std::unique_ptr<mm2::UserIoNet>&& ptr )
    {
        if( graphs.size() == 255U ){
            std::runtime_error e(" too many graphs\n");
            //orm_printf( e.what() ); // XXX orm...printf
            throw e;
        }
        if( !ptr ){
            std::runtime_error e(" ScalGraphs::push_back( node, empty-ptr)\n");
            //orm_printf( e.what() );
            throw e;
        }
        if( ptr->verts != this->verts ){
            std::runtime_error e(" ScalGraphs::push_back(node,unique_ptr-to-graph) graph size mismatch\n");
            //orm_printf( e.what() );
            throw e;
        }
        graphs.push_back( new WrapNet(node, std::move(ptr)) );
        // even emplace_back requires the WrapNet copy constructor, which is
        // deleted due to internal unique_ptr.
        // Soln: SCALIO==2 stores WrapNet* instead of object, and operator[] derefs as req'd.
        assert( !ptr );     // it has been moved, we no longer own it.
        return static_cast<IoNet_t>( graphs.size()-1U );
        assert("TBD"==nullptr);
    }

    std::string pretty_table( mm2::UserIoNet const* unet )
    {
        return unet->pretty_table(nullptr);
    }
    std::string pretty_table( mm2::GlobNet const& gnet )
    {
        Tnode const n = gnet.size();    // number of nodes
        std::ostringstream oss;
        // 1st pass: count unconnected sources, if empty skip print
        uint32_t unconnected_src = 0U;
        for(uint32_t src=0U; src<n; ++src) {    // uint32_t prints as a number
            //std::vector<Tnode> v( get_send_list( src,n,sync_model,nullptr ));
            if( gnet.send(src).size() == 0U ) ++unconnected_src;
        }

        oss<<"...   "<<gnet.name()<<" ...";
        if( unconnected_src == n ){             // empty graph
            oss<<" EMPTY GRAPH\n";
        }else{
            if( unconnected_src ) oss<<unconnected_src<<" unconnected sources";

            char const symbol_fill    = ' ';
            Tnode const grid = 5U;
            char const symbol_grid    = '.';
            char const symbol_lowprio = 'o';
            oss<<"\n";
            if(0){ //debug
                for( uint32_t i=0U; i<n; ++i ){
                    oss<<" "<<i;
                    //vector<Tnode> v( get_send_list( i,n,sync_model,nullptr ));
                    char c = '{';
                    for(auto x: gnet.send(i)){ oss<<c<<x; c=','; }
                    oss<<'}';
                }
                oss<<"\n";
            }
            if( n > 10U ){ // print tens line
                oss<<"    | ";
                for(uint32_t i=0U, ones=0U, tens=0U; i<n; ++i){
                    if(tens) oss<<tens; else oss<<' ';
                    if(++ones==10U){ ones=0U; if(++tens==10U) tens=0U; }
                }
                oss<<"\n";
            }
            { // print ones line
                oss<<"Src | ";
                for(uint32_t i=0U, ones=0U         ; i<n; ++i){
                    oss<<ones;
                    if(++ones==10U){ ones=0U;                          }
                }
                oss<<"\n";
            }
            { // print top separator line
                oss<<"--- + ";
                vector<char> vc( n, '-' );
                for(auto c: vc) oss<<c;
                oss<<"\n";
            }
            oss<<right;
            for(uint32_t src=0U; src<n; ++src) {    // uint32_t prints as a number
                oss<<setw(3)<<src<<" | ";
                std::vector<Tnode> const& v = gnet.send(src); //( get_send_list( src,n,sync_model,nullptr ));
                // default is background grid with some light guidelines
                vector<char> vc( n, (src%grid==grid-1U? symbol_grid: symbol_fill));
                if( src%grid!=grid-1U ) for(uint32_t i=grid-1U; i<n; i+=grid) vc[i] = symbol_grid;
                char symbol = '0'-1;
                for(Tnode prio=0U; prio<v.size(); ++prio){
                    if( prio<16U ){
                        ++symbol;
                        if(prio==10U) symbol='a';
                    }
                    else if(prio==16U) symbol=symbol_lowprio;
                    vc[ v[prio] ] = symbol;
                }
                for(auto c: vc) oss<<c;
                oss<<"\n";
            }
        }
        return oss.str();
    }
    /** this is an old-style call, using OldIoNetEnum */
    std::string pretty_table(Tnode const n, OldIoNetEnum sync_model)
    {
        // an OLD way to get at some of the built-in io networks
        auto gnet = detail::mkGlobNet( sync_model, n );
        assert( n == gnet->size() );
        std::string tabularIonet = pretty_table( *gnet );
        delete gnet;
        return tabularIonet;
    }
    std::string pretty_table( mm2::ScalNet const& scalnet )
    {
        return pretty_table( scalnet.unet.get() );
    }


    namespace detail {

#if ! WITH_LIBORM
        std::string SegSenderAcks::str(){
            std::ostringstream oss;
            oss<<" SegSenderAcks{";
            for(auto const x: state){
                oss<<((x==ACKED)? ' ': /*SENDING*/'S');
            }
            oss<<"} nSending = "<<nSending<<" / "<<ntNum;
            return oss.str();
        }

        uint32_t SegSenderAcks::wait( orm_timeout_t const timeout /*= ORM_BLOCK*/ )
        {
#define SEGSENDERACKS_DBG 0
            orm_cycles_t t0; NEED( orm_time_ticks( &t0 ));
            //std::ostringstream oss;
            // block on ALL notifications having arrived
            // NOTE 'state' only gives us better error messages, if everything is kosher,
            //      all we need to know is ntSending.
            if(nSending<=0){
                orm_cycles_t t1; NEED( orm_time_ticks( &t1 ));
                t_ack += t1 - t0;
#if SEGSENDERACKS_DBG
                //oss<<"\n\t\t SegSenderAcks::wait no-op (nSending was zero)";
                DST_COUT( "SegSenderAcks::wait no-op (nSending was zero)" );
#endif
                return nSending;
            }
            // XXX this loop might be more cpu-friendly by backing off
            do{
#if SEGSENDERACKS_DBG
                DST_COUT( //(nSending>0?"\n\t\tXX":"YY  ")<<
                          "SegSendAcks::wait for notification range [ "<<unsigned(ntBeg)<<", "<<unsigned(ntBeg+ntNum)<<" )"
                          " nSending "<<nSending<<" of "<<ntNum<<" notification id. "
                          " seg_id "<<(uint32_t)seg_id<<" waiting for "<<nSending<<"/"<<ntNum);
                assert( nSending <= ntNum );
#endif
                orm_notification_id_t id;         // which notification did we get?
                orm_return_t status = ORM_SUCCESS;
                if( (status=orm_notify_waitsome( seg_id, ntBeg, ntNum, &id, timeout )) != ORM_SUCCESS ){
                    if( status == ORM_TIMEOUT ){
                        DST_COUT("Warning: acks NOT all received. Increase timeout from "<<timeout<<" ?");
                        break;
                    }
                    throw( "SEGSYNC_NOTIFY* waitsome failure during reduce" );
                }
                assert( id >= ntBeg && id < ntBeg+ntNum );
                uint32_t snd = id - ntBeg;
#if SEGSENDERACKS_DBG
                DST_COUT(" NotAck: got notify id "<<(uint32_t)id<<", for write to sendlist["<<snd<<"]"<<endl);
#endif

                if( state[snd] != SENDING ) throw("Unexpected ACK");

                orm_notification_t val = 0;
                if( orm_notify_reset( seg_id, id, &val ) != ORM_SUCCESS )
                    throw( "SEGSYNC_NOTIFY reset failure during reduce" );
#if SEGSENDERACKS_DBG
                DST_COUT(" notify_reset old val == "<<(uint32_t)val<<" cf. NTF_ACK="<<NTF_ACK<<" NTF_DONE="<<NTF_DONE);
#endif
                assert( val == NTF_ACK || val == NTF_DONE );

                state[snd] = ACKED;
            }while(--nSending);
            //assert( nSending == 0 );
            orm_cycles_t t1; NEED( orm_time_ticks( &t1 ));
            t_ack += t1 - t0;
#if SEGSENDERACKS_DBG
            DST_COUT( "ack wait DONE, nSending="<<nSending );
#endif
            return nSending;
        }
#undef SEGSENDERACKS_DBG
#endif // ! WITH_LIBORM

        DstormCommInit::~DstormCommInit()
        {
#if WITH_LIBORM
            if( this->orm != nullptr )
            {
                orm_destruct( this->orm );
                const_cast<struct Orm*&>(this->orm) = nullptr;
            }
#endif
        }

        /** This is really constructor code, so we modify our \em const data. */
        void DstormCommInit::finalize( /*TransportEnum const tr*/ )
        {
#if WITH_LIBORM
            std::cout<<" DstormCommInit::finalize -- proc_init... orm @ "<<(void const*)orm<<std::endl;
            // Initialization procedure to start transport
            // valgrind reports many "uninitialized" errors here
            // NOTE: slight modification from previous signature.
            NEED(orm->proc_init (const_cast<Orm*>(orm), ORM_BLOCK));
            //if( tr == SHM ){
            //    std::cout<<this_thread::get_id()<<" DstormCommInit::finalize<SHM> --> WAITING AT BARRIER"<<std::endl;
            //    NEED(orm.barrier (&orm, ORM_GROUP_ALL, ORM_BLOCK));
            //}

            std::cout<<" DstormCommInit::finalize -- proc_init... DONE"<<std::endl;
            orm->printf(orm, " DstormCommInit:: proc_rank=%p proc_num=%p\n",
                        (void*)(orm->proc_rank), (void*)(orm->proc_num));
            /* Rank of this process */
            NEED(orm->proc_rank (orm, const_cast<orm_rank_t *>(&iProc)));
            /* Number of processes (ranks) started by orm_run */
            NEED(orm->proc_num  (orm, const_cast<orm_rank_t *>(&nProc)));

            orm->printf(orm, "DstormCommInit DONE: orm@%p iProc=%u nProc=%u\n",
                        (void*)orm, (unsigned)iProc,(unsigned)nProc);

#endif
        }

        int DstormCommInit::dprintf( const char* fmt, ... )
        {
            int ret=0;
            va_list args;
            va_start( args, fmt );
            char line[1024];
            int nchars;
            // XXX orm does not have a vprintf function
            nchars = vsnprintf( line, 1020, fmt, args );
            va_end(args);
            line[nchars] = '\0';

#if WITH_LIBORM
            if( this->orm != nullptr ){
                ret = this->orm->printf( this->orm, "%s", line); // a puts would be better
            }else{
                throw("Cannot printf through NULL orm");
            }
#else
#warning "really should have  WITH_LIBORM"
            cout<<&line[0]; cout.flush();
#endif
            return ret;
        }

        DstormLiveness::DstormLiveness( DstormCommInit const& dBase )
            :
#if WITH_LIBORM
                orm( *dBase.orm ),
#endif
                avoid_list(dBase.nProc,false)
                    , vec( new orm_state_vector_element [dBase.nProc] )
                    , survivors()
                    , fProc(0U)
        {
            //cout<<"+DstormLiveness: nProc="<<nProc<<" avoid_list.size() = "<<avoid_list.size()<<endl;
            this->set_dead();     // check bad nodes --> avoid_list[]
            if( this->nDead() != 0U ){
                ORM_COUT(&orm," Warning: starting DstormLiveness with "
                           <<avoid_list.size()<<" / "
                           <<dBase.nProc);
            }

            //auto const &al = avoid_list;
            //cout<<" after set_dead, al = {"; for(auto a: al) cout<<" "<<a; cout<<" }"<<endl;
            assert( nDead() == 0U );
            //cout<<" after nDead, al = {"; for(auto a: al) cout<<" "<<a; cout<<" }"<<endl;
            assert( nDeadCount() == 0U );
        }
        DstormLiveness::~DstormLiveness()
        {
            if(VERBOSE_DESTRUCTION) cout<<"-DstormLiveness"<<endl; cout.flush();
            delete[] vec;
            if(VERBOSE_DESTRUCTION) cout<<"-DstormLiveness"<<endl; cout.flush();
        }

        orm_rank_t DstormLiveness::nDeadCount()
        {
            fProc = 0U;
            for(auto a: avoid_list) if( a ) ++fProc;
            return fProc;
        }
        orm_return_t DstormLiveness::barrier( orm_timeout_t const timeout_ms ) const
        {
#if WITH_LIBORM
            if (fProc == 0)
                return orm.barrier(&orm, ORM_GROUP_ALL, timeout_ms);
            else
                return orm.barrier(&orm, survivors, timeout_ms);
#else
//#warning "do not know how to provide barrier op (no-op)"
            return ORM_SUCCESS;
#endif
        }
        void DstormLiveness::setSurvivors()
        {
            if( nDeadCount() > 0U ){
#if WITH_LIBORM
                NEED(orm.group_create(&orm,&survivors));
                for (orm_rank_t i = 0U; i < avoid_list.size(); ++i)
                    if ( live(i) ) NEED(orm.group_add(&orm,survivors, i));
                NEED(orm.group_commit(&orm,survivors, ORM_BLOCK));
#endif
            }
        }

        void DstormLiveness::state_update( bool exactMirror )
        {
            bool const verbose=0;
            // Maybe disable all ranks on the same host (?)
            //    If the common case is that *all* ranks on a host will get an error,
            //    we might want to disable more ranks than orm has noticed are gone.
#if WITH_LIBORM
            NEED(orm.state_vec_get(&orm,vec));
#endif
            orm_rank_t const nProc=avoid_list.size();
            if(verbose) cout<<" state_update, exactMirror = "<<exactMirror<<" nProc=avoid_list.size()="<<nProc<<endl;
            for(orm_rank_t n=0U; n < nProc; ++n) {
//#if WITH_GPU // ???????? TODO should this not always be done?
#if 1
                if (vec[n] != ORM_STATE_HEALTHY){
                    ORM_COUT(&orm,"Problem with node "<<n<<" detected");
                    if(verbose) cout<<"avoid_list["<<n<<"] = true"<<endl;
                    avoid_list[n] = true;
                }else
#endif
                    if( exactMirror ) {
                        if(verbose) cout<<"avoid_list["<<n<<"] = false"<<endl;
                        avoid_list[n] = false;
                    }
            }
            setSurvivors();
        }

        // setLive(r,false) CALLER must also redo build_lists and survivors!
        void DstormLiveness::setLive( orm_rank_t const r, bool const isAlive )
        {
            assert( r < avoid_list.size() );
            avoid_list[r] = isAlive;
            setSurvivors();
        }

        /* recover after a failure. \b NOTE: this blocks, but really
         * it should timeout (with many seconds timeout) to gracefully
         * deal with "unrecoverable" errors. */
        orm_return_t DstormLiveness::recover()
        {
            orm_return_t ret = ORM_ERROR;

            ORM_COUT(&orm,"Enter recover");

            while (ret != ORM_SUCCESS)
            {
#if WITH_LIBORM
                ret = orm.wait(&orm, 0, ORM_BLOCK);
#endif
            }

            ORM_COUT(&orm,"Exit recover with "<<ret);
            return ret;
        }

    }//detail::

    using namespace detail;

    template< TransportEnum TR >
        Dstorm::Dstorm( Transport<TR> const& tr )

        : detail::DstormCommInit( tr )          // invokes the proper orm->proc_init
        , seginfos()                            // std::unordered_map< unsigned, SegInfo* >
        , net(nullptr)
        , ipc(nullptr)
        , iographs(nProc)                       // ScalGraphs with nProc vertices per graph
        , send_bufnums()
        , recv_src()
        , barrier_timeout_ms( DSTORM_BARRIER_TIMEOUT_MS )
        //, maltaction .. set below
    {
        this->net = new detail::DstormLiveness( *this ); // as base class
        //this->ipc = (detail::DstormCommInit::transport == ORM?  new detail::DstormIPC(iProc): nullptr);
        // For now, always create it, just use it as
        // bip::scoped_lock< bip::named_mutex > push_one_at_a_time(this->ipc->dstorm_push_mutex, bip::defer_lock);

//#if !WITH_GPU // FIXME
        if( TR != GPU ){
            this->ipc = new detail::DstormIPC(iProc);
        }
//#endif
        { // NEW: mm2 runtime IoNets --> registered in iographs[]
            using namespace mm2;
            using namespace mm2::user;
            IoNet_t ALL         = iographs.push_back( iProc, unique_ptr<mm2::UserIoNet>(new UsrImpl<dStorm::ALL>::type(nProc)) );
#ifdef NDEBUG
            iographs.push_back( iProc, unique_ptr<mm2::UserIoNet>(new UsrImpl<dStorm::SELF>::type(nProc)) );
            iographs.push_back( iProc, unique_ptr<mm2::UserIoNet>(new UsrImpl<dStorm::CHORD>::type(nProc)) );
            iographs.push_back( iProc, unique_ptr<mm2::UserIoNet>(new UsrImpl<dStorm::HALTON>::type(nProc)) );
            iographs.push_back( iProc, unique_ptr<mm2::UserIoNet>(new UsrImpl<dStorm::RANDOM>::type(nProc)) );
            iographs.push_back( iProc, unique_ptr<mm2::UserIoNet>(new UsrImpl<dStorm::STREAM>::type(nProc)) );
            iographs.push_back( iProc, unique_ptr<mm2::UserIoNet>(new UsrImpl<dStorm::PARA_SERVER>::type(nProc)) );
            //IoNet_t BUTTERFLY   = dStorm::IONET_MAX;    // unimplemented
#else
            IoNet_t SELF        = iographs.push_back( iProc, unique_ptr<mm2::UserIoNet>(new UsrImpl<dStorm::SELF>::type(nProc)) );
            IoNet_t CHORD       = iographs.push_back( iProc, unique_ptr<mm2::UserIoNet>(new UsrImpl<dStorm::CHORD>::type(nProc)) );
            IoNet_t HALTON      = iographs.push_back( iProc, unique_ptr<mm2::UserIoNet>(new UsrImpl<dStorm::HALTON>::type(nProc)) );
            IoNet_t RANDOM      = iographs.push_back( iProc, unique_ptr<mm2::UserIoNet>(new UsrImpl<dStorm::RANDOM>::type(nProc)) );
            IoNet_t PARA_SERVER = iographs.push_back( iProc, unique_ptr<mm2::UserIoNet>(new UsrImpl<dStorm::PARA_SERVER>::type(nProc)) );
            IoNet_t STREAM      = iographs.push_back( iProc, unique_ptr<mm2::UserIoNet>(new UsrImpl<dStorm::STREAM>::type(nProc)) );
            IoNet_t BUTTERFLY   = dStorm::IONET_MAX;    // unimplemented
            // assert Dstorm builtin IoNet_t graph handles match OldIoNetEnum "shorthand" tags:
            assert( ALL         == dStorm::ALL );
            assert( SELF        == dStorm::SELF );
            assert( CHORD       == dStorm::CHORD );
            assert( HALTON      == dStorm::HALTON );
            assert( RANDOM      == dStorm::RANDOM );
            assert( PARA_SERVER == dStorm::PARA_SERVER );
	    assert( STREAM	== dStorm::STREAM );
            assert( BUTTERFLY   == dStorm::IONET_MAX ); // unimplemented
#endif
            cout << " iographs.size() = " << (unsigned)iographs.size() << endl;
            if ( 1 && iProc == 0U ) {
                // well, I don't really like to raw vector dumps of 'pretty()' ...
                cout << " mm2 TESTING:\n" << iographs[ALL].pretty() << endl;
                cout << " mm2 TESTING iographs[HALTON].pretty:\n" << iographs[HALTON].pretty() << endl;
                cout << " mm2 TESTING:\n" << pretty_table( nProc, dStorm::HALTON )<<endl;
                cout << " mm2 TESTING:\n" << pretty_table( getScalNet(dStorm::HALTON).unet.get() )<<endl;
                cout << " mm2 TESTING:\n" << pretty_table( getScalNet(dStorm::HALTON) )<<endl;
                // For user-created IoNet_t ionet handle, a nice way to get the
                // nicer TABULAR output from a dstorm.iographs[ionet] might be:
                //      pretty_table( dstorm.getScalNet( ionet ) )
            }
        }

        memset(&maltaction, 0, sizeof(struct sigaction));

        maltaction.sa_handler = dstorm_terminate;
        SIGACTION(SIGTERM, &maltaction, NULL); /* Kill command */
        // NOTE: "An invalid signal was specified. This will also be generated if an attempt is made to change the action for SIGKILL or SIGSTOP, which cannot be caught or ignored. "
        //SIGACTION(SIGKILL, &maltaction, NULL); /* Kill */
        SIGACTION(SIGSEGV, &maltaction, NULL); /* Segmentation fault */
        SIGACTION(SIGILL,  &maltaction, NULL); /* Illegal instruction or pointer */
        SIGACTION(SIGFPE,  &maltaction, NULL); /* Floating point exception */

        //decides nodes to send and nodes to receive
        build_lists();

        //sync                  REMOVE, because thread initialization is easier without this.
        //net->barrier();
    }

    // FORCE INSTANTIATION of Dstorm constructors into library
#if WITH_MPI
    template Dstorm::Dstorm( Transport<OMPI > const& );
#endif
#if WITH_GPU
    template Dstorm::Dstorm( Transport<GPU > const& );
#endif
#if WITH_SHM
    template Dstorm::Dstorm( Transport<SHM  > const& );
#endif

    void Dstorm::build_lists()
    {
        //int verbose = (get_iProc() == 0? 1: 0);
        int verbose = 0;
        std::ostringstream oss;
        // iographs are now invariant wrt. liveness concept --- always reflect full graph.
        if(verbose) oss << " build_lists send/recv ionets set with live nodes for:" << endl;
        //assert( iographs.size() >= IONET_MAX );
        for ( IoNet_t g = 0U; g < iographs.size(); ++g ) { // user graphs NOT supported yet XXX
            //verbose = (g==6? 2: 0);
            if(verbose) oss << " # " << setw(3) << (unsigned)g << " : " << iographs[g].name() << endl;
            // Note: OUR recv_lists[] is not so important.  Instead, the
            // important thing is where "iProc" is positioned within the
            // recv_lists of all the send_lists[] destNodes.
            // Why? because this offset determines the remote segment
            //      to which we should "push" our data.
            // ... and the notification id we'll use for write_notify on our out-edges
            //     (NID = our index in the recv list of our destination)
            send_bufnums[g].clear();
            {
                auto const& iog = iographs[g];
                for( auto destNode: iog.send() )
                {
                    // find "our node" within recv_list of destNode
                    std::vector<Tnode> const destRecv = mm2::ScalNet::mkRecvList( iog.unet.get(), destNode );
                    auto ourNode = find( destRecv.begin(), destRecv.end(), iProc );
                    if( ourNode == destRecv.end() ){
                        ORM_COUT(orm,"Error for graph "<<(unsigned)g<<" type "<<iog.name());
                        throw runtime_error(" recv_list | send_list mismatch");
                    }
                    send_bufnums[g].push_back( distance(destRecv.begin(), ourNode) );
                    if(verbose>1) oss<<"\tnode "<<iProc<<" is the "<<send_bufnums[g].back()
                        <<"'th element in recv list of destNode "<<destNode<<endl;
                }
            }
            // EACH rank uses recvlist_size NIDs for write_notify,
            //      followed by sendlist_size NIDs for ACKs.
            // so determining how we should ACK back along an in-edge is a bit tougher.
            recv_src[g].clear();
            {
                auto const& iog = iographs[g];
                for( auto sender: iog.recv() ) // sender is SENDING data and maybe notifications TO US
                {
                    // find "our node" within send_list of sender
                    //   Yes, this is not symmetric. Our scalable "net" is a send-oriented one!
                    //     graph--> unique_ptr<UserIoNet> --> UserIoNet* --> send list of sender
                    std::vector<Tnode> const senderSend = iog.unet.get()->mkSend( sender );
                    auto ourNode = find( senderSend.begin(), senderSend.end(), iProc );
                    if( ourNode == senderSend.end() ){
                        ORM_COUT(orm,"Error for graph "<<(unsigned)g<<" type "<<iog.name());
                        throw runtime_error(" send_list | recv_list mismatch");
                    }
                    vector<Tnode> const senderRecv = Graphs::WrapNet::mkRecvList( iog.unet.get(), sender );
                    {
                        SenderInfo senderInfo;
                        senderInfo.sendlist_index = distance(senderSend.begin(), ourNode);
                        senderInfo.sendlist_size  = senderSend.size();
                        senderInfo.recvlist_size  = senderRecv.size();
                        recv_src[g].push_back( senderInfo );
                        if(verbose>1) oss<<"\tnode "<<iProc<<" is the "<<senderInfo.sendlist_index
                            <<"'th element in send list["<<senderInfo.sendlist_size<<"] of sender "
                                <<sender<<" with recvlist["<<senderInfo.recvlist_size<<"]"<<endl;
                    }
                    // OK, so iProc will send an ACK back to rank sender = iographs[g].recv()[i].
                    //     s.t. iProc == sender:iographs[g].send()[ s ],
                    //     where s = iProc:recv_src[g][i].
                    // So... what notification_id (NID) is for the ack?
                    //    senderInfo.recvlist_size      : notifications reserved for sender's in-edges
                    //    + senderInfo.sendlist_index   : our position in the sender's sendlist
                }
            }
        }
        if(verbose) {ORM_COUT(orm, oss.str());ORM_COUT(orm,"----");}
    }

    void Dstorm::netRecover()
    {
        ORM_COUT (this->orm, "errors --> add_dead, net->recover");
        this->net->add_dead(); // do we ever setLive(r,true) again? Ohoh!
        NEED(this->net->recover());
        if( nDead() > 0U ){
            // oh.. this one is non-const?
            build_lists(); // just need to do this once, at end
        }// [ejk] *after* we've determined liveness, do this stuff (just once)
    }

    Dstorm::~Dstorm()
    {
#if 0
        //unordered_map<int, seg_info>::iter iter;
        if( segs().size() ){
            this->barrier(); // ensure destructor is in sync first
            ORM_COUT(orm,"~Dstorm: segs()[0.."<<segs().size()<<") cleanup...");
            for (size_t i = 0; i < segs().size(); ++i)
            {
                if (segs()[i].valid){
                    ORM_COUT(orm," delete segs()["<<i<<"] seg_id "<<segs()[i].seg_id);
                    orm->segment_delete(orm,segs()[i].seg_id);
                }
            }
        }
#endif

        if(VERBOSE_DESTRUCTION) ORM_COUT(orm,"~Dstorm: barrier before orm_proc_term...");
        this->barrier(); // ensure destructor is in sync first

        if(VERBOSE_DESTRUCTION) ORM_COUT(orm,"~Dstorm: orm_proc_term...");
        // TODO: go through this->segimpls[] and cleanly destruct
        //       all the top-level objects of the known SegInfos
        //
        // New: does this need serialization?
        //      We can abuse bip::scoped_lock< bip::named_mutex > (this->ipc->dstorm_push_mutex); to serialize if nec.
#if WITH_LIBORM
        NEED(orm->proc_term (orm,ORM_BLOCK)); /* Shutdown procedure */
#endif

        // NOTE: all non-local orm functions now have UNDEFINED BEHAVIOR
        if(VERBOSE_DESTRUCTION) cout<<"~Dstorm: orm_proc_term DONE";  // ... so NO orm->printf


        // for debugging (only?)
        //ORM_NEED( this->segInfoMap != nullptr );
        //this->segInfoMap->clear();
        //delete this->segInfoMap;
        //this->segInfoMap = nullptr;

        ORM_NEED( this->net != nullptr );
        delete this->net; this->net=nullptr;
//#if !WITH_GPU // FIXME
        if(this->ipc != nullptr){ delete this->ipc; this->ipc=nullptr; }
//#endif
    }

#if 0 // deprecated ... used for tests
    vector<orm_rank_t> Dstorm::get_send_list(int iProc, int nProc, IoNet_t sync_model)
    {
        int const dbg = 0;
        vector<orm_rank_t> send_list;
        if( nDeadCount() > 0U ) {
            orm->printf(orm," get_send_list: avoid_list size is %u, nDead=%u\n"
                         , unsigned(net->avoid_list.size()), unsigned(nDead()) );
        }
        if (sync_model == ALL) /* all to all */
        {
            for (int r = 0; r < nProc; r++)
                if (r != iProc && net->live(r))
                    send_list.push_back(r);
        }
        else if (sync_model == CHORD) /* chord */
        {
            for (int k = 1; 2 * k <= nProc; k *= 2)
            {
                int to_send = (iProc + nProc / (2 * k)) % nProc;
                if (net->dead(to_send)) ++to_send; // NOTE: this can be out of range
                //                     we should also remove duplicate destinations
                send_list.push_back(to_send);
            }
        }
        else if (sync_model == HALTON) /* halton */
        {
	    int base = 2;
	    if ((nProc & (nProc - 1)) == 0) // power of 2
	    	base = 3;
            if(dbg)cout<<" dbg: HALTON"<<" base="<<base;
            for (int l = 0, k = 1; 2 * k <= nProc; k *= 2, l++)
            {
                int to_send = (iProc + (int)(nProc * halton(base, l))) % nProc;
                if (to_send == iProc)
                    to_send = (to_send + nProc - 1) % nProc;
                if (net->dead(to_send)) ++to_send;
                send_list.push_back(to_send);
            }
        }
        else if (sync_model == RANDOM) /* random */
        {
            if(dbg)cout<<" dbg: RANDOM";
            srand(0);
            for (int r = 0; r < nProc; r++)
                if (r != iProc && net->live(r) && float(rand()) / RAND_MAX < 0.5f)
                    send_list.push_back(r);
            // must check for graph partitions XXX (0.5 is on cusp of highly-likely single huge component)
        }
#if 0
        else if (sync_model == TO_WORKER) /* to workers */
        {
            if (iProc == 0)
                for (int i = 1; i < nProc; i++)
                    if( net->live(i) ) send_list.push_back(i);
        }
        else if (sync_model == TO_SERVER) /* send to parameter server */
        {
            if( net->dead(0) ) cout<<"\nWARNING: TO_SERVER, but server is marked DEAD\n"<<endl;
            if (iProc != 0)
                send_list.push_back(0); // set 0 rank as default parameter server
        }
#else
        else if (sync_model == PARA_SERVER) /* to workers */
        {
            // NEW:  [ejk] don't really see a need for TO_WORKER and TO_SERVER.
            if(dbg)cout<<" dbg: PARA_SERVER";
            if (iProc == 0){
                for (int i = 1; i < nProc; i++){
                    if ( net->live(i) ) send_list.push_back(i);
                }
            }else{
                if ( net->dead(0)) cout << "\nWARNING: PARA_SERVER server is DEAD\n" << endl;
                send_list.push_back(0); // set 0 rank as default parameter server
            }
        }
#endif
        else if (sync_model == BUTTERFLY)  /* send to parameter server */
        {
            cout<<"\nWARNING: BUTTERFLY send_list not implemented"<<endl;
        }
        // Some of the above are obviously wrong, so fix these errors:
        // - possible range overrun after ++to_send
        for( size_t r=0U; r<send_list.size(); ++r ) send_list[r] %= nProc; // easy first correction

        // - mark duplicate destinations as veryBad
        int const veryBad = nProc;          // XXX really should be orm_rank_t
        for( size_t r=0U; r<send_list.size(); ++r ) {
            // set downstream duplicates to nProc, for later removal
            orm_rank_t const sr = send_list[r];
            for(size_t d=0U; d<r; ++d){
                if( send_list[d] == sr ){
                    send_list[r] = veryBad;
                    break;
                }
            }
        }

        // - remove veryBad entries
        auto newEnd = std::remove( send_list.begin(), send_list.end(), veryBad );
        if( newEnd != send_list.end() ){
            orm->printf(orm," get_send_list sync_model %u removed some dups");
            send_list.resize( std::distance( send_list.begin(), newEnd ));
        }

        // - some graphs my not be fully connected (ex. RANDOM) XXX TODO

        return send_list;
    }
#endif

#if 0 // moved to .hh (or .cuh)
    int Dstorm::get_iProc()
    {
        return iProc; // our base has iProc, nProc and Orm* orm
    }


    int Dstorm::get_nProc()
    {
        return nProc;
    }
#endif

    std::string Dstorm::print_sync_list(IoNet_t sync_model)
    {
        std::ostringstream oss;
        oss<<iographs[sync_model].pretty();
        if( sync_model > send_bufnums.size() ){
            oss<<" sync_model not in send_bufnums (ERROR)\n";
        }else{
            oss<<"\tsendbufnums["<<send_bufnums[sync_model].size()<<"]={";
            for(uint32_t i=0U; i<send_bufnums[sync_model].size(); ++i){
                oss<<" "<<unsigned{send_bufnums[sync_model][i]};
            }
            oss<<"}\n";
#ifndef NDEBUG
            { // Invariant Checking for send_bufnums vs. our sendList and send-destination's recvList.
                uint32_t nerr = 0U;
                std::vector<Tnode> const& snd = iographs[sync_model].send();    // sendList of iProc
                // that rcvList for rank snd[i] has rcvList[send_bufnums[i]] == iProc
                for(uint32_t i=0U; i<snd.size(); ++i){
                    // recvList of our sendList[i] destination:
                    //             static call to ScalGraphs::ScalNet::mkRecvList
                    vector<Tnode> const recvOfSnd_i = Graphs::WrapNet::mkRecvList( iographs[sync_model].unet.get(), snd[i] );
                    //             intentionally not very nice because it is SLOW
                    // must have send_bufnum_i'th entry pointing back to our Rank:
                    Tnode const send_bufnum_i = send_bufnums[sync_model][i];
                    if( ! (recvOfSnd_i[ send_bufnum_i ] == iProc) ){
                        oss<<" send_bufnums["<<i<<"] is WRONG: recvOfSnd_i[send_bufnums[i]] = "
                            <<recvOfSnd_i[send_bufnum_i]<<", not "<<iProc<<"\n";
                        ++nerr;
                    }
                }
                if(nerr) { oss<<"** ERROR ** send_bufnums had "<<nerr<<" bad entries"<<"\n"; }
            }
#endif
        }
        return oss.str();
    }

#if 0
    template< typename T >
    orm_segment_id_t Dstorm::add_segment( typename std::vector<T> const& v
                                            , IoNet_t ionet )
    {
        typedef typename std::vector<T> vtype;
        //SegInfo is NON-COPYABLE
        //SegInfo info(*this);
        auto emp = dynamic_cast<DstormSegInfoMap*>(this->segInfoMap)
            -> infos.emplace( seg_num, detail::SegInfo(*this) );
        ORM_NEED( /*bool const emp_new =*/ emp.second == true );
        SegInfo& info = emp.first->second;       // second part of deref of iterator

        //SegInfo& info = (*segInfoMap)[seg_num];
        info.seg_id     = seg_num++;
        info.ionet      = ionet;              // XXX temporarily
        info.bufBytes   = sizeof(vtype) * v.size();
        info.bufBytes   = DSTORM_ALIGN_UP( info.bufBytes );

        info.valid      = true;

        // simplistic for now XXX maybe have exceptions
        //info.nbuf       = 1U/* me */ + info.send().size()/* send */ + info.recv().size()/* receive */;
        info.nbuf       = 1U/* me */ + get_send_list(ionet).size() + get_recv_list(ionet).size();
        info.segBytes   = info.nbuf * info.bufBytes;
        NEED(orm->segment_create( orm, info.seg_id, info.segBytes,
                                   ORM_GROUP_ALL, ORM_BLOCK, ORM_MEM_INITIALIZED));
        // ORM_GROUP_ALL:         The group of ranks with which the segment should be registered
        // ORM_BLOCK:             Timeout in milliseconds (or ORM_BLOCK/ORM_TEST)
        // ORM_MEM_INITIALIZED:   Memory allocation policy
        NEED(orm->segment_ptr (orm, info.seg_id, &(info.mem)));

        orm->printf(orm, "add segment %d: \n", info.seg_id);
        info.valid = true;
#if defined(MILDE_TENSOR)
        memset( info.mem, 0, info.segBytes ); // zero entire segment (maybe paranoid?)
#endif
        return info.seg_id;
    }
#endif

#if 0
    void Dstorm::delete_segment(int seg_id) const
    {
        SegInfo& info = (*segInfoMap)[seg_id];
        if (info.valid)
        {
            orm->segment_delete(orm, seg_id);
            info.valid = false;
        }
    }
#endif

#if 0
    inline IoNet_t Dstorm::ionet( SegIdx seg_idx ) const
    {
        return segs()[seg_idx].ionet;
    }
#endif

    // --- force instantiation so common template functions make it into library ---

    //template orm_segment_id_t Dstorm::add_segment<float>( typename std::vector<float> const& v
    //                                                        , IoNet_t ionet );
    //template orm_segment_id_t Dstorm::add_segment<double>( typename std::vector<double> const& v
    //                                                         , IoNet_t ionet );
    //template orm_segment_id_t Dstorm::add_segment<complex>( typename std::vector<complex> const& v
    //                                                          , IoNet_t ionet );


}//dStorm::
