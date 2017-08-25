/** \file Deprrecated -- used to test ionet/scalIoNet.hh, but could be reworked like
 * prtIoNet for further tests. */

/** scalIoNet pegs SCALIO at 1 to test the distrib graph impl.  The
 * distr graph constructors are built on top of the full graph impls
 * that are tested by \ref prtIoNet.cpp */
#define SCALIO 1        // MUST be zero, since we work with full graphs here.

#include "dstorm_fwd.hpp"               // OldIoNetEnum tag values
#include "dstorm_net.hpp"               // ionet includes + some helpers for test/debug

// NOTE: we are NOT include dstorm.hpp.  We are doing very basic ionet tests here.

#include <string.h>     // strncmp
#include <sstream>
#include <stdexcept>
#include <iomanip>

using namespace std;
using namespace mm2::user;      // IoNet impls
using namespace mm2;            // IoNet classes
using namespace dStorm;         // for IoNetEnums like CHORD, ALL, ...
using dStorm::detail::mkScalNet;
using dStorm::detail::mkGlobNet;


#if 0   // deprecated code moved here (was in dstorm.cpp)
namespace dStorm
{
    /** deprecated. \b ALL should go via UserIoNet (soon) */
    std::vector<Tnode> get_send_list( Tnode const iProc, Tnode const nProc,
                                      OldIoNetEnum sync_model, dStorm::detail::LiveBase const* lv)
    {
        bool const dbg = false;

        std::vector<Tnode> dead(nProc,false);
        uint32_t nDead = 0U;
        if( lv ) {
            dead = lv->getDead();
            assert( dead.size() == nProc );
            for (auto a : dead) if ( a ) ++nDead;
            if ( nDead ) {
                MMPRINTF(" get_send_list: avoid_list size is %u, nDead=%u\n"
                         , unsigned(dead.size()), unsigned(nDead) );
            }
        }
        if(dbg)cout<<"get_send_list sync_model="<<name(sync_model)<<endl;

        vector<Tnode> send_list;

        if (sync_model == ALL) /* all to all */
        {
            if(dbg)cout<<" dbg: ALL";
            for (int r = 0; r < nProc; r++)
                if (r != iProc && !dead[r] )
                    send_list.push_back(r);
        }
        else if (sync_model == CHORD) /* chord */
        {
            if(dbg)cout<<" dbg: CHORD";
            for (int k = 1; 2 * k <= nProc; k *= 2)
            {
                int to_send = (iProc + nProc / (2 * k)) % nProc;
                if (dead[to_send]) ++to_send; // NOTE: this can be out of range
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
                int to_send = (iProc + (int)(nProc * mm2::user::halton(base, l))) % nProc;
                if (to_send == iProc)
                    to_send = (to_send + nProc - 1) % nProc;
                if (dead[to_send]) ++to_send;
                send_list.push_back(to_send);
            }
        }
        else if (sync_model == RANDOM) /* random */
        {
            if(dbg)cout<<" dbg: RANDOM";
            srand(0);
            for (int r = 0; r < nProc; r++)
                if (r != iProc && !dead[r] && float(rand()) / RAND_MAX < 0.5f)
                    send_list.push_back(r);
            // must check for graph partitions XXX (0.5 is on cusp of highly-likely single huge component)
        }
        else if (sync_model == PARA_SERVER) /* to workers */
        {
            // NEW:  [ejk] don't really see a need for TO_WORKER and TO_SERVER.
            if(dbg)cout<<" dbg: PARA_SERVER";
            if (iProc == 0){
                for (int i = 1; i < nProc; i++){
                    if ( !dead[i] ) send_list.push_back(i);
                }
            }else{
                if ( dead[0] ) cout << "\nWARNING: PARA_SERVER server is DEAD\n" << endl;
                send_list.push_back(0); // set 0 rank as default parameter server
            }
        }
        else if (sync_model == BUTTERFLY)  /* send to parameter server */
        {
            if(dbg)cout<<" dbg: BUTTERFLY";
            MMPRINTF("\nWARNING: BUTTERFLY send_list not implemented");
        }else{
            std::runtime_error e(" mmIoNet llegal / UNHANDLED sync_model!");
            MMWARN("%s\n", e.what());
            throw e;
        }
        // Some of the above are obviously wrong, so fix these errors:
        // - possible range overrun after ++to_send
        for ( size_t r = 0U; r < send_list.size(); ++r ) send_list[r] %= nProc; // easy first correction

        // - mark duplicate destinations as veryBad
        Tnode const veryBad = nProc;
        for ( size_t r = 0U; r < send_list.size(); ++r ) {
            // set downstream duplicates to nProc, for later removal
            Tnode const sr = send_list[r];
            for (size_t d = 0U; d < r; ++d) {
                if ( send_list[d] == sr ) {
                    send_list[r] = veryBad;
                    break;
                }
            }
        }

        // - remove veryBad entries
        auto newEnd = std::remove( send_list.begin(), send_list.end(), veryBad );
        if ( newEnd != send_list.end() )
        {
            MMPRINTF(" get_send_list sync_model %u removed some dups", (unsigned)sync_model);
            send_list.resize( std::distance( send_list.begin(), newEnd ));
        }

        // - some graphs my not be fully connected (ex. RANDOM) XXX TODO

        return send_list;
    }

    std::vector<Tnode> get_rece_list( Tnode const iProc, Tnode const nProc,
                                      OldIoNetEnum const sync_model, dStorm::detail::LiveBase const* lv)
    {
        vector<Tnode> rece_list;
        rece_list.push_back(iProc);
        for (Tnode r = 0U; r < nProc; ++r) {
            if ( r == iProc || (lv && lv->dead(r)) )
                continue;
            vector<Tnode> send_list = get_send_list(r, nProc, sync_model, lv);
            for (size_t s = 0; s < send_list.size(); s++) {
                if (send_list.at(s) == iProc)
                    rece_list.push_back(r);
            }
        }
        return rece_list;
    }
}dStorm::
#endif
void usage(){
    cout<<" Usage:  scalTest {old|new|both} {verts} [name]\n"
        <<"    or   scalTest test\n"
        <<" Example: scalTest old 15\n"
        <<"          - prints a 15x15 grid of send lists with dest\n"
        <<"            priorities marked with hex code, then 'os's\n"
        <<"          - n should be between 1 and 80\n"
        <<" Example: scalTest test\n"
        <<"          - reduce output, test a range of graph sizes for old/new equiv.\n"
        <<" Note: since support for 'old' graphs has been removed,\n"
        <<"       the old 'get_send_lists' function no longer exists!  So 'old'\n"
        <<"       and 'new' just run slightly different tests on the same nets.\n"
        <<endl;
}

int main(int argc,char** argv){
    bool do_old = false;
    bool do_new = false;
    bool do_tst = false;
    uint32_t do_n=0U;
    if( argc == 2U ) {
        if( strncmp(argv[1],"test",4) == 0U){
            do_old=true;
            do_new=true;
            do_tst=true;
            do_n=1U;
        }
    }else if( argc == 3U ) {
        if( strncmp(argv[1],"old",3) == 0U ) do_old = true;
        else if( strncmp(argv[1],"new",3) == 0U ) do_new = true;
        else if( strncmp(argv[1],"both",4) == 0U ) {do_old=true; do_new = true;}

        istringstream is( argv[2] );
        is >> do_n;
    }else{
        usage();
        return 0;
    }

    if( (do_old==false && do_new==false) || (do_n<=0U || do_n>80U) ){
        usage();
        return -1;
    }

    uint32_t nlo = (do_tst?  1U: do_n   );
    uint32_t nhi = (do_tst? 60U: do_n+1U);
    for( uint32_t n=nlo; n<nhi; ++n)
    {
        if(!do_tst) cout<<" constructing equivalent 'new' iographs (for comparison test)... ";
        //
        //  FullGraphs no longer exists.  See prtIoNet.cpp for new way to do this.
        //
        vector< mm2::GlobNet * > iographs; iographs.reserve(IONET_MAX);
        static_assert( ALL         == 0U, "Bad value for ALL" );
        static_assert( SELF        == 1U, "Bad value for SELF" );
        static_assert( CHORD       == 2U, "Bad value for CHORD" );
        static_assert( HALTON      == 3U, "Bad value for HALTON" );
        static_assert( RANDOM      == 4U, "Bad value for RANDOM" );
        static_assert( PARA_SERVER == 5U, "Bad value for PARA_SERVER" );
        static_assert( STREAM      == 6U, "Bad value for STREAM" );
        static_assert( BUTTERFLY   == 7U, "Bad value for BUTTERFLY" );
        static_assert( IONET_MAX   == 8U, "Bad value for IONET_MAX" );
        iographs.push_back( mkGlobNet(ALL,         n) );        assert( iographs.size() == 1U ); //assert( iographs.size() - 1U == ALL );
        iographs.push_back( mkGlobNet(SELF,        n) );
        iographs.push_back( mkGlobNet(CHORD,       n) );
        iographs.push_back( mkGlobNet(HALTON,      n) );
        iographs.push_back( mkGlobNet(RANDOM,      n) );
        iographs.push_back( mkGlobNet(PARA_SERVER, n) );
        iographs.push_back( mkGlobNet(STREAM,      n) );
        iographs.push_back( mkGlobNet(BUTTERFLY,   n) );  // empty for now
        // At this point, iographs have sendLists created but HAVE NOT called build_lists to create recvLists...
        for( uint32_t h=0U; h<iographs.size(); ++h ){
            // default LiveBase is nullptr, a.k.a. "live always"
            iographs[h]->update();
        }
        if(!do_tst) cout<<" 'new' iographs ... DONE"<<endl;

        vector<OldIoNetEnum> graphTypes = {ALL, SELF, CHORD, HALTON
            , RANDOM, PARA_SERVER, STREAM, BUTTERFLY};

        using ::dStorm::name;
        using ::dStorm::OldIoNet;         // disambiguator for name function
        for(auto g: graphTypes){
            bool equiv=true;
            if(do_old){
                if( !do_tst ){
                    //string s = pretty_table( n, g );
                    cout<<"old g = "<<g<<endl;
		    cout<< name(OldIoNet{g}) << endl;
                    cout<<"old "<<name(OldIoNet{g})<<iographs[g]->pretty();
                    if(!do_tst) cout<<" compare 'new' full with scal impls"<<endl;
                    {
                        // simple test: send_list for cached node of scalable impl is same
                        for( uint32_t i=0U; i<n; ++i ){
                            mm2::ScalNet* scal = mkScalNet(ALL,i,n);
                            {
                                cout<<" scal("<<n<<","<<i<<") = {";
                                auto v = scal->send();
                                for(uint32_t j=0U; j<v.size(); ++j) cout<<" "<<v[j];
                                cout<<" }"<<endl;
                            }{
                                cout<<" full("<<n<<","<<i<<") = {";
                                auto v = iographs[ALL]->send(i);
                                for(uint32_t j=0U; j<v.size(); ++j) cout<<" "<<v[j];
                                cout<<" }"<<endl;
                            }{
                                cout<<" scal recv ("<<n<<","<<i<<") = {";
                                auto v = scal->recv();
                                for(uint32_t j=0U; j<v.size(); ++j) cout<<" "<<v[j];
                                cout<<" }"<<endl;
                            }{
                                cout<<" full recv ("<<n<<","<<i<<") = {";
                                auto v = iographs[ALL]->recv(i);
                                for(uint32_t j=0U; j<v.size(); ++j) cout<<" "<<v[j];
                                cout<<" }"<<endl;
                            }
                            assert( scal->send().size() == iographs[ALL]->send(i).size() );
                            for( uint32_t j=0U; j<scal->send().size(); ++j ){
                                cout<<" j"<<j; cout.flush();
                                assert( scal->send()[j] == iographs[ALL]->send(i)[j] );
                            }
                            assert( scal->recv().size() == iographs[ALL]->recv(i).size() );
                            delete scal;
                        }
                        if(!do_tst) cout<<" scal impl equiv send lists (YAY)"<<endl;
                    }
                    cout<<"old "<<dStorm::name(dStorm::OldIoNet{g})<<iographs[g]->pretty()<<" DONE!"<<endl;
                }
                // check for equivalence: if so, elide duplicate 'new' output
                if( do_new ) {
#if 0
                    for(uint32_t src=0U; src<n; ++src){
                        //    NEXT LINE NO LONGER COMPILES
                        vector<Tnode> a( get_send_list(src,n,g,nullptr ));
                        vector<Tnode> b( iographs[g]->send(src) );   // NB: does NOT understand liveness
                        if( a != b ){
                            cout<<" Problem with "<<iographs[g]->name()<<"n verts = "<<n<<endl;
                            cout<<"old: "    <<setw(3)<<src<<" |"; for(auto x:a) cout<<" "<<setw(3)<<x;
                            cout<<"    new: "<<setw(3)<<src<<" |"; for(auto x:b) cout<<" "<<setw(3)<<x;
                            cout<<" DIFFER"<<endl;
                            equiv=false;
                            throw std::runtime_error(" New and Old not equivalent\n");
                        }
                    }
#endif
                }
            }
            if(!do_tst && do_new){
                cout<<"new "<<iographs[g]->name();
                if( do_old && equiv ){
                    cout<<" new is equivalent to above 'old' graph"<<endl;
                }else{
                    try {
                        cout<<iographs[g]->pretty()<<endl;
                    }catch( std::exception& e ){
                        cout<<" ERROR: "<<e.what();
                    }catch(...){
                        throw std::runtime_error(" ERROR: Huh");
                    }
                }
            }
            if(!do_tst) cout<<endl;
        }
        if(do_tst){
            cout<<" graphs of degree "<<setw(2)<<n<<" : old and new equivalent!"<<endl;
        }else{
            cout<<" n="<<n<<" DONE:  calling some destructors ..."<<endl;
        }
        for( uint32_t h=0U; h<iographs.size(); ++h ){
            // default LiveBase is nullptr, a.k.a. "live always"
            delete iographs[h];
            iographs[h] = nullptr;
        }
    }

    cout<<"\nGoodbye"<<endl;
    return 0U;
}
#if 0 // MkIoNetToServer and MkIoNetToWorker were removed (ParaServer does all)
namespace mm2 {
    namespace user {
        /// @name extensions that milde_malt2 shoud NOT support by default
        //@{
        /** a trivial partial network (a subset of PARA_SERVER) */
        struct MkIoNetToServer : public mm2::detail::MkIoNet ///< abstract base
        {
            virtual std::string name() const { return std::string("MkIoNetToServer");}
            MkIoNetToServer( uint32_t const verts )
                : mm2::detail::MkIoNet(verts)
            {
                MkIoNet::rebuild(nullptr); // will call our mkSend and fill up sendLists[]
            }
            virtual std::vector<Tnode> mkSend( Tnode const n, mm2::detail::LiveBase const *lv = nullptr) const
            {
                std::vector<Tnode> ret;
                if( n != 0U )
                    ret.push_back(0U);
                return ret;
            }
        };
        /** a trivial partial network (a subset of PARA_SERVER) */
        struct MkIoNetToWorker : public mm2::detail::MkIoNet ///< abstract base
        {
            virtual std::string name() const { return std::string("MkIoNetToServer");}
            MkIoNetToWorker( uint32_t const verts )
                : mm2::detail::MkIoNet(verts)
            {
                MkIoNet::rebuild(nullptr); // will call our mkSend and fill up sendLists[]
            }
            virtual std::vector<Tnode> mkSend( Tnode const n, mm2::detail::LiveBase const *lv = nullptr) const
            {
                std::vector<Tnode> ret;
                if( n == 0U )
                    for( Tnode i=1U; i<verts; ++i ){
                        if( lv && lv->dead(i) )
                            continue;
                        ret.push_back(i);
                    }
                return ret;
            }
        };
        //@}
    }//user::
}//mm2::
#endif


