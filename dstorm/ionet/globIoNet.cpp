/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef GLOBIONET_CPP
#define GLOBIONET_CPP

#include "ionet/globIoNet.hpp"
#include <sstream>
#include <iomanip>

#if defined(__CYGWIN__)
#include <cstdio>               // std::printf
#endif

#if defined(DSTORM_H_)
#define GLOB_PRT orm_printf
#define GLOB_NAG orm_printf
#else
#define GLOB_PRT printf
#define GLOB_NAG printf
#endif

using dStorm::detail::LiveBase;

namespace mm2 {
    using namespace std;
    namespace detail {
        using namespace std;

        std::string GlobSendNet::name() const {
            std::string s("g");
            s.append( unet->name() );
            return s;
        }

        std::string GlobSendNet::shortname() const {
            std::string s("g");
            s.append( unet->shortname() );
            return s;
        }

        std::string GlobSendNet::pretty( LiveBase const *lv /*= nullptr*/ ) const {
            std::ostringstream oss( this->name() );
            oss<<"\nGraph "<<name()<<" current GlobSendNet lists, x means following node is dead";
            //oss<<"\n     | Destination x prefix means 'dead' ...\n";
            oss<<"\n Src | Priority  ";
            Tnode const verts = sendLists.size();
            assert( verts == unet->verts );
            Tnode const maxcol = std::min(verts, Tnode(20U));      // limit length of output lines
            for( Tnode d=0U; d<maxcol; ++d){
                if( d < 3U ) continue;
                oss<<setw(4)<<d<<" ";
            }
            oss<<"\n";
            std::vector<bool> dead(verts,false);        // none dead,
            if( lv != nullptr )                         // ...
                for( Tnode r=0U; r<verts; ++r )         // unless lv
                    if( lv->dead(r) )                   // says otherwise
                        dead[r] = true;

            for( Tnode s=0U; s<verts; ++s) {
                oss<<( dead[s]? "x": " ")<<setw(3)<<s<<" | ";
                vector<Tnode> const& svec = this->sendLists[s];
                for( Tnode prio=0U; prio<maxcol && prio<svec.size(); ++prio) {
                    Tnode const d = svec[prio];
                    oss<<( dead[d]? 'x': ' ')<<setw(3)<<(uint32_t)d<<" ";
                }
                oss<<"\n";
            }
            return oss.str();
        }

        void GlobSendNet::rebuild( LiveBase const *lv )
        {
            // just use default sanitizer from userIoNet.hpp
            // each sendlist individually treated only for rather obvious errors.
            for( Tnode n=0U; n<sendLists.size(); ++n ){
                sendLists[n] = unet->mkSend(n);
                try {
                    niceSendList( sendLists[n], n, sendLists.size(), lv );
#ifndef NDEBUG
                    // a simple sanity check
                    assert( sendLists.size() == unet->verts);
                    for(auto d: sendLists[n]) {
                        assert( d != n );
                        assert( d < unet->verts );
                    }
#endif
                }
                catch( std::runtime_error& e ){
                    // rethrow, with better diagnostics
                    {
                        ostringstream oss;
                        oss<<"\nRaw graph:\n"<<unet->pretty()
                            <<"\nsendLists["<<n<<"] = {";
                        for(auto s: sendLists[n]) oss<<" "<<s;
                        oss<<"}\n";
                        GLOB_PRT( "%s", oss.str().c_str() );
                    }
                    {
                        ostringstream oss(e.what());
                        oss<<" in sendLists["<<n<<"] for graph "<<this->name();
                        std::runtime_error e2( oss.str() );
                        throw e2;
                    }
                }
            }
        }

    }//detail::

    // transform our base class sendLists into this->recvLists[].
    // Full graphs apply any this->lv liveness to all lists.
    void GlobNet::build_lists()
    {
        assert( this->unet );
        cout<<" GlobNet::build_lists() for "<<unet->name()<<endl;
        //cout<<" unet->pretty ... "<<unet->pretty()<<endl;
        //cout<<" sendLists[0] = "; for(uint32_t i=0U; i<sendLists[0].size(); ++i) cout<<" "<<sendLists[0][i]; cout<<endl;
        //cout<<" sendLists[1] = "; for(uint32_t i=0U; i<sendLists[1].size(); ++i) cout<<" "<<sendLists[1][i]; cout<<endl;
        // reconstruct base class sendLists[][]
        // This accounts for dead nodes and sanitizes if necessary.
        this->rebuild( this->lv );
        //cout<<" this->pretty ... "<<this->pretty()<<endl;
        //cout<<" sendLists[0] = "; for(uint32_t i=0U; i<sendLists[0].size(); ++i) cout<<" "<<sendLists[0][i]; cout<<endl;

        // xform to our recvLists
        assert( sendLists.size() == unet->verts );
        assert( recvLists.size() == sendLists.size() );
        for( Tnode rcv=0U; rcv < recvLists.size(); ++rcv )
            recvLists[rcv].clear();

        //cout<<" sendLists[0] = "; for(uint32_t i=0U; i<sendLists[0].size(); ++i) cout<<" "<<sendLists[0][i]; cout<<endl;
        for( Tnode src=0U; src < sendLists.size(); ++src ){
            if( lv && lv->dead(src) )
                continue;
            //std::vector<Tnode> sl = this->sendLists[src];
            //vector<Tnode> const& sl = this->sendLists[src];
            //cout<<" src="<<src<<" ... size "<<sl.size()<<" = "<<sendLists[src].size(); for( auto dest: sl ) { cout<<" sl:"<<dest; } cout<<endl;
            for( auto dest: this->sendLists[src] ) {
                assert( dest < recvLists.size() );
                recvLists[dest].push_back(src);
            }
        }

        // at this point, we could see if recvLists look OK,
        // but for now we have no sanity checks or conditions.
    }

}//mm2::
#endif // GLOBIONET_CPP
