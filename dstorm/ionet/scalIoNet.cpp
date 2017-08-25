/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef SCALIONET_CPP
/** kludge for milde_malt which "links" to dstorm by compiling a monolithic code agglomeration, dStorm.c */
#define SCALIONET_CPP

#include "ionet/scalIoNet.hpp"
#include "liveness.hpp"
#include <sstream>
#include <iomanip>
//#include <stdexcept>
//#include <algorithm>    // std::remove

#if defined(__CYGWIN__)
#include <cstdio>               // std::printf
#endif

#if defined(DSTORM_H_)
#define MKPRINTF orm_printf
#define MKWARN orm_printf
#else
#define MKPRINTF printf
#define MKWARN printf
#endif

using namespace std;
using dStorm::detail::LiveBase;

namespace mm2 {
        using namespace std;

    namespace detail {
        using namespace std;

        std::string ScalSendNet::name() const {
            std::ostringstream oss;
            oss<<"s"<<this->node<<"-"<<unet->name();
            return oss.str();
        }

        std::string ScalSendNet::shortname() const {
            std::ostringstream oss;
            oss<<"s"<<this->node<<unet->name();
            return oss.str();
        }


        std::vector<Tnode>& ScalSendNet::check( std::vector<Tnode>&& sl
                                                 , mm2::UserIoNet const* const unet
                                                 , Tnode const n )
        {
            assert( unet );
            try {
                niceSendList( sl, n, unet->verts, nullptr );
#ifndef NDEBUG
                // a simple sanity check
                for(auto d: sl ){
                    assert( d < unet->verts );
                }
#endif
            }
            catch( std::runtime_error& e ){
                // rethrow, with better diagnostics
                {
                    ostringstream oss;
                    oss<<"\nScalSendNet raw graph:\n"<<unet->pretty()
                        <<"\nsendList for node "<<n<<" = {";
                    for(auto s: sl ) oss<<" "<<s;
                    oss<<"}\n";
                    MKPRINTF( "%s", oss.str().c_str() );
                }
                {
                    ostringstream oss(e.what());
                    oss<<" in sendList for node "<<n<<" for graph "<<unet->name();
                    std::runtime_error e2( oss.str() );
                    throw e2;
                }
            }
            return sl;
        }

        std::string ScalSendNet::pretty( LiveBase const *lv /*=nullptr*/ ) const
        {

            std::vector<bool> dead(unet->verts,false);        // none dead,
            uint32_t nDead = 0U;
            if( lv != nullptr )                         // ...
                for( Tnode r=0U; unet->verts; ++r )         // unless lv
                    if( lv->dead(r) )                   // says otherwise
                        dead[r] = true;

            std::ostringstream oss( this->name() );
            oss<<"\n Graph "<<name();
            if( lv && nDead > 0U ) oss<<" 'x' ~ dead";

            oss<<"\n Node "<<(dead[node]? "x ":"  ")<<left<<setw(3)<<(unsigned)node;
            oss<<"\n        sendList["<<sendList.size()<<"]={ ";
            for( Tnode s=0U; s<sendList.size(); ++s)
                oss<<(dead[s]? "x ": "  ")<<setw(3)<<sendList[s];
            oss<<" }";
            // no recv list to print
            oss<<"\n";
            return oss.str();
        }
    }//detail::

    std::string ScalNet::pretty( dStorm::detail::LiveBase const *lv /*=nullptr*/ ) const
    {

        std::vector<bool> dead(unet->verts,false);        // none dead,
        uint32_t nDead = 0U;
        if( lv != nullptr )                         // ...
            for( Tnode r=0U; unet->verts; ++r )         // unless lv
                if( lv->dead(r) )                   // says otherwise
                    dead[r] = true;

        std::ostringstream oss( this->name() );
        oss<<"\n Graph "<<name();
        if( lv && nDead > 0U ) oss<<" 'x' ~ dead";

        oss<<"\n Node "<<(dead[node]? "x ":"  ")<<left<<setw(3)<<(unsigned)node;
        oss<<"\n        sendList["<<sendList.size()<<"]={ ";
        for( Tnode s=0U; s<sendList.size(); ++s)
            oss<<(dead[s]? "x ": "  ")<<setw(3)<<sendList[s];
        oss<<" }";
        oss<<"\n        recvList["<<recvList.size()<<"]={ ";
        for( Tnode r=0U; r<recvList.size(); ++r)
            oss<<(dead[r]? "x ": "  ")<<setw(3)<<recvList[r];
        oss<<" }";
        oss<<"\n";
        return oss.str();
    }

    /** Note: recvLists have NO ORDERING in this implementation */
    std::vector<Tnode> ScalNet::mkRecvList( UserIoNet* unet, Tnode const node )
    {
        std::vector<Tnode> ret;

        std::vector<Tnode> osend;
        for( Tnode other=0U; other < unet->verts; ++other ){
            if( other == node )
                continue;
            osend = detail::ScalSendNet::check( unet->mkSend(other)
                                                , unet
                                                , other );
            //cout<<" other="<<other<<" osend"; for(auto o:other)cout<<" "<<o;
            for( auto dest: osend ){            // IF element in send list of
                assert( dest < unet->verts );
                if( dest == node ){             // other points to node,
                    //cout<<" x";
                    ret.push_back(other);       // THEN other is node's recvList
                    break;
                }
            }
            //cout<<endl;
        }
        return ret;
    }

#if 0
        void ScalIoNet::build()
        {
            cout<<" ScalIoNet::build ..."<<endl;
            // First build TEMPORARY full network
            // full network constructor "sanitizes" the user-defined 'mkSend' network by
            // i) removing self-loops or out-of-range sendList entries
            // ii) snipping out dead nodes (no effect, since we have "all alive")
            mm2::user::GlobAdapt adapted( *this );  // a globIoNet Impl that re-uses our mkSend function
            GlobIoNet* glob = dynamic_cast< GlobIoNet* >( &adapted );
            {
                std::ostringstream oss;
                oss<<" ScalIoNet::build based on full network: "<<glob->pretty(nullptr);
                MKWARN("%s\n",oss.str().c_str());
            }

            // Next, create the global graph that includes sanitizing and receive lists
            mm2::GlobNet sane( glob );
            // sane.setLiveBase( nullptr );     // no dead nodes, default setting
            sane.update();                      // sanitize send lists and create recv lists
            {
                std::ostringstream oss;
                oss<<" GlobNet version (has recvLists too : "<<sane.pretty(nullptr);
                MKWARN("%s\n",oss.str().c_str());
            }

            // Finally, finish our constructor by caching the sanitized send & recv
            // list particular to this->node.
            {
                cout<<"this->node = "<<this->node<<endl;
                std::vector<Tnode> sanitized = sane.send( this->node );
                cout<<" sane sl.size "<<sanitized.size()<<" : "; for(uint32_t x=0U; x<sane.size(); ++x) cout<<" "<<x; cout<<endl;
                this->sendList = sanitized;
                cout<<" sl.size "<<sendList.size()<<endl;
            }
            {
                auto rl = const_cast< std::vector<Tnode>& >(this->recvList);
                std::vector<Tnode> sanitized = sane.recv( this->node );
                cout<<" sane rl.size "<<sanitized.size()<<endl;
                this->recvList = sanitized;
                cout<<" rl.size "<<this->recvList.size()<<endl;
            }

            // full graph temporaries disappear here
        }

        std::string ScalIoNet::pretty( LiveBase const *lv /*=nullptr*/ ) const
        {

            std::vector<bool> dead(verts,false);        // none dead,
            uint32_t nDead = 0U;
            if( lv != nullptr )                         // ...
                for( Tnode r=0U; r<verts; ++r )         // unless lv
                    if( lv->dead(r) )                   // says otherwise
                        dead[r] = true;

            std::ostringstream oss( this->name() );
            oss<<"\n Node "<<(unsigned)node<<" Graph "<<name();
            if( lv && nDead > 0U ) oss<<" 'x' ~ dead";

            oss<<"\n      sendList["<<sendList.size()<<"]={ "<<left;
            for( Tnode s=0U; s<sendList.size(); ++s)
                oss<<(dead[s]? " x": " ")<<setw(4)<<sendList[s];
            oss<<" }\n      recvList["<<recvList.size()<<"]={ ";
            for( Tnode r=0U; r<recvList.size(); ++r)
                oss<<(dead[r]? " x": " ")<<setw(4)<<recvList[r];
            oss<<" }\n";
            return oss.str();
        }
#endif


}//mm2::
#undef MKPRINTF
#undef MKWARN
#endif // SCALIONET_CPP
