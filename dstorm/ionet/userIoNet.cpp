/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef USERIONET_CPP
/** kludge for milde_malt which "links" to dstorm by compiling a monolithic code agglomeration, dStorm.c */
#define USERIONET_CPP

#include "ionet/userIoNet.hpp"
#include "liveness.hpp"

#include <sstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <algorithm>    // std::remove std::min

#if defined(__CYGWIN__)
#include <cstdio>               // std::printf
#endif

#if defined(DSTORM_H_)
#define USR_PRT orm_printf
#define USR_NAG orm_printf
#else
#define USR_PRT printf
#define USR_NAG printf
#endif

using namespace std;

namespace mm2 {

    std::string UserIoNet::pretty_table( dStorm::detail::LiveBase const*lv /*=nullptr*/ ) const
    {
        Tnode const n = this->verts;
        std::ostringstream oss;
        // 1st pass: count unconnected sources, if empty skip print
        oss<<"...   "<<this->name()<<" ...";
        char const symbol_fill    = ' ';
        Tnode const grid = 5U;
        char const symbol_grid    = '.';
        char const symbol_lowprio = 'o';
        char const symbol_dead    = 'x';
        oss<<"\n";
        if(0){ //debug
            for( uint32_t i=0U; i<n; ++i ){
                oss<<" "<<i;
                char c = '{';
                for(auto x: this->mkSend(i)){ oss<<c<<x; c=','; }
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
        uint32_t unconnected_src = 0U;
        for(uint32_t src=0U; src<n; ++src) {    // uint32_t prints as a number
            std::vector<Tnode> v = this->mkSend(src); //( get_send_list( src,n,sync_model,nullptr ));
            oss<<setw(3)<<src<<" "
                <<((lv && lv->dead(src) )? "X": (v.size()? "|" : "U"))
                <<" ";
            if( v.size() == 0U ){ ++unconnected_src; continue; }
            // default is background grid with some light guidelines
            vector<char> vc( n, (src%grid==grid-1U? symbol_grid: symbol_fill));
            if( src%grid!=grid-1U ) for(uint32_t i=grid-1U; i<n; i+=grid) vc[i] = symbol_grid;
            char symbol = '0'-1;
            for(Tnode prio=0U; prio<v.size(); ++prio){
                if( prio<16U ){
                    ++symbol;
                    if(prio==10U) symbol='a';
                }
                else if(prio>=16U) symbol=symbol_lowprio;
                if( lv && lv->dead(v[prio]) ) symbol = symbol_dead;
                    vc[ v[prio] ] = symbol;
            }
            for(auto c: vc) oss<<c;
            oss<<"\n";
        }
        if( unconnected_src == n ){             // empty graph
            oss.clear();
            oss<<left<<"...   "<<this->name()<<" ...";
            oss<<" EMPTY GRAPH\n";
        }
        return oss.str();
    }

    namespace user
    {
        // Note: these would not normally be in a header, except for wanting
        //       to make sure they are always available as example code.

        static inline std::string classVerts( char const* classname, uint32_t verts )
        {
            std::ostringstream oss;
            oss<<classname<<"("<<verts<<")";
            return oss.str();
        }

        template< typename T >
        static inline std::string classVertsParm( char const* classname, uint32_t verts,
                                                    T const parm, T const default_parm  )
        {
            if( parm == default_parm ) {
                return classVerts( classname, verts );
            }else{
                std::ostringstream oss;
                oss<<classname<<"("<<verts<<","<<parm<<")";
                return oss.str();
            }
        }

        inline std::string IoNetAll::name() const
        {
            IoNetAll const tmp(this->verts);
            uint32_t const default_degree = tmp.degree;
            return classVertsParm( "IoNetAll", verts, degree, default_degree );
        }

        std::string IoNetHalton::name() const
        {
            IoNetHalton const tmp(this->verts);
            uint32_t const default_degree = tmp.degree;
            return classVertsParm( "IoNetHalton", verts, degree, default_degree );
        }

        std::string IoNetChord::name() const
        {
            IoNetChord const tmp(this->verts);
            uint32_t const default_degree = tmp.degree;
            return classVertsParm( "IoNetChord", verts, degree, default_degree );
        }

        std::string IoNetRandom::name() const {
            IoNetRandom const tmp(this->verts);
            float const default_accept = tmp.accept;
            return classVertsParm( "IoNetRandom", verts, accept, default_accept );
        }

        std::string IoNetStream::name() const {
            IoNetStream const tmp(this->verts);
            uint32_t const default_node = tmp.from_node;
            return classVertsParm( "IoNetStream", verts, from_node, default_node );
	}

        std::string IoNetParaServer::name() const {
            return classVerts( "IoNetParaServer", verts );
        }

    }//user::

    // print raw mkSend vectors, with optional 'x' prefix for dead nodes
    // Note: I like the pretty_table format more [ejk]
    std::string UserIoNet::pretty( dStorm::detail::LiveBase const *lv /*=nullptr*/ ) const
    {
        std::ostringstream oss( this->name() );
        oss<<"\nGraph "<<name()<<" raw mkSend send lists, x means following node is dead";
        //oss<<"\n     | Destination x prefix means 'dead' ...\n";
        oss<<"\n Src | Priority  ";
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
            vector<Tnode> raw( mkSend(s) ); // full network destinations for s
            //oss<<" raw["<<raw.size()<<"] ";
            for( Tnode prio=0U; prio<maxcol && prio<raw.size(); ++prio) {
                Tnode const d = raw[prio];
                oss<<( dead[d]? 'x': ' ')<<setw(3)<<d<<" ";
            }
            oss<<"\n";
        }
        return oss.str();
    }

    void niceSendList( std::vector<Tnode>& sl, Tnode const n, Tnode const verts
                       , dStorm::detail::LiveBase const* lv /*= nullptr*/ )
    {
        bool const autofix = true;  // false --> print and throw; true --> print and maybe try to patch

        // \post sendLists[*][*] in range [0,verts-1U]
        uint32_t nErr1 = 0U;
        for(auto & s: sl ){
            if( s >= verts ){
                ++nErr1;
                if( autofix ){ // remap to "somewhere", if bad, we'll remove it later.
                    s = s % verts;
                }
            }
        }
        if( nErr1 ){
            if(!autofix ){      // throw, before any mods to sl
                ostringstream oss;
                oss<<nErr1<<" vertex out-of-range errors"; // in "<<this->name()<<"\n";
                std::runtime_error e(oss.str());
                USR_NAG("%s",e.what());
                throw e;
            }
            for(auto & s: sl ){         // modify sl (these guesses might be bad too)
                if( s >= verts ){
                    s = s % verts;
                }
            }
        }

        // remove self-loops and dead nodes [opt.]  (no error)
        for( auto s: sl )
            if( s==n || (lv && lv->dead(s)) )
                sl.erase( std::remove( sl.begin(), sl.end(), s )
                            , sl.end() );

        // remove multiple sends to same destination (duplicates)
        // This could easily happen for implementations with some random edges.
        uint32_t nErr2 = 0U;
        for ( size_t j = 0U; j < sl.size(); ++j ) {
            Tnode const sj = sl[j];
            for (size_t i = 0U; i < j; ++i) {
                if ( sl[i] == sj ) {     // bad j:  i<j has same value
                    ++nErr2;
                    break;
                }
            }
        }
        Tnode const veryBad = verts;  // a marker value for bad nodes.
        if( nErr2 ){
            if( !autofix ){
                ostringstream oss;
                oss<<nErr2<<" duplicate destination vertices"; // in "<<this->pretty()<<"\n";
                std::runtime_error e(oss.str());
                USR_NAG("%s",e.what());
                throw e;
            }
            for ( size_t j = 0U; j < sl.size(); ++j ) {
                Tnode const sj = sl[j];
                for (size_t i = 0U; i < j; ++i) {
                    if ( sl[i] == sj ) {     // remove j because i<j is a duplicate
                        sl[j] = veryBad;
                    }
                }
            }
            sl.erase( std::remove( sl.begin(), sl.end(), veryBad )
                        , sl.end() ); // remove all veryBad in one swoop.
        }

    }

}//mm2::
#undef USR_PRT
#undef USR_NAG
#endif // USERIONET_CPP
