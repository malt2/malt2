/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef GLOBIONET_HPP
#define GLOBIONET_HPP

#include "userIoNet.hpp"
#include "liveness.hpp"

#include <assert.h>

#include <cstdint>
#include <string>
#include <vector>
#include <memory>       // unique_ptr
#include <algorithm>    // std::max

// debug...
#include <iostream>

/** namespace mm2 denotes code backported from milde_malt2.
 * No point making any huge changes to anything in this namespace. */
namespace mm2 {

    /** graph with precalculated send lists & recv lists */
    //template< typename USERionet > class GlobNet;
    class GlobNet;

    namespace detail {
        /** graph with precalculated send lists */
        class GlobSendNet;
    }

    namespace detail {
        using mm2::Tnode;

        class GlobSendNet
        {
            GlobSendNet(GlobSendNet const&) = delete;
            GlobSendNet& operator=(GlobSendNet const&) = delete;

        public:
            GlobSendNet( std::unique_ptr<mm2::UserIoNet>&& ptr )
                : unet( std::move(ptr) )      // USERionet now fully constructed
                  , sendLists( unet->verts )
            {
                assert( sendLists[0].size() == 0U );
                //std::cout<<" +GlobSendNet: sendLists[0] = "; for(uint32_t i=0U; i<sendLists[0].size(); ++i) std::cout<<" "<<sendLists[0][i]; std::cout<<std::endl;
                this->rebuild(nullptr);
                //std::cout<<" +GlobSendNet: sendLists[0] = "; for(uint32_t i=0U; i<sendLists[0].size(); ++i) std::cout<<" "<<sendLists[0][i]; std::cout<<std::endl;
                assert( sendLists.size() == unet->verts );
            }
            ~GlobSendNet() {}

            std::vector<Tnode> const& send( Tnode const n ) const {
                assert( n < sendLists.size() );
                return sendLists[n];
            }

            /** string for debug msgs, prepends 'G' to USERionet name */
            std::string       name() const;
            /** string good to embed in filenames, prepends 'G' to USERionet shortname. */
            std::string       shortname() const;

            /** This prints the \b full network \e without affecting stored
             * sendLists[]. I.e. this is a \c const function, and
             * the \p lv parameter does not \c update anything. */
            std::string pretty( dStorm::detail::LiveBase const *lv=nullptr ) const;

        protected:
            //using USERionet::mkSend;  // with one arg only

            /** recreate send lists from \c unet and (if \p lv != nullptr)
             * remove dead nodes. May be slow if unet has errors or
             * fixups get complicated!
             *
             * \post sendLists.size() == unet->verts
             * \post sendLists[*][*] in range [0,verts-1U]
             * \post no self-loops
             * \post no multiple arcs (duplicates in sendLists.
             * \note \c GlobNet may also have restrictions on recvLists that
             *       modify our sendLists[]. We don't know about that here.
             *
             * easy impl: just leverage \c niceSendList(...) in \ref UserIoNet.hpp
             */
            void rebuild( dStorm::detail::LiveBase const *lv );

            std::unique_ptr<mm2::UserIoNet> unet;

            /** sendLists[r] is an <B>ordered list</b> of destination nodes.
             * In case of network bottleneck, you can always send to just a few
             * of the first entries in sendLists[r]. */
            std::vector< std::vector<Tnode> > sendLists;

        };


    }//detail::


    /** Encapsulate graph lookup stuff and add liveness concept.
     *
     * - Why?
     *   - GlobNet defaults to providing (\e only) an ordered list
     *     of outgoing nodes.
     *   - Dstorm actually likes to have more info available.
     *
     * - Usage:
     *   - Constructor \e adopts a user-supplied GlobNet,
     *   - \c setLiveBase(lv) builds full send/recv lists,
     *     - remembering the liveness impl \c lv.
     *   - if you suspect liveness change, call \c update
     *     to rebuild send/recv lists
     *
     * - points to a MKIoNet base, actually implememted by
     *   a simple, user-defined graph generator object
     * - \e extend a particular \c GlobIoNetUser class with 
     *   sendLists construction object.
     *   - typical constructor args would be nProc (\# of nodes)
     *   - and (often) degree (\# io in/out)
     * - graph may reflect liveness after suitable \c setLiveBase
     *   installed. Default reflects complete graph.
     *
     * - No thread safety
     */
    class GlobNet : protected detail::GlobSendNet
    {
    public:
        typedef uint_least16_t Tnode;
        typedef detail::GlobSendNet BaseImpl;

        std::string name() const { return this->unet->name(); }
        std::string shortname() const { return this->unet->shortname(); }

        /** accessor for GlobNet::sendLists data  (to whom does node \c n send data to?) */
        std::vector<Tnode> const& send(Tnode const n) const
        { return BaseImpl::send(n); }

        /** accessor for GlobNet::pretty string version of the \e full graph XXX use GlobSendNet version or print both send and recv lists !!! */
        std::string pretty_raw( dStorm::detail::LiveBase const *lv=nullptr ) const
        { return this->unet->pretty( lv ); }
        std::string pretty( dStorm::detail::LiveBase const *lv=nullptr ) const
        { return this->BaseImpl::pretty( lv ); }

        /** accessor for \c this->recvLists (<-- \c build_lists <-- \c update) */
        std::vector<Tnode> const& recv(Tnode const iProc) const
        { return this->recvLists[iProc]; }

        Tnode size() const { return this->unet->verts; }
        void setLiveBase( dStorm::detail::LiveBase * lv ) {
            this->lv = lv;
            this->build_lists();
        }
        bool live( Tnode const r ){ return (lv? lv->live(r): true); }
        void update() {
            if( lv ) lv->update();      // get current list of active nodes
            this->build_lists();        // rebuild send (& recv) lists
        }
    public: // for milde_malt, public.
        /** Construct from unique ptr to a mm2::user::UserIoNet base ptr.
         * - \sa IoNetAll for example implementation of a \c UserIoNet.
         * Only Dstorm (by derivation) gets to make a GlobNet (user can ignore).
         */
        GlobNet( std::unique_ptr<mm2::UserIoNet>&& ptr )
            : GlobSendNet( std::move(ptr) )
              , lv(nullptr)
              , recvLists( sendLists.size() )
        {
            // already done in base class: this->rebuild(nullptr);
            assert( sendLists.size() == unet->verts );
            assert( unet );
            //std::cout<<" +Globnet: sendLists[0] = "; for(uint32_t i=0U; i<sendLists[0].size(); ++i) std::cout<<" "<<sendLists[0][i]; std::cout<<std::endl;
            build_lists();      // build up our recvLists[]
        }
        //virtual ~GlobNet() = 0;
        ~GlobNet() {};
    private:
        /** transform our base class sendLists into this->recvLists[].
         * Full graphs apply any this->lv liveness to all lists. */
        void build_lists();
    private: // data
        dStorm::detail::LiveBase * lv;                  ///< NULL equiv. LiveAlways

        /** recvLists are a trivial transform of ionet->sendLists */
        std::vector< std::vector<Tnode> > recvLists;
    }; // class GlobNet
}//mm2::
#endif // MKIONET_HPP
