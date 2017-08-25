/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef SCALIONET_HPP
#define SCALIONET_HPP

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

//#warning "This file is not yet mainstream"

/** namespace mm2 denotes code backported from milde_malt2.
 * No point making any huge changes to anything in this namespace. */
namespace mm2 {

    /** distributed graph, caching only send/recv list of a particular node.
     * These graphs never reflect liveness within send/recv lists. */
    class ScalNet;

    /// \sa prtIonet.cpp test code can define other ones.
    namespace detail {
        /** graph with precalculated send list particular to a particular node. */
        class ScalSendNet;
    }

    namespace detail {
        using mm2::Tnode;

        class ScalSendNet
        {
            ScalSendNet(ScalSendNet const&) = delete;
            ScalSendNet& operator=(ScalSendNet const&) = delete;

        public: // for testing
            ScalSendNet( Tnode const node, std::unique_ptr<mm2::UserIoNet>&& ptr )
                : unet( std::move(ptr) )      // USERionet now fully constructed
                  , node(node)
                  , sendList( ScalSendNet::check( unet->mkSend(node), unet.get(), node ))
            { assert(unet); }
            ~ScalSendNet() {}

        public:
            /** abstract string for debug msgs */
            std::string       name() const;
            /** abstract string good to embed in filenames */
            std::string       shortname() const;

            /** shorthand for send list of \c this->node */
            std::vector<Tnode> const& send() const { return this->sendList; }

            /** This prints the \b full network \e without affecting the
             * \e const \c sendList[]. */
            std::string pretty( dStorm::detail::LiveBase const *lv=nullptr ) const;

        public: // all data is const, so can be exposed
            std::unique_ptr<mm2::UserIoNet> const unet;

            /** A distributed graph caches only the sendlist of one node */
            Tnode const node;

            /** sendList[] is an <B>ordered list</b> of destination nodes for \c this->node.
             * In case of network bottleneck, you can always send to just a few
             * of the first entries in sendLists[r]. */
            std::vector< Tnode > const sendList;

        protected:
            /** sanity checks on \c sl send list rvalue.
             * \return modified \c sl or throw runtime_error.
             * called only during construction. */
            static std::vector<Tnode>& check( std::vector<Tnode>&& sl
                                               , mm2::UserIoNet const* const unet
                                               , Tnode const n );

        };


    }//detail::

    /** Encapsulate graph ptr with a utility funcs.
     *
     * - \e extend a particular \c ScalIoNetUser class with 
     *   sendLists construction object.
     *   - typical constructor args would be nProc (\# of nodes)
     *   - and (often) degree (\# io in/out)
     * - graph may reflect liveness after suitable \c setLiveBase
     *   installed. Default reflects complete graph.
     *
     * - No thread safety
     */
    class ScalNet: public detail::ScalSendNet
    {
    public:
        typedef uint_least16_t Tnode;
        typedef detail::ScalSendNet Base;

        std::string name() const { return Base::name(); }
        std::string shortname() const { return Base::shortname(); }

        /** fast access to send list of \c this->node */
        std::vector<Tnode> const& send() const { return Base::send(); }

        /** fast access to recv list of \c this->node */
        std::vector<Tnode> const& recv() const { return this->recvList; }

#if 0
        /** generic send list access, slow if \c n is not =c this->node. */
        std::vector<Tnode> const send(Tnode const n) const { return this->ionet->send(n); }

        /** accessor for \c this->recvLists (<-- \c mkRecvList <-- \c update) */
        std::vector<Tnode> const recv(Tnode const iProc) const
        {
            if( iProc == this->ionet->node ) return this->ionet->recvList;
            // create a temporary globIoNet to service this one call ?
            throw std::runtime_error( "TBD" );
        }
#endif

        std::string pretty( dStorm::detail::LiveBase const *lv=nullptr ) const;

        /** return graph size, nodes 0 .. size()-1. */
        Tnode size() const { return unet->verts; }

    public: // for milde_malt, public.
        ScalNet( Tnode const node, std::unique_ptr<mm2::UserIoNet>&& ptr )
            : ScalSendNet( node, std::move(ptr) )
              , recvList( std::move(mkRecvList( unet.get(), this->node )))
        {
            assert( unet );
            //std::cout<<" +Scalnet: sendLists[0] = "; for(uint32_t i=0U; i<sendLists[0].size(); ++i) std::cout<<" "<<sendLists[0][i]; std::cout<<std::endl;
        }
        //virtual ~ScalNet() = 0;
        ~ScalNet() {};

        /** build send lists and auto-generate the recvLists view.
         *
         * - Creating a GlobNet in order to retrieve our recvList would be best,
         *   but GlobNet does no recvList massaging, so there is an easier way:
         * - const sendList of ScalSendNet base class has been checked and is trusted.
         * - Iterate over temporary ScalSendNet of all other nodes
         * - and build up the receive list that we want.
         *
         * Note: our \c recvList is built as a monotonically increasing sequence.
         *       So far there is no motivation to order the recvList according to
         *       the sender's priority (this order is not unique, anyway).
         */
        static std::vector<Tnode> mkRecvList( UserIoNet* unet, Tnode const node );

        /** node-local recvList. While sendList is ordered, recvList nodes are in no
         * particular order. (For read-oriented graphs, consider reversing graph edges;
         * i.e. swapping the role of sender and receiver.) */
        std::vector<Tnode> const recvList;
    };

}//mm2::

#endif // SCALIONET_HPP
