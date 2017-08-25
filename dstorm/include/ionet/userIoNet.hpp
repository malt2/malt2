/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef USERIONET_HPP
#define USERIONET_HPP

#include "ionet_fwd.hpp"
#include "liveness.hpp"         // it's even shorter than dstorm_fwd.hpp
#include <assert.h>

#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>    // std::max
#include <random>

// debug...
#include <iostream>
#include <type_traits>

#ifndef OLDIONET
/** OLDIONET 1 uses historical net settings, good for compataibilty test routines */
#define OLDIONET 0
#endif

//#warning "This file is not yet mainstream"

/** namespace mm2 denotes code backported from milde_malt2.
 * No point making any huge changes to anything in this namespace. */
namespace mm2 {

    /** Modify a send list by removing graph errors and [opt] dead nodes.
     * User graph implementations can easily have bad characteristics
     * or even errors.  Check out the send lists to ensure nice graph
     * properties. This function may be slow.
     *
     * \p sl    sendList to sanitize
     * \p n     sending node
     * \p verts number of nodes in graph
     * \p lv [opt.] nullptr assumes all destinations [0..verts-1U] available
     *
     * Most things are handled just by deleting entries, but \c niceSendList is
     * permitted to adds new destinations to \c sl too.
     *
     * \post sl.size() < verts
     * \post sl[*] in range [0,verts-1U]
     * \post no self-loops
     * \post no multiple arcs (duplicates in sendLists.
     *
     * \throw runtime_error for bad errors, guaranteeing \e no modifications
     *        have been done to \c sl, if you wish to handle the exception.
     */
    void niceSendList( std::vector<Tnode>& sl, Tnode const n, Tnode const verts
                       , dStorm::detail::LiveBase const* lv = nullptr );

    /** provides implementations for name, shortname and mkSend. */
    class UserIoNet
    {
    public:
        /** abstract string for debug msgs */
        virtual std::string       name() const = 0;

        /** abstract string good to embed in filenames */
        virtual std::string       shortname() const = 0;

        /** print raw \c mkSend vectors, \e before any sanity checks.
         * \p lv if non-NULL dead entries are marked with an 'x'.  */
        std::string pretty( dStorm::detail::LiveBase const *lv=nullptr ) const;

        /** A utility function (possibly slow) that prints a tabular form of a
         * \c usrIoNet. This calls \c mkSend, which is allowed to be slow (and return a temporary vector). */
        std::string pretty_table( dStorm::detail::LiveBase const*lv=nullptr ) const;
        /** <EM>Virtually always</em> we expect send to be called with exactly
         * the same argument.  Rarely, like generating recv lists,  we get
         * called for send lists of other nodes. Since user-defined code
         * might be untrustworty, return a non-const temporary. */
        virtual std::vector<Tnode> mkSend( Tnode const n ) const = 0;

        virtual ~UserIoNet() {}

        /** A distributed graph caches only the sendlist of one node */
        Tnode const verts;

    protected:
        /** Derived class \b must set verts to number of nodes in full graph. */
        UserIoNet( uint32_t const verts ) : verts( verts )
        {}
    };


    namespace user
    {
        /// @name helpers
        //@{
        /** integer log base 2, rounding down, so \f$2^{lg2(n)} <= n\f$, except \c lg2(0)==0. ? optimize ? */
        inline uint32_t lg2( uint32_t n ){
            uint32_t ret = 0U;
            //std::cout<<"lg2("<<n<<")=..."; std::cout.flush();
            while (n >>= 1) ++ret;
            //std::cout<<ret<<std::endl;
            return ret;
        }

        /** XXX improve this */
        static inline unsigned pow2_le( unsigned const n ){
            return (1U<<mm2::user::lg2(n));        // there are better ways
        }


        /** if \c deg==-1U return lg2(verts), else return \c min(deg,verts-1U) */
        inline uint32_t default_lg2( uint32_t const verts, uint32_t const deg){
            uint32_t ret = (deg==-1U)
                ? lg2(verts)
                : std::min( deg, verts-1U );
            //std::cout<<" default_lg2(verts="<<verts<<",deg="<<deg<<")="<<ret;
            return ret;
        }

        /** base is a prime number */
        inline float halton(unsigned base, unsigned index)
        {
            float result = 0;
            float f = 1.0 / base;
            unsigned i = index + 1;
            while (i > 0)
            {
                //unsigned digit = i % base;
                // digit = sigma[digit];
                result += f * (i % base);
                i = i / base;
                f = f / base;
            }
            return result;
        }

        //@}

        /** empty graph is easy :) */
        class IoNetEmpty : public mm2::UserIoNet
        {
        public:
            virtual std::string name() const { return std::string("IoNetEmpty"); }
            virtual std::string shortname() const { return std::string("Empty"); }
            IoNetEmpty( uint32_t const verts ) : mm2::UserIoNet(verts) {}
            virtual std::vector<Tnode> mkSend( Tnode const /*n*/ ) const
            {
                return std::vector<Tnode>();
            }
        };

        /** send-to-self only (graph matrix nonzero along diagonal) */
        class IoNetSelf : public mm2::UserIoNet ///< abstract base
        {
        public:
            virtual std::string name() const { return std::string("IoNetSelf"); }
            virtual std::string shortname() const { return std::string("Self"); }
            IoNetSelf( uint32_t const verts ) : mm2::UserIoNet(verts) {}
            /** node n-> n */
            virtual std::vector<Tnode> mkSend( Tnode const n ) const
            {
                return std::vector<Tnode>( 1U, n );
            }
        };

        /** simple all-to-all graph. We help later graph sanity checks by
         * avoiding self-loops (no src-->src entry).  Special case:
         * an ALL graph of degree 1 is the cyclic graph 0-->1-->2-->...-->0
         */
        class IoNetAll : public mm2::UserIoNet ///< abstract base
        {
        public:
            virtual std::string name() const;
            virtual std::string shortname() const { return std::string("All"); }
            /** \c max_degree default will use graph of degree \c verts-1U */
            IoNetAll( uint32_t const verts, uint32_t max_degree = -1U )
                : mm2::UserIoNet(verts)
                , degree( std::max(1U, std::min(max_degree,verts-1U)))
            {}
            /** node n-> n+1, n+2, n+verts-1, modulo verts */ 
            virtual std::vector<Tnode> mkSend( Tnode const n ) const
            {
                std::vector<Tnode> ret;
                ret.reserve( this->degree );
#if OLDIONET==1
                // default, IGNORES degree setting, and "priority" order is bad
                for(Tnode r=0U; r<verts; ++r){
                    if( r == n )                        // self-loop
                        continue;                       // gets ignored
                    ret.push_back(r);
                }
#else // better, and respects degree
                for(Tnode r=0U, rmod=n+1U; r<degree; ++rmod, ++r){
                    if( rmod >= verts) rmod = 0U;
                    ret.push_back(rmod);
                }
#endif
                //std::cout<<" node "<<n<<" mkSend:size:"<<ret.size();
                return ret;
            }
            // extra functions:
            //uint32_t degree() const {return this->degree;}
            uint32_t const degree;
        };
        /** DATAXFR */
        struct IoNetStream : public mm2::UserIoNet ///<abstract base
        {
            virtual std:: string name() const;
            virtual std:: string shortname() const {return std::string("Stream");}
            /** max_degree defaults to all */
            IoNetStream( uint32_t const verts, uint32_t from_node = 0U )
                : mm2::UserIoNet(verts)
                , from_node(from_node)
            {}
            virtual std::vector<Tnode> mkSend( Tnode const n ) const
            {
                std::vector<Tnode> ret;

		if( n == from_node ){
                    ret.reserve( verts - 1U );
                    for( Tnode i=0U; i<verts; ++i ){
                        if( i != n )
                            ret.push_back(i);
                    } 
		} 
                //std::cout<<" node "<<n<<" mkSend:size:"<<ret.size();
                return ret;
            }
            uint32_t const from_node;
        };
        /** CHORD */
        struct IoNetChord : public mm2::UserIoNet ///< abstract base
        {
            virtual std::string name() const;
            virtual std::string shortname() const { return std::string("Chord"); }
            /** max_degree defaults to maximum of lg2(verts) */
            IoNetChord( uint32_t const verts, uint32_t max_degree = -1U )
                : mm2::UserIoNet(verts)
                  , degree( std::max(1U, default_lg2(verts,max_degree)))
            {}
            virtual std::vector<Tnode> mkSend( Tnode const n ) const {
                std::vector<Tnode> ret;
                ret.reserve( this->degree );
                for (int k = 1; 2 * k <= verts; k *= 2) {
                    Tnode dst = (n + verts/(2*k)) % verts;;
                    ret.push_back(dst);
                    if( ret.size() >= degree ) break;
                }
                return ret;
            }
            uint32_t const degree;
        };

        /** HALTON */
        class IoNetHalton : public mm2::UserIoNet ///< abstract base
        {
        public:
            virtual std::string name() const;
            virtual std::string shortname() const { return std::string("Halton"); }
            /** graph constructor only requires number of vertices
             * (i.e. \# of transport \em ranks, or \c nproc) */
            IoNetHalton( uint32_t const verts, uint32_t max_degree = -1U )
                : mm2::UserIoNet(verts)
                  , degree( std::min(verts-1U,default_lg2(verts,max_degree)))
            {}
            std::vector<Tnode> mkSend( Tnode const n ) const {
                assert( n < verts );
                std::vector<Tnode> ret;
                ret.reserve( this->degree );
#if OLDIONET==1
                // fixes verts==8 and verts==16, but graphs still not fully connected for verts==24,...
                unsigned base = 2U;
                if ((verts & (verts - 1U)) == 0) base = 3U;
                std::cout<<" new:base="<<base;
                for(Tnode r=0u; r<this->degree; ++r){
                    uint32_t dst = (n + (uint32_t)(verts * halton(base,r))) % verts;
                    if( dst==n )                              // no self-loops
                        dst = (dst+verts-1U) % verts;
                    ret.push_back(dst);
                }
                std::cout<<" halton n = "<<n<<"    degree = "<<degree<<"    ret.size() = "<<ret.size()<<std::endl;
                //if( degree == 0U ) assert( ret.size() == 0U );
                //else if( verts==2 && degree == 1U ) assert( ret.size() == 1U );     // BAD, but equivalent to old
                //else               assert( ret.size() > 0U && ret.size() < verts );

                if( degree < lg2(verts) ) assert( ret.size() == degree ); // may be ok not to hold.
#elif 0 // Asim's suggestion, using a relatively prime base fixes connectivity issues
                // but head-to-head contest had it getting slightly lower 2nd singular values
                Tnode relativePrime(Tnode const n) { 
                    Tnode const primes[] = [2,3,5,7,11,13,17,19,23,29,31,37];   // good for n < (2^42 + loose change)
                    uint8_t iprime;
                    for( iprime = 0U; verts % primes[iprime] == 0U; ) {
                        ++iprime;
                        assert( iprime < sizeof(primes) / sizeof(Tnode) );
                    }
                }
                unsigned const base = relativePrime(verts);
                for( unsigned i=0U; ret.size() < degree; ++i ){
                    uint32_t dst = (n + (uint32_t)(verts * halton(base,i))) % verts;
                    if (dst == n ) continue;    // avoid self-loops
                    assert( dst < verts );
                    ret.push_back(dst);
                }
                assert( ret.size() > 0U && ret.size() < verts );
#elif 1 // better, had more high 2nd sing value over range 4-70-ish for degree = log2(verts)
                // cyclically shift above series so n-->n+1 is always the first item.
                // This guarantees full connectivity all the way down to graph out_degree 1
                // Halton sequence output value 1 occurs at position
                //      r = pow2_le(verts) / 2U - 1U
                Tnode const rshift = pow2_le(this->verts) / 2U - 1U;
                // Halton seq cyclic only for verts = power of two, I think, so need to downshift
                for(Tnode rr=rshift, r=0U; r<this->degree; ++rr, ++r){
                    if( rr == verts )
                        rr -= verts;
                    unsigned uHalton = verts * halton(2,rr);
                    assert( uHalton < verts );
                    if( r == 0U ) assert( uHalton == 1U ); // assert that we got rshift correct.
                    unsigned dst = (n + uHalton) % verts;
                    if( dst==n )                              // no self-loops
                        continue;
                    ret.push_back(dst);
                }
                assert( ret.size() > 0U && ret.size() < verts );
#endif
                return ret;
            }
            uint32_t const degree;      // degree == max_degree
        };

        /** RANDOM \todo need a multithreaded rand here (or a trivial local one) */
        struct IoNetRandom : public mm2::UserIoNet ///< abstract base
        {
            virtual std::string name() const;
            virtual std::string shortname() const { return std::string("Random"); }
            IoNetRandom( uint32_t const verts, float accept = 0.5f )
                : mm2::UserIoNet(verts)
                    , accept(std::max(0.0f,std::min(accept,1.0f)))
            {}
            /** exponentially distributed out_degree where each destination node has
             * const acceptance probability.  out_degree fixed up be one or more. */
            virtual std::vector<Tnode> mkSend( Tnode const n ) const
            {
                using std::rand;
                using std::srand;
                std::vector<Tnode> ret;
#if OLDIONET == 1
                srand(0);
#else
                //ret.reserve( std::min( accept*1.1f, 1.0f ) * this->verts ); // a lot of times will fit here
                // It is important that the seed depend on n
                srand(uint32_t(32768U*accept) + n*31U );
#endif
                for (int r = 0; r < verts; ++r){
                    if( r==n )                                  // ignore self-loops
                        continue;
                    if( float(rand()) / RAND_MAX < accept )     // accept with some prob
                        ret.push_back(r);
                }
#if OLDIONET == 0
                if( ret.size() == 0U ) {
                    uint32_t nn = n + 1U + uint32_t(float(std::rand())/RAND_MAX * (verts-1U));
                    if( nn >= verts ) nn -= verts;
                    assert( /* nn >= 0 && */ nn != n && nn < verts );
                    ret.push_back( nn );
                }else{
                    // ret is now monotonically increasing, not great if everybody sends
                    // only to their first few guys!  Improve by random permutation.
                    std::random_shuffle( ret.begin(), ret.end() );
                }
#endif
                // ret should already have no self loops and no duplicate arcs
                return ret;
            }
            float const accept;
        };

        /** parameter server has all nodes sending to node 0, 0 sends to all others */
        struct IoNetParaServer : public mm2::UserIoNet
        {
            virtual std::string name() const;
            virtual std::string shortname() const { std::string s("Paraserver"); return s; }
            IoNetParaServer( uint32_t const verts ) : mm2::UserIoNet(verts) {}
            virtual std::vector<Tnode> mkSend( Tnode const n ) const {
                std::vector<Tnode> ret;
                if( n == 0U ){
                    for( Tnode i=1U; i<verts; ++i ){
                        ret.push_back(i);
                    }
                }else{
                    ret.push_back(0U);
                }
                return ret;
            }
        };
    }//user::
}//mm2::

#endif // USERIONET_HPP
