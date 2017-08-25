
//#include "mmIoNet.hpp"
#include "ionet/globIoNet.hpp"
#include "ionet/scalIoNet.hpp"
//#include "dstorm_fwd.hpp"                  // ALL, HALTON OldIoNetEnum constants

#include <string.h>     // strncmp
#include <sstream>
#include <stdexcept>
#include <iomanip>
#include <algorithm>

using namespace std;
using namespace dStorm;         // dStorm::IoNet_t

template<typename T, typename... Args> static inline
std::unique_ptr<T> make_unique(Args&&... args) {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

namespace mm2 {
    namespace user {
        //
        // ------------ helpers to print IoNet class names ---------------
        //
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

        //
        // ------------ define some additional IoNets ------------------
        //
        /** HALTON_ASIM */
        class IoNetHalton_Asim : public mm2::UserIoNet ///< abstract base
        {
        public:
            IoNetHalton_Asim( uint32_t const verts, uint32_t max_degree = -1U )
                : mm2::UserIoNet(verts)
                  , degree( std::min(verts-1U,default_lg2(verts,max_degree)))
            {}
            virtual std::string name() const;
            virtual std::string shortname() const { return std::string("Halton_Asim"); }
            std::vector<Tnode> mkSend( Tnode const n ) const {
                std::vector<Tnode> ret;
                ret.reserve( this->degree );
                unsigned base = ((verts & (verts-1)) == 0 // power of two
                                 ? 3U : 2U);
                for(Tnode r=0u; r<this->degree; ++r){
                    uint32_t dst = (n + (uint32_t)(verts * halton(base,r)+0.01)) % verts;
                    if( dst==n )                              // no self-loops
                        continue;
                    ret.push_back(dst);
                }
                return ret;
            }
            uint32_t const degree;      // degree == max_degree
        };
        inline std::string IoNetHalton_Asim::name() const {
            auto default_degree = IoNetHalton_Asim(this->verts).degree;
            return classVertsParm( "Halton_Asim", verts, degree, default_degree );
        }

        /** return a cyclic permutation of numbers [0..len-1] beginning
         * and ending at '0' */
        std::vector<unsigned> halton_perm( unsigned const base, unsigned const len )
        {
            typedef typename std::pair<unsigned,double> Vweight;
            std::vector<Vweight> vw; // assign weights of halton seq to vertices
            vw.reserve(len);
            for( unsigned i=0U; i<len; ++i )
                vw.push_back( Vweight{i,halton(base,i)} );
            std::sort(vw.begin(),vw.end()
                      , []( Vweight const& a, Vweight const& b ){
                      return a.second < b.second; }
                      );
            std::vector<unsigned> ret;
            ret.reserve(len);
            for( auto const p: vw ) ret.push_back(p.first);
            if(0){
                cout<<" halton_perm("<<base<<","<<(unsigned)len<<")={";
                for( auto const u: ret ) cout<<" "<<u;
                cout<<" }"<<endl;
            }
            return ret;
        }

        /** HALTON_EJK.
         * OK, but really needs a global view to resolve "clashes".
         * - mkSend(n) (row-wise) as Halton gives bad networks (lots of
         *   clashes if used as cyclic offset)
         * - better (Here) is Halton permutations column-wise.
         *   but still should look at global matrix and choose which
         *   cycle of the permutation leads to fewest clashes with
         *   previous column.
         */
        class IoNetHalton_ejk : public mm2::UserIoNet ///< abstract base
        {
        public:
            IoNetHalton_ejk( uint32_t const verts, uint32_t max_degree = -1U )
                : mm2::UserIoNet(verts)
                  , degree( std::min(verts-1U,default_lg2(verts,max_degree)))
            {}
            virtual std::string name() const;
            virtual std::string shortname() const { return std::string("Halton_ejk"); }
            std::vector<Tnode> mkSend( Tnode const n ) const
            {
                std::vector<Tnode> ret;
                ret.reserve( this->degree );
                // perms generate cols of sendList matrix
                std::vector<unsigned> perm[3];
                {
                    // halton_perm is OK for any base.
                    unsigned primes[3U] = {5U,3U,2U};
                    for(uint32_t i=0U; i<3U; ++i){
                        perm[i] = halton_perm( primes[i%3U], verts );
                        //cout<<" perm["<<i<<"={";for( auto p: perm[i] ) cout<<" "<<p; cout<<endl;
                    }
                }
                for(uint32_t i=0U; i<degree; ++i ){
                    // connect node n to n+ii, where ii is i'th item of one of the cyclic perms
                    uint32_t add = perm[ i%3U ][ n ];
                    //cout<<" perm["<<i%3U<<"]["<<n<<"]="<<add;
                    //Tnode cyc = add;
                    Tnode cyc = (i < verts-add ? i+add: i-(verts-add));
                    ret.push_back( cyc );
                }
                return ret;
            }
            uint32_t const degree;      // degree == max_degree
        };
        inline std::string IoNetHalton_ejk::name() const {
            auto default_degree = IoNetHalton_ejk(this->verts).degree;
            return classVertsParm( "Halton_ejk", verts, degree, default_degree );
        }

    }//user::
}//mm2::
void usage(){
    cout<<" Usage:  prtIoNet {test | verts | test verts} [name [,args] TBD]\n"
        <<"    test:  more output (ideas for new graphs)\n"
        <<"    verts: # N in range 0-80 of graph vertices (or an  N-M  range)\n"
        <<" Example: prtIoNet 5\n"
        //<<"          - prints a 5x5 grid of send lists with dest\n"
        //<<"            priorities marked with hex code, then 'os's\n"
        <<"          - n should be between 1 and 80\n"
        <<" Example: prtIoNet test\n"
        <<"          - reduce output, covers a range of verts from 1 to some big value\n"
        <<" Example: prtIoNet test 10\n"
        <<"          - reduce output, only for 10 verts\n"
        <<" Example: prtIoNet 1-20\n"
        <<"          - print for a range of graphs sizes\n"
        <<endl;
}

int main(int argc,char** argv){
    uint32_t do_n=0U;
    uint32_t do_n2=0U;
    uint32_t do_tst=false;
    char sep;
    // this arg parsing DOES NOT extend to provided graph names and graph constructor arguments nicely
    if( argc == 2U ) {
        if( strncmp(argv[1],"test",4) == 0U){
            do_tst=true;
            do_n=2U;
            // NOTE: GlobNet for n==1 FAILS ? (fix this assertion)
        }else{
            istringstream is( argv[1] );
            is >> do_n >> sep >> do_n2 ;
        }
    }else if( argc == 3U ) {
        if( strncmp(argv[1],"test",4) == 0U){
            do_tst=true;
            do_n=2U;
        }else{
            usage();
            return 0;
        }
        istringstream is( argv[2] );
        is >> do_n >> sep >> do_n2 ;
    }else{
        usage();
        return 0;
    }

    if( (do_n<=0U || do_n>80U) ){
        usage();
        return -1;
    }

    cout<<" do_tst = "<<do_tst<<"     do_n = "<<do_n<<"     d_n2 = "<<do_n2<<endl;
    uint32_t nlo = do_n;
    uint32_t nhi = (do_tst? 65U: do_n+1U);
    if( do_n2 ){
        nhi = std::max( do_n, do_n2 ) + 1U;
    }

    cout<<" nlo, nhi = "<<nlo<<" , "<<nhi<<endl;
    using namespace mm2;            // GlobNet, UserIoNet
    using namespace mm2::user;      // IoNetAll/Chord/Halton/...
    for( uint32_t n=nlo; n<nhi; ++n)
    {
        vector< GlobNet * > iographs;

        // do not change order -- they correspond to oldIoNetENUM values in mmIoNet.hpp
        iographs.push_back( new GlobNet(make_unique<IoNetAll       >(n)) );
        iographs.push_back( new GlobNet(make_unique<IoNetChord     >(n)) );
        iographs.push_back( new GlobNet(make_unique<IoNetHalton    >(n)) );     // opt. maxdegree
        iographs.push_back( new GlobNet(make_unique<IoNetRandom    >(n)) );
        iographs.push_back( new GlobNet(make_unique<IoNetParaServer>(n)) );
        iographs.push_back( new GlobNet(make_unique<IoNetEmpty     >(n)) ); // for BUTTERFLY

        if(!do_tst) cout<<" 'new' iographs ... DONE"<<endl;

        //vector<OldIoNetEnum> graphTypes = {ALL, CHORD, HALTON
        //    , RANDOM, PARA_SERVER, BUTTERFLY};

        //for(auto g: graphTypes)
        for(auto const& g: iographs)
        {
            // check for equivalence: if so, elide duplicate 'new' output
            //  -- removed: milde_malt2 has no OldIoNetEnum stuff.
            //  -- do_tst maybe just produces more output ?
            try {
                cout<<" ----- prtIoNet:"<<g->name();
                cout<<g->pretty()<<endl;
            }catch( std::exception& e ){
                cout<<" ERROR: "<<e.what();
            }catch(...){
                throw std::runtime_error(" ERROR: Huh");
            }
            cout<<endl;
        }
        if(do_tst) cout<<" n="<<n<<" DONE:  calling iographs[] destructors ..."<<endl;
    }
    //
    // Add custom IoNet printouts here
    //
    cout<<"\n**** Custom Ionets ****\n";
    if(do_tst) for( uint32_t n=nlo; n<nhi; ++n)
    {
        vector< GlobNet * > iographs;
        using namespace mm2::user;
        iographs.push_back( new GlobNet(make_unique<IoNetHalton_Asim>(n)));
        size_t const asim1 = iographs.size()-1U;
        cout<<iographs[asim1]->pretty_raw()<<endl;
        cout<<iographs[asim1]->pretty()<<endl;

        iographs.push_back( new GlobNet(make_unique<IoNetHalton_ejk >(n)));
        size_t const ejk1 = iographs.size()-1U;
        cout<<iographs[ejk1]->pretty_raw()<<endl;
        cout<<iographs[ejk1]->pretty()<<endl;
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


