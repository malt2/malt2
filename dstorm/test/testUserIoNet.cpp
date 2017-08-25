/** \file testUserIoNet.cpp
 * test basic User, Scalable and Full Io network classes. */
#include "ionet/userIoNet.hpp"
#include "ionet/globIoNet.hpp"
#include "ionet/scalIoNet.hpp"

#include <iomanip>
using namespace std;

/** supposed to be in C++11, but apparently got delayed to C++14? */
template<typename T, typename... Args> static inline
std::unique_ptr<T> make_unique(Args&&... args) {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

static inline unsigned pow2_le( unsigned const n ){
    return (1U<<mm2::user::lg2(n));        // there are better ways
}

int main(int,char**)
{
    cout<<" This is a quick test that some basic IoNet stuff compiles and runs"<<endl;
    using mm2::user::IoNetEmpty;
    using mm2::user::IoNetAll;
    using mm2::user::IoNetChord;
    using mm2::user::IoNetHalton;
    using mm2::user::IoNetRandom;
    using mm2::user::IoNetStream;
    using mm2::user::IoNetParaServer;

    IoNetEmpty      uempty(8);   cout<<uempty .pretty()<<endl;
    IoNetAll        uall(8);     cout<<uall   .pretty()<<endl;
    IoNetChord      uchord(8);   cout<<uchord .pretty()<<endl;
    IoNetHalton     uhalton(8);  cout<<uhalton.pretty()<<endl;
    IoNetRandom     urandom(8);  cout<<urandom.pretty()<<endl;
    IoNetStream     ustream(8);  cout<<ustream   .pretty()<<endl;
    IoNetParaServer ups(8);      cout<<ups    .pretty()<<endl;
    
    using mm2::GlobNet;
    GlobNet gempty (std::unique_ptr<IoNetEmpty     >(new IoNetEmpty     (8))); cout<<gempty .pretty()<<endl;
    GlobNet gall   (std::unique_ptr<IoNetAll       >(new IoNetAll       (8))); cout<<gall   .pretty()<<endl;
    GlobNet gstream(std::unique_ptr<IoNetStream     >(new IoNetStream    (8))); cout<<gstream   .pretty()<<endl;
    GlobNet gchord (std::unique_ptr<IoNetChord     >(new IoNetChord     (8))); cout<<gchord .pretty()<<endl;
    GlobNet ghalton(std::unique_ptr<IoNetHalton    >(new IoNetHalton    (8))); cout<<ghalton.pretty()<<endl;
    GlobNet grandom(std::unique_ptr<IoNetRandom    >(new IoNetRandom    (8))); cout<<grandom.pretty()<<endl;
    GlobNet gps    (std::unique_ptr<IoNetParaServer>(new IoNetParaServer(8))); cout<<gps    .pretty()<<endl;
    //GlobNet grand2 (std::unique_ptr<IoNetRandom    >(new IoNetRandom(8,0.9f))); cout<<grand2 .pretty()<<endl;
    //  nicer construction syntax:
    GlobNet grand2 (make_unique<IoNetRandom>(8,0.9f)); cout<<grand2 .pretty()<<endl;
    
    cout<<setw(10)<<"|";
    for(uint32_t v=1U; v<20U; ++v)
        cout<<" "<<setw(2)<<v;
    cout<<endl;
    for(uint32_t v=1U; v<30U; ++v){
        bool const verbose = true;
        using mm2::user::halton;
        using mm2::user::lg2;
        if(verbose) cout<<" verts "<<setw(2)<<v<<"|";
        for( uint32_t r=0U; r<v; ++r ){
            uint32_t halton2r = uint32_t( v * halton(2,r) + 0.01 );
            if(verbose) cout<<" "<<setw(2)<<halton2r;
            if( uint32_t( v * halton(2,r) + 0.01 ) == 1U ){
                if(0 && verbose){
                    cout<<"            v = "<<setw(3)<<v<<":  halton( 2,"<<setw(3)<<r<<" ) --> 1";
                    cout<<" ?? "<<( pow2_le(v)/2U - 1U );
                }
                // Also conjecture Halton sequence is cyclic only for v = power of two
                if( r < v ) assert( r == pow2_le(v)/2U - 1U );
            }
        }
        cout<<endl;
    }
    if(1){
        GlobNet ghalt2(make_unique<IoNetHalton>(8,8)); cout<<ghalt2.pretty()<<endl;
        for(uint32_t v=2U; v<16U; ++v){
            GlobNet ghalt3(make_unique<IoNetHalton>(v,v));
            cout<<ghalt3.pretty()<<endl;
        }
    }

    cout<<uchord .pretty()<<endl;
    // compare... recvlist for node 3 should be 1 2 7
    mm2::detail::ScalSendNet sschord(3U, make_unique<IoNetChord>(8)); cout<<sschord.pretty()<<endl;
    mm2::ScalNet schord(3U, make_unique<IoNetChord>(8)); cout<<schord.pretty()<<endl;
    assert( schord.recv().size() == 3U );
    assert( schord.recv()[0] == 1U );
    assert( schord.recv()[1] == 2U );
    assert( schord.recv()[2] == 7U );

    // Copy constructor tests.
    // Useful to copy UserIoNet's to get at the GlobIoNet::pretty() printer
    //mm2::UserIoNet *a = &uhalton;
    // mm2::UserIoNet *b = new mm2::UserIoNet(*a); // cannot allocate abstract type

    cout<<"\nGoodbye"<<endl;
}
