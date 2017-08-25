
#include "shm/layout.hpp"

#include <iostream>
#include <stdexcept>
#include <cstring>              // std::memcpy
#include <type_traits>

#include <malloc.h>             // cygwin needed this
using namespace std;
using namespace layout;

/** Wrap a particular layout object and provide easy access.
 * - Example:  create an 'inline' vector of n=2 floats
 *\verbatim
 MemLay m;
 uint32_t n = 2U;
 auto oSz   = m.lay.add<uint32_t>();    // First region is 1 uint32_t
 auto oData = m.lay.add<float>(n);      // Second region is n floats

 m.setMem( malloc(m.lay.size()) );      // make m dereferencable, to proper types, from Osz and oData Regions

 *(m+oSz) = n;                          // dereference via (MemLay + Region), returns ptr to correct type.

 for( uint32_t i=0U; i<n; ++i ) {
    (m+oData)[i] = sqrt(float(i));      // every Region is potentially an array.
 }
 ...
 free(m); m->setMem(nullptr);
 \endverbatim
 * Note that in some cases you \em might not need to store 'n', since the capacity
 * of a Region in Region::data_type units is always obtainable as
 * \c m.lay.cap(oData) or \c layout::cap(m.lay,oData)
 *
 * The capacity might be greater than original due to later \c extend, \c pad or \p align
 * \c Lay operations, but by default, \c Lay offsets are assigned fully-packed.
 */
template< int const Tag >
class MemLay {
private:
    void *mem;                  ///< we never free this
public:
    layout::Lay<Tag> lay;
    MemLay()
        : mem(nullptr)
          , lay()
    {}
    MemLay(void *base, layout::Lay<Tag> const src )
        : mem(base)
          , lay(src)
    {}
    ~MemLay() {}
    // passing in mem to own not supported (don't known which allocate/free pair was used)
    /** After a setMem, lay should not be changed, and operator+ becomes usable.
     * - ? enforce lay size doesn't change after this ? (at least in debug mode)
     */
    void setMem( void *base ) { mem=base; }
    void releaseMem() { mem=nullptr; }

    void malloc() { assert( mem==nullptr ); mem=malloc(lay.size()); }
    void free() { if(mem) free(mem); this->mem = nullptr; }

    /** access region handles returned by lay.add(*) calls via operator+. */
    template< typename T >
        typename layout::Lay<Tag>::template Region<T>::data_type *
        operator+( typename layout::Lay<Tag>::template Region<T> const r )
        {
            assert( mem != nullptr );
            return layout::ptr( mem, lay, r );
        }
};


int main(int,char**)
{
    int i = 1234U;
    {
        Lay<7> lay;
        auto oDims = lay.add<float>(4); // 4 float "dimensions"
        cout<<" 4 float dimensions: lay.size()     = "<<lay.size()<<endl;
        cout<<"                     lay.off(oDims) = "<<lay.off(oDims)<<endl;
        void * base = malloc(lay.size());
        if( base == nullptr ) throw std::runtime_error(" memory allocation failed");
        auto dims = ptr( base, lay, oDims );
#if 0
        {
            Lay<8> lay2;
            auto dims2 = ptr( base, lay2, oDims );
            // GOOD:
            // " No matching function for call to ‘layout::Lay<8>::off(const layout::Lay<7>::Region<float>&) const’"
            // and static_assertion message.
        }
#endif
        cout<<"\t\t base mem @ "<<base<<" pDims @ "<<(void*)dims<<endl;
        assert( base == dims );
        assert( lay.size() == 4 * sizeof(float) );
        free(base);
    }
    {
        cout<<"Testing two regions... "; cout.flush();
        Lay<7> lay;
        auto oDims = lay.add<float>(3U,16U/*initial offset alignments*/);
        assert( lay.size(oDims) == 3U * sizeof(float) );
        assert( lay.size() == 3U * sizeof(float) );
        auto oVals = lay.add<float>(3U,64U/*initial offset alignment*/);
        // oDims and oVals are disambiguated by their offset parameter.
        assert( oDims.i == 0U );
        assert( oVals.i == 1U );
        assert( lay.off(oDims) == 0U );
        assert( lay.off(oVals) == 64U );
        //  NOTE:
        //      even though oDims was defined as representing 3 floats,
        //      we have FORGOTTEN the 3, since we padded the region upward
        //      to a bigger alignment of 64U.
        assert( lay.size( oDims ) == 64U );      // <-- now this is larger
        assert( lay.size( oDims ) > 3U * sizeof(float) );
        //
        assert( lay.size() == 64U + 3U * sizeof(float) );
        cout<<" PASSED"<<endl;
    }
    {
        cout<<"Testing object construction: vector... "; cout.flush();
        vector<float> v{1.1,2.2,3.3,4.4,5.5};
        Lay<333> lay;
        auto oDim = lay.add<uint32_t>();
        auto oData = lay.add<float>(v.size());
        // allocate and initialize a raw memory for Lay<333>
        void * base = malloc(lay.size());
        *ptr(base,lay,oDim) = v.size();
        std::memcpy((void*)ptr(base,lay,oData), &v[0], v.size() * sizeof(v[0]));
        // assert content matches original vector.
        assert( *ptr(base,lay,oDim) == 5U );
        assert( ptr(base,lay,oData)[0] == 1.1f );
        assert( ptr(base,lay,oData)[1] == 2.2f );
        assert( ptr(base,lay,oData)[2] == 3.3f );
        assert( ptr(base,lay,oData)[3] == 4.4f );
        assert( ptr(base,lay,oData)[4] == 5.5f );
        free(base);
        cout<<" PASSED"<<endl;
    }
    {
        cout<<"Testing object construction: class with variable-length array... "; cout.flush();
        struct Obj {
            Obj() {}            ///< required to avoid zero-initialization default ?
            uint32_t a;
            uint32_t b;
            uint32_t n;
            /** if you encounter a compiler without the zero-length array hack,
             * I read that data[1] should work as well, as long as you modify
             * the treatment below (extend by variable_arrar_size - 1) */
            float data[0];
        };
        // allocate and initialize an Obj
        Obj *x = (Obj*) malloc( sizeof(Obj) + 5U*sizeof(float) ); // allocate as if "float data[6];"
        x->a = 0U;
        x->b = 1U;
        x->n = 3U;
        x->data[0] = 1.1f;
        x->data[1] = 2.2f;
        x->data[2] = 3.3f;
        // Given Obj, define the layout
        assert( std::is_standard_layout<Obj>() ); // offsetof REQUIRES is_standard_layout
        Lay<333> lay;
        auto oObj = lay.add<Obj>();
        assert( oObj.i == 0U );                   // requirement for is_standard_layout<Obj>
        assert( lay.size() == 12U );
        assert( lay.size() == 3U * sizeof(uint32_t) );
        cout<<" lay.size() = "<<lay.size()<<endl;
        // "lay.size() = 12"  is wrong. It has NO contribution for data[n] at all.
        uint32_t variable_array_size = x->n;
        assert( variable_array_size == 3U );
        assert( variable_array_size > 0U );
        cout<<" variable_array_size = "<<variable_array_size<<"   lay.size() = "<<lay.size()<<endl;
        lay.extend( (variable_array_size) * sizeof(float) );   // extend Region capacity for Obj by 2 extra floats
        assert( lay.size() == 3*sizeof(uint32_t) + 3*sizeof(float) );
        // at this point Lay and Obj really match !
        {
            void *mem = x;
            assert( x->b == 1U );
            Obj *y = new(ptr(mem,lay,oObj)) Obj(); // placement-new, POD, so no mem should be changed
            assert( x->b == 1U );
            // compare for no-change over FULL size of data[]
            // (note: x will have changed too, so aseert item-by-item)
            assert( y->a == 0U );
            assert( y->b == 1U );
            assert( y->n == 3U );
            assert( y->data[0] == 1.1f );
            assert( y->data[1] == 2.2f );
            assert( y->data[2] == 3.3f );
        }
        free(x);
        cout<<" PASSED"<<endl;
    }
    {
        cout<<"Testing MemLay     ... "; cout.flush();
        vector<float> v{1.1,2.2,3.3,4.4,5.5};
        MemLay<333> m;
        auto oDim = m.lay.add<uint32_t>();
        auto oData = m.lay.add<float>(v.size());
        // allocate and initialize a raw memory for Lay<333>
        void * base = malloc(m.lay.size());
        m.setMem(base);
        assert( v.size() == 5U );
        assert( (void*)(m+oDim) == base );
        *(m+oDim) = v.size();
        (m+oData)[0] = v[0];
        (m+oData)[1] = v[1];
        (m+oData)[2] = v[2];
        (m+oData)[3] = v[3];
        (m+oData)[4] = v[4];
    }

    cout<<__FILE__<<" tests passed.\nGoodbye"<<endl;
}
