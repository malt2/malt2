/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef LAYOUT_HPP_
#define LAYOUT_HPP_

#include <cstdint>
#include <vector>
#include <cassert>

#ifndef NDEBUG
#include <iostream>
#endif

/** utilities to lay out objects in memory regions as offsets */
namespace layout {

    /** <em>vector of offsets</em> helper for laying out objects.
     *
     * \c Lay generates a sequences of increasing offsets.
     * \tparam differentiates Layouts of different things.
     * \c add results returned for one Lay cannot be used
     * to obtain offsets from another Lay object.
     *
     * - You call \c add with a \b type and a count, and get back a
     *   strongly typed storage offset handle.
     * - Later, via the same \c Lay class, this can be converted
     *   back to a byte offset.
     * - Or directly to \b type* pointer if you supply a base address,
     *   (such as you might have for a shared memory region).
     *
     * - WARNING: If allocating objects, we do not provide introspection.
     *   - so if you plan to placement-new objects, you had better
     *     check with compatibility of \em offsetof and the layout handles!
     *   - OR assert( std::is_standard_layout<Obj>() ) and hand-correct
     *     \c extend the size of a terminal zero-length array upward.
     *
     * - NOTE: Consider a Lay32 version that replaces the vector
     *       within a std::array<uint32_t,N> plus a size variable <= N
     *   - Benefits:
     *     - no malloc of std::vector
     *     - fixed, small memory for byte-offset vector
     *     - can be placed into shared memory
     *   - Disadvantages:
     *     - support max 4G total size
     *     - support only some fixed number (ex. 8) of Regions per Layout.
     */
    template<int const Tag>
        class Lay {
        private:
            /** vector of byte offsets. Byte offsets begin at 0.
             * The i'th object (or vector) should begin at v[i]
             * and should end before byte v[i+1]. */
            std::vector<std::size_t> v;
        public:
            Lay() : v(1/*count*/,0/*value*/) {}
            ~Lay() {}
            std::size_t size() const { return v.back(); }
            /** return value from \c add is a handle convertible to a typed pointer.
             *
             * - Use \c ptr(void* base, Lay const& layout, Region const region )  to
             *   do the type-checked pointer conversions. 
             * - Use \c cap(layout,region) to get the maximum array-offset usable for
             *   a \c ptr(base,layout,region).
             * - In debug mode, all accesses \c ptr(base,layout,region)[k] will do
             *   a runtime check that \c k does not exceed the regions capacity.
             */
            template< typename T > struct Region {
                uint_least8_t const i;          ///< index into private vector of offsets, Lay<T>::v
                static int const tag = Tag;
                typedef T data_type;
            };
            /// @name modifying functions
            //@{
            /** take current total size and \b align it upward */
            std::size_t pad( std::size_t const align) {
                std::size_t upward = (v.back() + align-1U)/align * align;
                return v.back() = upward;
            }
            /** extend size of last \c add by arbitrary \c padBytes.
             * This routine adds to the current total size.
             *
             * \ref layout-demo.hpp has an example usage to mimic variable-length
             * arrays such as you might have at runtime with the zero-length array trick.
             * This can be used to embed a runtime-sized vector within an object.
             *
             * For example, if \c n is a runtime value for the length of a terminal
             * <TT>float data[0]</tt> array, then the compiler uses 1 float for the \em size
             * of \c data.  This needs to be padded upward by \c n-1 floats.  So you
             * would call Lay::pad( (n-1U) * sizeof(float) ) to get the correct layout.
             */
            std::size_t extend( std::size_t const padBytes) {
                return v.back() += padBytes;
            }
            /** Basic modifying function */
            template< typename T > Region<T>
                add( std::size_t const n = 1U ) {
                    assert( v.size() < 255U );
                    Region<T> ret{ static_cast<uint_least8_t>(v.size()-1U) };
                    v.push_back( v.back() + sizeof(T)*n );
                    return ret;
                } 
            /** add n \em packed objects of sizeof(T), \em beginning at an aligned offset */
            template< typename T > Region<T>
                add( std::size_t const n, std::size_t const align ) {
                    assert( v.size() < 255U );
                    Region<T> ret{static_cast<uint_least8_t>(v.size()-1U)};
                    v.push_back( pad(align) + sizeof(T) * n );
                    return ret;
                }
            //@}
            /// @name const getters for byte offset / size info
            //@{
            /** beginning byte of region handle \c i. */
            template< typename T > std::size_t
                off( Region<T> const i ) const {
                    static_assert( i.tag == Tag, " incorrect Region handle for layout::Lay");
                    assert( i.i < v.size() - 1U );
                    return v[i.i];
                }
            /** beginning byte of \p nth packed item of region handle \p i.
             * \pre \c nth item of i::data_type is within capacity of region \c i. */
            template< typename T > std::size_t
                off( Region<T> const i, std::size_t const nth ) const {
                    static_assert( i.tag == Tag, " incorrect Region handle for layout::Lay");
                    assert( i.i < v.size() - 1U );
                    assert( sizeof(T) * nth < this->size(i) );
                    return v[i.i + sizeof(T) * nth  ];
                }
            template< typename T > std::size_t
                end( Region<T> const i ) const {
                    static_assert( i.tag == Tag, " incorrect Region handle for layout::Lay");
                    assert( i.i < v.size() - 1U );
                    return v[i.i+1U];
                }
            /** size in BYTES of a Region */
            template< typename T > std::size_t
                size( Region<T> const i ) const {
                    static_assert( i.tag == Tag, " incorrect Region handle for layout::Lay");
                    assert( i.i < v.size() - 1U );
                    return v[i.i+1U] - v[i.i];
                }
            /** capacity in T's that could fit in a Region */
            template< typename T > std::size_t
                cap( Region<T> const i ) const {
                    static_assert( i.tag == Tag, " incorrect Region handle for layout::Lay");
                    assert( i.i < v.size() - 1U );
                    return (v[i.i+1U] - v[i.i]) / sizeof(T);
                }
            //@}

        };
#if 0
    // problems with type-deduction
    template< typename T, int Tag >
        T* ptr( void* base,
                Lay<Tag> const& layout,
                typename Lay<Tag>::template Region<T> const region )
        {
            T* p = reinterpret_cast< T* >( (uint_least8_t*)base + layout.off(region) );
        }
#endif
    /** what's the max array index usable for a T* return value from \c layout::ptr? */
    template< typename Region, int const Tag > typename Region::data_type *
        cap( Lay<Tag> const& layout,
             Region const region )
        {
            static_assert( Region::tag == Tag, "incorrect Region index into layout::Lay layout" );
            typedef typename Region::data_type T;
            return layout.size(region) / sizeof(T);
        }
#if defined(NDEBUG) || defined(TEST_OPT)
    /** Naive transform of \c base memory ptr via layout \c Lay into a a \c T* pointer.
     *
     * Use of the returned pointer as a C-array is \b unchecked.  We have an alternate
     * implementation for debug mode that allows \c operator[](k) to check that
     * \c k does not exceed the region capacity.
     *
     * Compile with -DTEST_OPT to use this optimized implementation within a debug compile.
     */
    template< typename Region, int const Tag > typename Region::data_type *
        ptr( void* base,
             Lay<Tag> const& layout,
             Region const region )
        {
            static_assert( Region::tag == Tag, "incorrect Region index into layout::Lay layout" );
            typedef typename Region::data_type T;
            //typename Lay<Tag>::template Region<T> &i = region;
            T* p = reinterpret_cast< T* >( (uint_least8_t*)base + layout.off(region) );
            return p;
        }
#else
    /** a debug-mode wrapper for a pointer to \c n objects of type T, where
     * \c n is a runtime value.  \c n is used internally for \c always asserting
     * that operator[](k) on the return value of layout:ptr satisfies <TT>k < n</tt>. */
    template< class T >
        class RegionPtr {
        private:
            T* ptr;                         ///< pointer
        public:
            size_t const cap;               ///< capacity for operator[] check
            explicit RegionPtr( T* ptr, size_t cap ) : ptr(ptr), cap(cap)
            {
                //std::cout<<" RegionPtr@"<<(void*)ptr<<std::endl;
            }
            operator T*() const {return ptr;}        ///< auto-converts into plain pointer
            T& operator[]( size_t  const k ) {
                assert( k < cap );
                return ptr[k];
            }
        };

    /** debug implementation of ptr checks operator[](k) on the ptr return value
     * would not exceed the region's capacity. */
    template< typename Region, int const Tag > RegionPtr<typename Region::data_type>
        ptr( void* base,
             Lay<Tag> const& layout,
             Region const region )
        {
            static_assert( Region::tag == Tag, "incorrect Region index into layout::Lay layout" );
            typedef typename Region::data_type T;
            typedef RegionPtr<T> VptrT;
            //typename Lay<Tag>::template Region<T> &i = region;
            T* p = reinterpret_cast< T* >( (uint_least8_t*)base + layout.off(region) );
            return RegionPtr<typename Region::data_type>( /*pointer       */ p,
                                                          /*array capacity*/ layout.size(region) / sizeof(T) );
        }
#endif
#if 0
    // Notice that associating layout with a base addr really need only be done once
    // per process (or perhaps per thread, if not passed down from master thread).
    // Also note that the Type info (but not the array sizes) are all known at compile time.
    // So we could store the entire set of Region handles, the base pointer
    // all within one big metaclass to ease use.
    template< int const Tag, ... >
        class Rooted : public const Lay<Tag> {
        public:
            explicit Rooted( Lay<Tag> const& layout, void* base )
                : Lay<Tag>( layout )
                  , base(base)
            {
                assert( base != nullptr );
            }
            template< typename Region > RegionPtr<typename Region::data_type>
                ptr( region )
        };
#endif
}//layout::
#endif // LAYOUT_HPP_
