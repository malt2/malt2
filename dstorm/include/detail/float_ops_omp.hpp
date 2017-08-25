/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef FLOAT_OPS_HPP
#define FLOAT_OPS_HPP
/** \file float_ops.hpp
 * Do various \em set and \em avg operations on float* vectors in a way
 * that can be efficiently optimized.
 */

/** RDMA-friendly fast vector averaging: set, update, weighted update.
 *
 * These routines assume disjoint vectors, i.e. no pointer aliasing.
 *
 * - For <B>fset_avg_restrict</B>* and <B>fupd_avg_restrict</B>* functions,
 *   <B>the modified \c dest vector is modified directly to the final value</B>.
 *   - Why?
 *     - RDMA algorithms are often OK with getting version n or n-1 or n+1 or some
 *       mix of versions of a vector,
 *       - and this happens quite often.  Even if you try 'weak' synchronization
 *         like notify/waitsome.
 *     - temporaries, like forming a sum in one loop, and scaling in a second
 *       loop <em>have the wrong scale</em>.
 *       - Even if the temporary is there just ... temporarily ..., it can
 *         totally screw up an algorithm -- like diverging, or NaNs.
 *
 * Note that implementing these averages using blas/MKL or several simpler
 * libraries may use temporaries.  If we switch to blaze or a decent C++
 * meta-expression library, this would be OK too.
 *
 * \todo openmp support for floatops.hpp (or switch to blaze-lib)
 */
#include <cstdint>
namespace favg {
    /** \name favg:: specialized 'set' helpers -- set dest to average of input vectors.
     * \f$\vec{d} = \sum_{i=0}^{\mathrm{N}-1} \vec{s}_i / \mathrm{N}\f$, for N=2..6 */
    //@{
    template<typename T> inline void set_restrict2( T* __restrict__ dest, uint32_t const cnt,
                                                    T const* __restrict__ s1, T const* __restrict__ s2 ){
#pragma omp parallel for
        for(uint32_t i=0U; i<cnt; ++i){
            dest[i] = 0.5 * ( s1[i] + s2[i] );
        }
    }
    template<typename T> inline void set_restrict3( T* __restrict__ dest, uint32_t const cnt,
                                                    T const* __restrict__ s1, T const* __restrict__ s2,
                                                    T const* __restrict__ s3 ){
#pragma omp parallel for schedule(static,896)
        for(uint32_t i=0U; i<cnt; ++i){
            dest[i] = T(0.333333333333333333) * ( s1[i] + s2[i] + s3[i] );
        }
    }
    template<typename T> inline void set_restrict4( T* __restrict__ dest, uint32_t const cnt,
                                                    T const* __restrict__ s1, T const* __restrict__ s2, 
                                                    T const* __restrict__ s3, T const* __restrict__ s4 ){
#pragma omp parallel for schedule(static,768)
        for(uint32_t i=0U; i<cnt; ++i){
            dest[i] = T(0.25) * ( s1[i] + s2[i] + s3[i] + s4[i] );
        }
    }
    template<typename T> inline void set_restrict5( T* __restrict__ dest, uint32_t const cnt,
                                                    T const* __restrict__ s1, T const* __restrict__ s2,
                                                    T const* __restrict__ s3, T const* __restrict__ s4,
                                                    T const* __restrict__ s5 ){
#pragma omp parallel for schedule(static,640)
        for(uint32_t i=0U; i<cnt; ++i){
            dest[i] = T(0.20) * ( s1[i] + s2[i] + s3[i] + s4[i] + s5[i] );
        }
    }
    template<typename T> inline void set_restrict6( T* __restrict__ dest, uint32_t const cnt,
                                                    T const* __restrict__ s1, T const* __restrict__ s2,
                                                    T const* __restrict__ s3, T const* __restrict__ s4,
                                                    T const* __restrict__ s5, T const* __restrict__ s6 ){
#pragma omp parallel for schedule(static,512)
        for(uint32_t i=0U; i<cnt; ++i){
            dest[i] = T(0.166666666666666667) * ( s1[i] + s2[i] + s3[i] + s4[i] + s5[i] + s6[i] );
        }
    }
    //@}

    /** \f$\vec{d} = \sum_{i=0}^{\mathrm{ssz}-1} \vec{s}_i / \mathrm{ssz}\f$.
     * \tparam T is float type of \arg dest[0..cnt-1], and inputs are <tt>T const* s[0..ssz-1]</tt>.
     * Each s[i] ptr-to-array-of-T <em>MAY</em> be <B>modified to point at the END of the vector,
     * i.e. at s[i]+cnt</B>. Use a temporary if you need to preserve <tt>s</tt>'s pointers. */
    template<typename T> inline void set( T* __restrict__ dest, uint32_t const cnt,
                                          uint32_t const ssz, T const* __restrict__ * s ){
        switch(ssz){
          case(0): break;
          case(1): memcpy( dest, *s, cnt*sizeof(T) ); break;
          case(2): favg::set_restrict2( dest, cnt, s[0], s[1] ); break;
          case(3): favg::set_restrict3( dest, cnt, s[0], s[1], s[2] ); break;
          case(4): favg::set_restrict4( dest, cnt, s[0], s[1], s[2], s[3] ); break;
          case(5): favg::set_restrict5( dest, cnt, s[0], s[1], s[2], s[3], s[4] ); break;
          case(6): favg::set_restrict6( dest, cnt, s[0], s[1], s[2], s[3], s[4], s[5] ); break;
          default: { // This simple loop MIGHT actually generate decent code
                       T const sszinv = 1.0 / ssz;
#pragma omp parallel for schedule(static,1024)
                       for(uint32_t i=0U; i<cnt; ++i){
                           T sum = T(0);        // not a more-precise version of T
                           for(uint32_t j=0U; j<ssz; ++j){
                               sum += s[j][i];
                           }
                           // Here is the single write to dest:
                           dest[i] = sum * sszinv;
                       }
                   }
        }
    }

    /** \f$\vec{d} = \sum_{i=0}^{\mathrm{s.size()}-1} \vec{s}_i / \mathrm{s.size()}\f$.
     * T is float type of dest[0..cnt-1], and <tt>S s</tt> is an object
     * with a <tt>T [const] *data</tt> member. <B> This exits with s[i]
     * pointers pointing at the END of the data vector, &s[i][cnt]</b>
     */
    template<typename T> inline void set( T* __restrict__ dest, uint32_t const cnt,
                                          typename std::vector< T const* > const& s ){
        favg::set( dest, cnt, s.size(), &s[0] );
    }


    /** \f$\vec{d} = \sum_{i=0}^{\mathrm{s.size()}-1} s_i.\overrightarrow{\mathrm{data}} / \mathrm{s.size()}\f$.
     * T is float type of dest[0..cnt-1], and <tt>S s</tt> is an object
     * with a <tt>T [const] *data</tt> member. <B> This returns with s[i].data
     * pointers pointing at the END of the data vector, s[i].data[cnt]</b>
     */
    template<typename T, typename S> inline void set( T* __restrict__ dest, uint32_t const cnt,
                                                      std::vector< S >& s ){
        switch(s.size()){
          case(0): break;
          case(1): memcpy( dest, s[0].data, cnt*sizeof(T) ); break;
          case(2): favg::set_restrict2( dest, cnt, s[0].data, s[1].data ); break;
          case(3): favg::set_restrict3( dest, cnt, s[0].data, s[1].data, s[2].data ); break;
          case(4): favg::set_restrict4( dest, cnt, s[0].data, s[1].data, s[2].data,
                                        s[3].data );
                   break;
          case(5): favg::set_restrict5( dest, cnt, s[0].data, s[1].data, s[2].data,
                                        s[3].data, s[4].data );
                   break;
          case(6): favg::set_restrict6( dest, cnt, s[0].data, s[1].data, s[2].data,
                                        s[3].data, s[4].data, s[5].data );
                   break;
          default: { // This simple loop MIGHT actually generate decent code
                       T const sszinv = 1.0 / s.size();
#pragma omp parallel for schedule(static,1024)
                       for(uint32_t i=0U; i<cnt; ++i){            // openmp !!!
                           T sum = T(0);        // not a more-precise version of T
                           for(uint32_t j=0U; j<s.size(); ++j){
                               sum += s[j].data[i];
                           }
                           // Here is the single write to dest:
                           dest[i] = sum * sszinv;
                       }
                   }
        }
    }

    /** \name favg:: specialized 'upd'ate helpers -- update vector dest to average of dest and input vectors,
     * \f$ \vec{d} = \left( \vec{d} + \sum_{i=0}^{\mathrm{N}-1} \vec{s}_i \right) / (1+\mathrm{N})\f$,
     * for N=1..6 */
    //@{
    template<typename T> inline void upd_restrict1( T* __restrict__ dest, uint32_t const cnt,
                                                         T const* __restrict__ s1 ){
#pragma omp parallel for schedule(static,1024)
        for(uint32_t i=0U; i<cnt; ++i){
            dest[i] = T(0.5) * ( dest[i] + s1[i] );
        }
    }
    /** update vector dest to average of dest, s1 and s2, no ptr aliasing */
    template<typename T> inline void upd_restrict2( T* __restrict__ dest, uint32_t const cnt,
                                                         T const* __restrict__ s1, T const* __restrict__ s2 ){
#pragma omp parallel for schedule(static,896)
        for(uint32_t i=0U; i<cnt; ++i){
            dest[i] = T(0.333333333333333333) * ( dest[i] + s1[i] + s2[i] );
        }
    }
    template<typename T> inline void upd_restrict3( T* __restrict__ dest, uint32_t const cnt,
                                                         T const* __restrict__ s1, T const* __restrict__ s2,
                                                         T const* __restrict__ s3 ){
#pragma omp parallel for schedule(static,768)
        for(uint32_t i=0U; i<cnt; ++i){
            dest[i] = T(0.25) * ( dest[i] + s1[i] + s2[i] + s3[i] );
        }
    }
    template<typename T> inline void upd_restrict4( T* __restrict__ dest, uint32_t const cnt,
                                                         T const* __restrict__ s1, T const* __restrict__ s2, 
                                                         T const* __restrict__ s3, T const* __restrict__ s4 ){
#pragma omp parallel for schedule(static,640)
        for(uint32_t i=0U; i<cnt; ++i){
            dest[i] = T(0.2) * ( dest[i] + s1[i] + s2[i] + s3[i] + s4[i] );
        }
    }
    template<typename T> inline void upd_restrict5( T* __restrict__ dest, uint32_t const cnt,
                                                         T const* __restrict__ s1, T const* __restrict__ s2,
                                                         T const* __restrict__ s3, T const* __restrict__ s4,
                                                         T const* __restrict__ s5 ){
#pragma omp parallel for schedule(static,512)
        for(uint32_t i=0U; i<cnt; ++i){
            dest[i] = T(0.166666666666666667) * ( dest[i] + s1[i] + s2[i] + s3[i] + s4[i] + s5[i] );
        }
    }
    template<typename T> inline void upd_restrict6( T* __restrict__ dest, uint32_t const cnt,
                                                         T const* __restrict__ s1, T const* __restrict__ s2,
                                                         T const* __restrict__ s3, T const* __restrict__ s4,
                                                         T const* __restrict__ s5, T const* __restrict__ s6 )
    {
#pragma omp parallel for schedule(static,512)
        for(uint32_t i=0U; i<cnt; ++i){
            dest[i] = T(0.142857142857142857) * ( dest[i] + s1[i] + s2[i] + s3[i] + s4[i] + s5[i] + s6[i] );
        }
    }
    //@}
    //
    /** \f$ \vec{d} = \left( \vec{d} + \sum_{i=0}^{\mathrm{ssz}-1} \vec{s}_i \right) / (1+\mathrm{ssz})\f$.
     * T is float type of \c dest[0..cnt-1], and inputs are <tt>T const* s[0..ssz-1]</tt>.
     * Each s[i] ptr-to-array-of-T <em>MAY</em> be <B>modified to point at the END of the vector,
     * i.e. at s[i]+cnt</B>. Use a temporary if you need to preserve <tt>s</tt>'s pointers. */
    template<typename T> inline void upd( T* __restrict__ dest, uint32_t const cnt,
                                          uint32_t const ssz, T const* __restrict__ * s ){
        switch(ssz){
          case(0): break;
          case(1): favg::upd_restrict1( dest, cnt, s[0] ); break;
          case(2): favg::upd_restrict2( dest, cnt, s[0], s[1] ); break;
          case(3): favg::upd_restrict3( dest, cnt, s[0], s[1], s[2] ); break;
          case(4): favg::upd_restrict4( dest, cnt, s[0], s[1], s[2], s[3] ); break;
          case(5): favg::upd_restrict5( dest, cnt, s[0], s[1], s[2], s[3], s[4] ); break;
          case(6): favg::upd_restrict6( dest, cnt, s[0], s[1], s[2], s[3], s[4], s[5] ); break;
          default: { // This simple loop MIGHT actually generate decent code
                       T const sszinv = 1.0/ (ssz+1U);
#pragma omp parallel for schedule(static,512)
                       for(uint32_t i=0U; i<cnt; ++i){            // openmp !!!
                           T sum = dest[i];       // not a more-precise version of T
                           for(uint32_t j=0U; j<ssz; ++j){
                               sum += s[j][i];
                           }
                           // Here is the single write to dest:
                           dest[i] = sum * sszinv;
                       }
                   }
        }
    }

    /** \f$ \vec{d} = \left( \vec{d} + \sum_{i=0}^{\mathrm{s.size()}-1} \vec{s}_i \right) / (1+\mathrm{s.size()})\f$.
     * T is float type of dest[0..cnt-1], and <tt>S s</tt> is an object
     * with a <tt>T [const] *data</tt> member. <B> This exits with s[i]
     * pointers pointing at the END of the data vector, s[i][cnt]</b>
     */
    template<typename T> inline void upd( T* __restrict__ dest, uint32_t const cnt,
                                          typename std::vector< T const* > const& s ){
        favg::upd( dest, cnt, s.size(), &s[0] );
    }

    /** \f$ \vec{d} = \left( \vec{d} + \sum_{i=0}^{\mathrm{s.size()}-1} s_i.\overrightarrow{\mathrm{data}} \right) / (1+\mathrm{s.size()})\f$.
     * T is float type of dest[0..cnt-1], and <tt>S s</tt> is an object
     * with a <tt>T [const] *data</tt> member. <B> This returns with s[i].data
     * pointers pointing at the END of the data vector, s[i].data[cnt]</b>
     */
    template<typename T, typename S> inline void upd( T* __restrict__ dest, uint32_t const cnt,
                                                      std::vector< S >& s ){
        switch(s.size()){
          case(0): break;
          case(1): favg::upd_restrict1( dest, cnt, s[0].data ); break;
          case(2): favg::upd_restrict2( dest, cnt, s[0].data, s[1].data ); break;
          case(3): favg::upd_restrict3( dest, cnt, s[0].data, s[1].data, s[2].data ); break;
          case(4): favg::upd_restrict4( dest, cnt, s[0].data, s[1].data, s[2].data,
                                        s[3].data );
                   break;
          case(5): favg::upd_restrict5( dest, cnt, s[0].data, s[1].data, s[2].data,
                                        s[3].data, s[4].data );
                   break;
          case(6): favg::upd_restrict6( dest, cnt, s[0].data, s[1].data, s[2].data,
                                        s[3].data, s[4].data, s[5].data );
                   break;
          default: { // This simple loop MIGHT actually generate decent code
                       T const sszinv = 1.0/ (s.size()+1U);
#pragma omp parallel for schedule(static,512)
                       for(uint32_t i=0U; i<cnt; ++i){            // openmp !!!
                           T sum = dest[i];        // not a more-precise version of T
                           for(uint32_t j=0U; j<s.size(); ++j){
                               sum += s[j].data[i];
                           }
                           // Here is the single write to dest:
                           dest[i] = sum * sszinv;
                       }
                   }
        }
    }


    /** \name <B>w</B>eighted <B>upd</B>date of dest and input vectors,
     * \f$\vec{d} = \alpha * \vec{d} + \beta * \sum_0^{N-1} \vec{s}_i\f$,
     * where \f$\alpha \in (0,1)\f$ and \f$\alpha + N \cdot \beta = 1.0\f$.
     *
     * Only \f$\alpha\f$ is supplied as a scaling parameter,
     * and typically \f$\alpha\in (0,1)\f$
     */
    namespace detail{
        /** \name weighted update specialized helpers
         * set vector \f$ \textrm{dest} = \textrm{dscal} \times \textrm{dest} + \textrm{sscal} \times \sum{ \textrm{s}_i }\f$.
         * i.e. dest is set to the average of <em>scaled</em>-dest and unscaled s1, no ptr aliasing.
         *
         * This operation is needed for BATCHED, NO-COPY averaging, where we never want to write a
         * value into dest that is completely wrong order-of-magnitude, but we have more than 6
         * \f$s_i\f$ input vectors.
         *
         * - Small example:  We want a = (a+b+c+d+e+f) to be calculated in two batches,
         * - b+c \f$sz_1=2\f$; and d+e+f \f$sz_2=3\f$
         * - 1. calculate \f$a' = = \frac{a+b+c}{3} = \frac{a+b+c}{sz_1+1}\f$ (for b+c inputs)
         * - 2. calculate our final answer which is \f$a'' = \frac{a+b+c+d+e+f}{sz_1+sz_2+1}\f$ as:
         *      \f$ a'' = \frac{ 3*a' +d+e+f }{6} = \frac{3}{6} \times a' + \frac{1}{6} \times (d+e+f)\f$
         *   - or \f$ \frac{sz_1+1}{sz_1+sz_2+1} \times a' + sscal \times (d+e+f)\f$
         *   - here dscal = 3/6 = \f$ \frac{sz_1+1}{sz_1+sz_2+1} = \frac{3}{6}\f$
         * - 3. for scale invariance, dscal and sscal MUST actually related
         *   - dscal + 3 * sscal = 1.0 = dscal + <number of s_i in batch 2> * sscal
         *   - so sscal = 1/6 = \f$ \frac{1.0 - dscal}{sz_2}\f$
         *
         * The upshot is a frontend that does \em upd_restrict for an arbitrarily long vector of s_i
         * can be written, that <B>never writes a wrongly-scaled temporary value into dest</B>
         *
         * \b NOTE: currently \c favg::upd will use a longhand loop for N>6 input vectors,
         * being carefully to accumulate (during the loop) into a temporary before updating
         * any \c dest[i] destination value.
         */
        //@{
        template<typename T> inline void wupd_restrict1( T const dscal, T* __restrict__ dest, uint32_t const cnt,
                                                         /*T const sscal,*/ T const* __restrict__ s1 ){
            T const sscal = (1.0 - dscal);
#pragma omp parallel for schedule(static,1024)
            for(uint32_t i=0U; i<cnt; ++i){
                dest[i] = dscal * dest[i] + sscal * (s1[i]);
            }
        }
        /** set vector dest to average of <em>scaled</em>-dest, unscaled s1 and s2, no ptr aliasing */
        template<typename T> inline void wupd_restrict2( T const dscal, T* __restrict__ dest, uint32_t const cnt,
                                                         T const* __restrict__ s1, T const* __restrict__ s2 ){
            T const sscal = (1.0 - dscal) * 0.5;
#pragma omp parallel for schedule(static,896)
            for(uint32_t i=0U; i<cnt; ++i){
                dest[i] = dscal * dest[i] + sscal * ( s1[i] + s2[i] );
            }
        }
        template<typename T> inline void wupd_restrict3( T const dscal, T* __restrict__ dest, uint32_t const cnt,
                                                         T const* __restrict__ s1, T const* __restrict__ s2,
                                                         T const* __restrict__ s3 ){
            T const sscal = (1.0 - dscal) * 0.333333333333333333;
#pragma omp parallel for schedule(static,768)
            for(uint32_t i=0U; i<cnt; ++i){
                dest[i] = dscal * dest[i] + sscal * ( s1[i] + s2[i] + s3[i] );
            }
        }
        template<typename T> inline void wupd_restrict4( T const dscal, T* __restrict__ dest, uint32_t const cnt,
                                                         T const* __restrict__ s1, T const* __restrict__ s2, 
                                                         T const* __restrict__ s3, T const* __restrict__ s4 ){
            T const sscal = (1.0 - dscal) * 0.25;
#pragma omp parallel for schedule(static,640)
            for(uint32_t i=0U; i<cnt; ++i){
                dest[i] = dscal * dest[i] + sscal * ( s1[i] + s2[i] + s3[i] + s4[i] );
            }
        }
        template<typename T> inline void wupd_restrict5( T const dscal, T* __restrict__ dest, uint32_t const cnt,
                                                         T const* __restrict__ s1, T const* __restrict__ s2,
                                                         T const* __restrict__ s3, T const* __restrict__ s4,
                                                         T const* __restrict__ s5 ){
            T const sscal = (1.0 - dscal) * 0.2;
#pragma omp parallel for schedule(static,512)
            for(uint32_t i=0U; i<cnt; ++i){
                dest[i] = dscal * dest[i] + sscal * ( s1[i] + s2[i] + s3[i] + s4[i] + s5[i] );
            }
        }
        template<typename T> inline void wupd_restrict6( T const dscal, T* __restrict__ dest, uint32_t const cnt,
                                                         T const* __restrict__ s1, T const* __restrict__ s2,
                                                         T const* __restrict__ s3, T const* __restrict__ s4,
                                                         T const* __restrict__ s5, T const* __restrict__ s6 ){
            T const sscal = (1.0 - dscal) * 0.166666666666666667;
#pragma omp parallel for schedule(static,512)
            for(uint32_t i=0U; i<cnt; ++i){
                dest[i] = dscal * dest[i] + sscal * ( s1[i] + s2[i] + s3[i] + s4[i] + s5[i] + s6[i] );
            }
        }
        //@}
    }//detail::
}//favg::




/** \c fsum:: function are \em dangerous if \c dest is an RDMA push buffer,
 * because you may be calculating something of <em>incorrect scale</em>.
 * They are fine for other purposes, or reduction into a (temporary) iBuf
 * of a \c Dstorm segment. */
namespace fsum {
    /** \name set dest to average of input vectors.
     * \f$\vec{d} = \sum_{i=0}^{\mathrm{cnt}-1} \vec{s}_i / \mathrm{cnt}\f$ */
    //@{
    /** set vector of dest to sum of s1 and s2, no ptr aliasing */
    template<typename T> inline void set_restrict2( T* __restrict__ dest, uint32_t const cnt,
                                                         T const* __restrict__ s1, T const* __restrict__ s2 ){
#pragma omp parallel for schedule(static,1024)
        for(uint32_t i=0U; i<cnt; ++i){
            dest[i] = ( s1[i] + s2[i] );
        }
    }
    template<typename T> inline void set_restrict3( T* __restrict__ dest, uint32_t const cnt,
                                                         T const* __restrict__ s1, T const* __restrict__ s2,
                                                         T const* __restrict__ s3 ){
#pragma omp parallel for schedule(static,896)
        for(uint32_t i=0U; i<cnt; ++i){
            dest[i] = ( s1[i] + s2[i] + s3[i] );
        }
    }
    template<typename T> inline void set_restrict4( T* __restrict__ dest, uint32_t const cnt,
                                                         T const* __restrict__ s1, T const* __restrict__ s2, 
                                                         T const* __restrict__ s3, T const* __restrict__ s4 ){
#pragma omp parallel for schedule(static,768)
        for(uint32_t i=0U; i<cnt; ++i){
            dest[i] = ( s1[i] + s2[i] + s3[i] + s4[i] );
        }
    }
    template<typename T> inline void set_restrict5( T* __restrict__ dest, uint32_t const cnt,
                                                         T const* __restrict__ s1, T const* __restrict__ s2,
                                                         T const* __restrict__ s3, T const* __restrict__ s4,
                                                         T const* __restrict__ s5 ){
#pragma omp parallel for schedule(static,640)
        for(uint32_t i=0U; i<cnt; ++i){
            dest[i] = ( s1[i] + s2[i] + s3[i] + s4[i] + s5[i] );
        }
    }
    template<typename T> inline void set_restrict6( T* __restrict__ dest, uint32_t const cnt,
                                                         T const* __restrict__ s1, T const* __restrict__ s2,
                                                         T const* __restrict__ s3, T const* __restrict__ s4,
                                                         T const* __restrict__ s5, T const* __restrict__ s6 ){
#pragma omp parallel for schedule(static,512)
        for(uint32_t i=0U; i<cnt; ++i){
            dest[i] = ( s1[i] + s2[i] + s3[i] + s4[i] + s5[i] + s6[i] );
        }
    }
    //@}
    /** \f$\vec{d} = \sum_{i=0}^{\mathrm{ssz}-1} \vec{s}_i \f$.
     *  T is float type of \c dest[0..cnt-1], and inputs are <tt>T const* s[0..ssz-1]</tt>.
     * Each s[i] ptr-to-array-of-T <em>MAY</em> be <B>modified to point at the END of the vector,
     * i.e. at s[i]+cnt</B>. Use a temporary if you need to preserve <tt>s</tt>'s pointers. */
    template<typename T> inline void set( T* __restrict__ dest, uint32_t const cnt,
                                          uint32_t const ssz, T const* __restrict__ * s ){
        switch(ssz){
          case(0): break;
          case(1): memcpy( dest, *s, cnt*sizeof(T) ); break;
          case(2): fsum::set_restrict2( dest, cnt, s[0], s[1] ); break;
          case(3): fsum::set_restrict3( dest, cnt, s[0], s[1], s[2] ); break;
          case(4): fsum::set_restrict4( dest, cnt, s[0], s[1], s[2], s[3] ); break;
          case(5): fsum::set_restrict5( dest, cnt, s[0], s[1], s[2], s[3], s[4] ); break;
          case(6): fsum::set_restrict6( dest, cnt, s[0], s[1], s[2], s[3], s[4], s[5] ); break;
          default: { // This simple loop MIGHT actually generate decent code
#pragma omp parallel for schedule(static,512)
                       for(uint32_t i=0U; i<cnt; ++i){            // openmp !!!
                           T sum = T(0);        // not a more-precise version of T
                           for(uint32_t j=0U; j<ssz; ++j){
                               sum += s[j][i];
                           }
                           // Here is the single write to dest:
                           dest[i] = sum;
                       }
                   }
        }
    }

    /** \f$\vec{d} = \sum_{i=0}^{\mathrm{s.size()}-1} \vec{s}_i\f$.
     * T is float type of dest[0..cnt-1], and <tt>S s</tt> is an object
     * with a <tt>T [const] *data</tt> member. <B> This exits with s[i].data
     * pointers pointing at the END of the data vector, s[i].data[cnt]</b>
     */
    template<typename T> inline void set( T* __restrict__ dest, uint32_t const cnt,
                                          typename std::vector< T const* > const& s ){
        fsum::set( dest, cnt, s.size(), &s[0] );
    }


    /** \f$\vec{d} = \sum_{i=0}^{\mathrm{s.size()}-1} s_i.\overrightarrow{\mathrm{data}}\f$.
     * T is float type of dest[0..cnt-1], and <tt>S s</tt> is an object
     * with a <tt>T [const] *data</tt> member. <B> This returns with s[i].data
     * pointers pointing at the END of the data vector, s[i].data[cnt]</b>
     */
    template<typename T, typename S> inline void set( T* __restrict__ dest, uint32_t const cnt,
                                                      std::vector< S >& s ){
        switch(s.size()){
          case(0): break;
          case(1): memcpy( dest, s[0].data, cnt*sizeof(T) ); break;
          case(2): fsum::set_restrict2( dest, cnt, s[0].data, s[1].data ); break;
          case(3): fsum::set_restrict3( dest, cnt, s[0].data, s[1].data, s[2].data ); break;
          case(4): fsum::set_restrict4( dest, cnt, s[0].data, s[1].data, s[2].data,
                                        s[3].data );
                   break;
          case(5): fsum::set_restrict5( dest, cnt, s[0].data, s[1].data, s[2].data,
                                        s[3].data, s[4].data );
                   break;
          case(6): fsum::set_restrict6( dest, cnt, s[0].data, s[1].data, s[2].data,
                                        s[3].data, s[4].data, s[5].data );
                   break;
          default: { // This simple loop MIGHT actually generate decent code
#pragma omp parallel for schedule(static,512)
                       for(uint32_t i=0U; i<cnt; ++i){            // openmp !!!
                           T sum = T(0);        // not a more-precise version of T
                           for(uint32_t j=0U; j<s.size(); ++j){
                               sum += s[j].data[i];
                           }
                           // Here is the single write to dest:
                           dest[i] = sum;
                       }
                   }
        }
    }
    /** \name UPDate vector dest to sum of dest and input vectors,
     * \f$ \vec{d} = \left( \vec{d} + \sum_{i=0}^{\mathrm{cnt}-1} \vec{s}_i \right) \f$ */
    //@{
    /** set vector dest to sum of dest and s1, no ptr aliasing */
    template<typename T> inline void upd_restrict1( T* __restrict__ dest, uint32_t const cnt,
                                                         T const* __restrict__ s1 ){
        for(uint32_t i=0U; i<cnt; ++i){
            dest[i] = ( dest[i] + s1[i] );
        }
    }
    /** set vector dest to sum of dest, s1 and s2, no ptr aliasing */
    template<typename T> inline void upd_restrict2( T* __restrict__ dest, uint32_t const cnt,
                                                         T const* __restrict__ s1, T const* __restrict__ s2 ){
        for(uint32_t i=0U; i<cnt; ++i){
            dest[i] = ( dest[i] + s1[i] + s2[i] );
        }
    }
    template<typename T> inline void upd_restrict3( T* __restrict__ dest, uint32_t const cnt,
                                                         T const* __restrict__ s1, T const* __restrict__ s2,
                                                         T const* __restrict__ s3 ){
        for(uint32_t i=0U; i<cnt; ++i){
            dest[i] = ( dest[i] + s1[i] + s2[i] + s3[i] );
        }
    }
    template<typename T> inline void upd_restrict4( T* __restrict__ dest, uint32_t const cnt,
                                                         T const* __restrict__ s1, T const* __restrict__ s2, 
                                                         T const* __restrict__ s3, T const* __restrict__ s4 ){
        for(uint32_t i=0U; i<cnt; ++i){
            dest[i] = ( dest[i] + s1[i] + s2[i] + s3[i] + s4[i] );
        }
    }
    template<typename T> inline void upd_restrict5( T* __restrict__ dest, uint32_t const cnt,
                                                         T const* __restrict__ s1, T const* __restrict__ s2,
                                                         T const* __restrict__ s3, T const* __restrict__ s4,
                                                         T const* __restrict__ s5 ){
        for(uint32_t i=0U; i<cnt; ++i){
            dest[i] = ( dest[i] + s1[i] + s2[i] + s3[i] + s4[i] + s5[i] );
        }
    }
    template<typename T> inline void upd_restrict6( T* __restrict__ dest, uint32_t const cnt,
                                                         T const* __restrict__ s1, T const* __restrict__ s2,
                                                         T const* __restrict__ s3, T const* __restrict__ s4,
                                                         T const* __restrict__ s5, T const* __restrict__ s6 ){
        for(uint32_t i=0U; i<cnt; ++i){
            dest[i] = ( dest[i] + s1[i] + s2[i] + s3[i] + s4[i] + s5[i] + s6[i] );
        }
    }
    //@}
}//fsum::
#endif // FLOAT_OPS_HPP

