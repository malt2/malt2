/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */

#include "include/segVecDense.hh"

namespace dStorm {

    namespace user {
        // Explicit instantiation -- help detect "incomlete type" errors early
        //     AND can force some funcs into the library.
        //     template classes into the library ?
        template class Seg_VecDense<float>;
        template class Seg_VecDense<double>;
    }

    // We must also instantiate the corresponding SegBase classes
    // into the library...
    template class detail::SegBase< user::Seg_VecDense<float>, float >;
    template class detail::SegBase< user::Seg_VecDense<double>, double >;

    //template class detail::SegImpl< seg::VecDense<float> >;
    //template class detail::SegImpl< seg::VecDense<double> >;

}//dStorm::
