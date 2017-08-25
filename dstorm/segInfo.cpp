/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */

#include "segInfo.hpp"

namespace dStorm {

    // constexpr class members are declared to the compiler.  To get them
    // into libraries, we also need to define them.
    // Header declares them with a value.  Here we give them a storage location.
    constexpr uint32_t SegInfo::oBufs[SEG_LAYOUTS]; // = { 0U, 0U };
    constexpr uint32_t SegInfo::iBufs[SEG_LAYOUTS]; // = { 1U, 0U };
    constexpr uint32_t SegInfo::rBufs[SEG_LAYOUTS]; // = { 2U, 0U };

}//dStorm::
