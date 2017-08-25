/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */

#include "dstorm_fwd.hpp"

#include "ionet/userIoNet.hpp"
#include "ionet/scalIoNet.hpp"
#include "ionet/globIoNet.hpp"

namespace dStorm {

    /** OldIoNetEnum --> impl type at compile time*/
    template< OldIoNetEnum BUILTIN > struct UsrImpl {
        typedef mm2::user::IoNetEmpty type;
    };
    template<> struct UsrImpl<ALL> { typedef mm2::user::IoNetAll type; };
    template<> struct UsrImpl<SELF> { typedef mm2::user::IoNetSelf type; };
    template<> struct UsrImpl<CHORD> { typedef mm2::user::IoNetChord type; };
    template<> struct UsrImpl<HALTON> { typedef mm2::user::IoNetHalton type; };
    template<> struct UsrImpl<RANDOM> { typedef mm2::user::IoNetRandom type; };
    template<> struct UsrImpl<STREAM> { typedef mm2::user::IoNetStream type; };
    template<> struct UsrImpl<PARA_SERVER> { typedef mm2::user::IoNetParaServer type; };

    namespace detail {

        /** OldIoNetEnum --> to GlobNet at runtime (for test/debug) */
        mm2::GlobNet* mkGlobNet( OldIoNetEnum const tag, mm2::Tnode const sz );

        /** OldIoNetEnum --> to ScalNet at runtime (for test/debug) */
        mm2::ScalNet* mkScalNet( OldIoNetEnum const tag, mm2::Tnode const rank, mm2::Tnode const sz );

    }//detail::

}//dStorm::

