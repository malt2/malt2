/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef IONET_FWD_HPP
#define IONET_FWD_HPP

#include <cstdint>
#include <type_traits>

namespace mm2
{
    // userIoNet.hpp fwd decl ...

    /** Support up to 64k processing "nodes" */
    typedef std::uint_least16_t Tnode;
    static_assert( std::is_unsigned<Tnode>(), "Tnode must be unsigned" );

    /** base class for user-defined graphs */
    class UserIoNet;

    /** Example implementations of mm2::UserIoNet */
    namespace user{
        class IoNetEmpty;
        class IoNetSelf;
        class IoNetAll;
        class IoNetChord;
        class IoNetHalton;
        class IoNetRandom;
        class IoNetStream;
        class IoNetParaServer;
    }

    class GlobNet;      // see globIoNet.hpp
    class ScalNet;      // see scalIoNet.hpp

}//mm2::
#endif // IONET_FWD_HPP
