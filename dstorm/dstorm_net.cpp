/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
/** \file
 * a few ionet addons related to dStorm::OldIoNetEnum tag values.
 */
#include "include/dstorm_net.hpp"

using std::unique_ptr;
using namespace std;
using namespace dStorm;

namespace dStorm {

    namespace detail {
        // for testing
        mm2::GlobNet* mkGlobNet( OldIoNetEnum const tag, mm2::Tnode const sz ){
            switch(tag){
              case(ALL):         return new mm2::GlobNet(unique_ptr<mm2::UserIoNet>(new UsrImpl<ALL        >::type(sz)));
              case(CHORD):       return new mm2::GlobNet(unique_ptr<mm2::UserIoNet>(new UsrImpl<CHORD      >::type(sz)));
              case(HALTON):      return new mm2::GlobNet(unique_ptr<mm2::UserIoNet>(new UsrImpl<HALTON     >::type(sz)));
              case(RANDOM):      return new mm2::GlobNet(unique_ptr<mm2::UserIoNet>(new UsrImpl<RANDOM     >::type(sz)));
              case(PARA_SERVER): return new mm2::GlobNet(unique_ptr<mm2::UserIoNet>(new UsrImpl<PARA_SERVER>::type(sz)));
              case(STREAM):      return new mm2::GlobNet(unique_ptr<mm2::UserIoNet>(new UsrImpl<STREAM     >::type(sz)));
              default:           return new mm2::GlobNet(unique_ptr<mm2::UserIoNet>(new mm2::user::IoNetEmpty(sz)));
            }
        }
        // runtime (not fast)
        mm2::ScalNet* mkScalNet( OldIoNetEnum const tag, mm2::Tnode const rank, mm2::Tnode const sz ){
            switch(tag){
              case(ALL):         return new mm2::ScalNet(rank,unique_ptr<mm2::UserIoNet>(new UsrImpl<ALL        >::type(sz)));
              case(CHORD):       return new mm2::ScalNet(rank,unique_ptr<mm2::UserIoNet>(new UsrImpl<CHORD      >::type(sz)));
              case(HALTON):      return new mm2::ScalNet(rank,unique_ptr<mm2::UserIoNet>(new UsrImpl<HALTON     >::type(sz)));
              case(RANDOM):      return new mm2::ScalNet(rank,unique_ptr<mm2::UserIoNet>(new UsrImpl<RANDOM     >::type(sz)));
              case(PARA_SERVER): return new mm2::ScalNet(rank,unique_ptr<mm2::UserIoNet>(new UsrImpl<PARA_SERVER>::type(sz)));
              case(STREAM):      return new mm2::ScalNet(rank,unique_ptr<mm2::UserIoNet>(new UsrImpl<STREAM     >::type(sz)));
              default:           return new mm2::ScalNet(rank,unique_ptr<mm2::UserIoNet>(new mm2::user::IoNetEmpty(sz)));
            }
        }

    }//detail::
}//dStorm::
