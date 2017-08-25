/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */

#include "dstorm_any.hh"        // segInfo + Dstorm::getSegInfo
#include "demangle.hpp"
#include <sstream>
#include <iomanip>

namespace dStorm {

    std::string Dstorm::name( IoNet_t const &sync_model ) const
    {
        return iographs[sync_model].name();
    }

    void Dstorm::print_seg_info( SegNum const s ) const
    {
        using namespace std;
        dStorm::SegInfo const& info = getSegInfo(s);
        ostringstream oss;

        oss <<"\n Dstorm segment type "<<demangle(typeid(info).name());
        oss<<"  "<<setw(15)<<"seg::FMT::value"<<setw(8)<<unsigned(info.fmtValue);
        switch( info.fmtValue ){
          case( seg::VecDense<float>::value ): oss<<" seg::"<<seg::VecDense<float>::name; break;
          case( seg::VecGpu<float>::value ): oss<<" seg::"<<seg::VecGpu<float>::name; break;
          default: oss<<" (see dstorm_msg.hpp)";
        }
        if( info.datacode == -1U ) oss<<" datacode unset ";
        oss<<endl;

        oss <<"\n  "<<setw(15)<<left<<"ionet"<<setw(8)<<unsigned(info.ionet)
            <<"  "<<dStorm::name(SegPolicyIO{info.policy})<<"\n"
            <<"  "<<setw(15)<<left<<"segNum:"<<setw(8)<<unsigned(info.segNum)
            <<"  "<<setw(15)<<left<<"seg_id:"<<setw(8)<<unsigned(info.seg_id)
            <<"  "<<setw(15)<<left<<"valid:"<<setw(8)<<info.valid<<"\n"
            <<"  "<<setw(15)<<left<<"obuf:"<<setw(8)<<unsigned(info.obuf)
            <<"  "<<setw(15)<<left<<"ibuf:"<<setw(8)<<unsigned(info.ibuf)
            <<"  "<<setw(15)<<left<<"rbuf:"<<setw(8)<<unsigned(info.rbuf)
            <<"  "<<setw(15)<<left<<"nbuf:"<<setw(8)<<unsigned(info.nbuf)<<"\n"
            <<"  "<<setw(15)<<left<<"datacode:"<<setw(8)<<hex<<info.datacode<<dec
            <<"  "<<setw(15)<<left<<"datasize:"<<setw(8)<<info.datasize
            <<"  "<<setw(15)<<left<<"cnt:"<<setw(8)<<info.cnt
            <<"  "<<setw(15)<<left<<"bufBytes:"<<setw(8)<<unsigned(info.bufBytes)<<" bytes\n"
            <<"  "<<setw(15)<<left<<"mem @"<<setw(8)<<(void*)info.mem
            <<"  "<<setw(15)<<left<<"sizeofMsgHeader:"<<setw(8)<<info.sizeofMsgHeader<<" bytes\n"
            <<"  "<<setw(15)<<left<<"ptrIbuf @"<<setw(8)<<(void*)info.ptrIbuf()
            <<"  "<<setw(15)<<left<<"ptrObuf @"<<setw(8)<<(void*)info.ptrObuf()
            <<"  "<<setw(15)<<left<<"segBytes:"<<setw(8)<<unsigned(info.segBytes)<<" bytes\n"
            //<<"  "<<setw(15)<<left<<"VEC_HDRSZ:"<<setw(8)<<info.vec_max_size
            //<<"  "<<setw(15)<<left<<"oBuf():"<<setw(8)<<info.oBuf()
            //<<"  "<<setw(15)<<left<<"iBuf():"<<setw(8)<<info.iBuf()
            //<<"  "<<setw(15)<<left<<"recvBeg():"<<setw(8)<<info.recvBeg()
            //<<"  "<<setw(15)<<left<<"rbufEnd():"<<setw(8)<<info.rbufEnd()
            <<"\n";
        //ORM_COUT( this->orm, oss.str().c_str() );
#if WITH_LIBORM
        this->orm->printf( this->orm, oss.str().c_str() );
#else
        cout<<oss.str().c_str(); cout.flush();
#endif
    }
}//Dstorm::
