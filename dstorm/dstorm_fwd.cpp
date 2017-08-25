/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */

#include "dstorm_fwd.hpp"
#include <sstream>
//#include "demangle.hpp"

namespace dStorm {

    static char const* oldIoNetEnumNames[IONET_MAX+1U] = {
        "ALL", "SELF", "CHORD", "DATA", "HALTON",
        "RANDOM", "PARA_SERVER", "BUTTERFLY",
        "IONET_MAX" };
    static_assert( sizeof(oldIoNetEnumNames) == (IONET_MAX+1U) * sizeof(char const*),
                   "Please fix oldIoNetEnumNames[] table to match the OldIoNetEnum's" );

    //std::string name( OldIoNetEnum const sync_model )
    std::string name( OldIoNet const sync_model )
    {
        return std::string( oldIoNetEnumNames[sync_model.ionet] );
    }

    //char const* name( orm_return_t const x )
    char const* name( OrmReturn const retcode )
    {
        char const* ret;
        orm_return_t x = retcode.orm_xxx;
        if( x >= -1 && x <= 5 ){
            static char const* orm_errors[7]={
                "ORM_ERROR", "ORM_SUCCESS", "ORM_TIMEOUT", "ORM_ERR_EMFILE"
                    , "ORM_ERR_ENV", "ORM_ERR_SN_PORT", "ORM_ERR_CONFIG" };
            ret=orm_errors[x+1];
        }else{
            ret="ORM_unknown";
        }
        return ret;
    }

    void throw_orm_error( orm_return_t const orm_return_code,
                            char const * file, unsigned line,
                            char const * msg )
    {
        std::ostringstream oss;
        oss<<" orm error code "<<orm_return_code
            <<" in "<<__FILE__<<"["<<__LINE__<<"]: "
            <<name(OrmReturn{orm_return_code})
            <<"\n\tNEED( "<<msg<<" )";
        fprintf(stderr,"%s\n", oss.str().c_str() );
        throw( orm_error(oss.str()) );
    }

    std::string name( SegPolicyIO const policy )
    {
        char const* lay;
        {
            SegPolicy const layout = policy.policy & SEG_LAYOUT_MASK;
            switch(layout){
              case(SEG_FULL): lay = "SEG_FULL"; break;
              case(SEG_ONE):  lay = "SEG_ONE"; break;
              default:        lay = "SEG_unknown";
            }
        }

        char const* red;
        {
            SegPolicy const reduce = policy.policy & REDUCE_OP_MASK;
            switch(reduce){
              case(REDUCE_AVG_RBUF):      red = "REDUCE_AVG_RBUF"; break;
              case(REDUCE_AVG_RBUF_OBUF): red = "REDUCE_AVG_RBUF_OBUF"; break;
              case(REDUCE_SUM_RBUF):      red = "REDUCE_SUM_RBUF"; break;
              case(REDUCE_STREAM):        red = "REDUCE_STREAM"; break;
              case(REDUCE_NOP):           red = "REDUCE_NOP"; break;
              default:                    red = "REDUCE_OP_unknown";
            }
        }

        char const* sub;
        {
            SegPolicy const rbuf_subview = policy.policy & RBUF_SUBVIEW_MASK;
            switch(rbuf_subview){
              case(RBUF_SUBVIEW_NONE):             sub = "RBUF_SUBVIEW_NONE"; break;
              case(RBUF_SUBVIEW_HOMOG):            sub = "RBUF_SUBVIEW_HOMOG"; break;
              case(RBUF_SUBVIEW_HOMOG_OR_NONOVLP): sub = "RBUF_SUBVIEW_HOMOG_OR_NONOVLP"; break;
              case(RBUF_SUBVIEW_OVLP_RELAXED):     sub = "RBUF_SUBVIEW_OVLP_RELAXED"; break;
              case(RBUF_SUBVIEW_ANY):              sub = "RBUF_SUBVIEW_ANY"; break;
              default:                             sub = "RBUF_SUBVIEW_unknown";
            }
        }

        char const* err;
        {
            SegPolicy const subview_err = policy.policy & SUBVIEW_ERR_MASK;
            switch(subview_err){
              case(SUBVIEW_ERR_THROW):  err = "SUBVIEW_ERR_THROW"; break;
              case(SUBVIEW_ERR_WARN):   err = "SUBVIEW_ERR_WARN"; break;
              case(SUBVIEW_ERR_IGNORE): err = "SUBVIEW_ERR_IGNORE"; break;
              default:                  err = "SUBVIEW_ERR_unknown";
            }
        }

        char const* sync;
        {
            SegPolicy const subview_err = policy.policy & SEGSYNC_MASK;
            switch(subview_err){
              case(SEGSYNC_NONE):       sync = "SEGSYNC_NONE"; break;
              case(SEGSYNC_NOTIFY):     sync = "SEGSYNC_NOTIFY"; break;
              case(SEGSYNC_NOTIFY_ACK): sync = "SEGSYNC_NOTIFY_ACK"; break;
              default:                  sync = "SYGSYNC_unknown";
            }
        }
        std::ostringstream oss;
        oss << lay << "|" << red << "|" << sub << "|" << err << "|" << sync;
        return oss.str();
    }

}//Dstorm::
