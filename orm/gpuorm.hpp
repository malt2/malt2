/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef GPUORM_HPP
#define GPUORM_HPP
/** @file
 * opaque 'C++' info for ORM_GPU transport currently identical to MormConf
 */

#include "orm.h"        // struct Orm dispatch table
#if WITH_GPU==0
#error "gpuorm.hpp, but not WITH_GPU?"
#endif

#include "mpiorm.hpp"



#endif //MPI_ORM_H
