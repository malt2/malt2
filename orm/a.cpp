/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
/** \file a.cpp
 * \brief demo of function table lookup syntax using gcc named structure member syntax
 */
#include <stdio.h>
#include <iostream>

#ifdef __cplusplus
extern "C" {
#endif

    int transport_foo();
    /** encapsulate the transport functions we use in Dstorm.
     *
     * - Alternate I/O fabrics can substitute their own versions
     *   of just these functions, and have dstorm <em>just work</em>.
     *   - Orm_shm ? (shared memory)
     *   - Orm_udp ?
     *   - Orm_mpich ?
     * - usage:
     *   - orm = Orm_transport
     */
    struct Orm {
        int (*foo)(void);
        int (*bar)(int const i);
    };//struct Orm_transport

    //Orm_shm;
    // ...

   int transport_foo() {
       printf("transport_foo 7\n");
       return 7;
   }
   int transport_bar(int const i) {
       printf(" transport_bar-->%d\n",i+1);
       return i+1;
   }
   int shm_foo() {
       printf("shm_foo\n");
       return 7;
   }
   int shm_bar(int const i) {
       printf(" shm_bar-->%d\n",i-1);
       return i-1;
   }

#ifdef __cplusplus
}//extern "C"
#endif

/*extern*/ struct Orm Orm_transport={
    .foo = &transport_foo,
        .bar = &transport_bar

};
/*extern*/ struct Orm Orm_shm={
    .foo = &shm_foo,
        .bar = &shm_bar
};

using namespace std;
int main(int,char**)
{
    struct Orm *orm = &Orm_transport;
    (void)orm->foo();
    (void)orm->bar(0);
    orm = &Orm_shm;
    (void)orm->foo();
    (void)orm->bar(0);
    cout<<"\nGoodbye"<<endl;
}
