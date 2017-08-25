/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef DEMANGLE_HPP_
#define DEMANGLE_HPP_
#include <string>
#include <typeinfo>
std::string demangle( char const* name );

template<class T>
std::string type_str() {
    return demangle( typeid(T).name() );
}
#ifdef __GNUG__

#include <cstdlib>
#include <memory>
#include <cxxabi.h>
#if 0 && __cplusplus >= 199711L      // c++11 ?
inline std::string demangle( char const* name ) {
    int status = -4;            // -4 to squash warning
    std::unique_ptr<char, void(*)(void*)> res {
        abi::__cxa_demangle(name,NULL,NULL,&status);
        std::free
    };
    return (status==0)? res.get() : name ;
}
#else // C++98...
struct demangle_handle {
    char *p;
    demangle_handle(char* ptr) : p(ptr) {}
    ~demangle_handle() { std::free(p); }
};
inline std::string demangle( char const* name ) {
    int status = -4;
    demangle_handle result( abi::__cxa_demangle(name,NULL,NULL,&status ));
    return (status==0)? result.p : name;
}
#endif

#else
#error "do not know how to demangle (not __GNUG__)"
std::string demangle( char const* name) { // NOP
    return name;
}
#endif

#endif // DEMANGLE_HPP_
