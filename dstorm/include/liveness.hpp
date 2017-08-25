/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef LIVENESS_HPP
#define LIVENESS_HPP

#include <cstdint>
#include <vector>

namespace dStorm {

    // /** track live/dead nodes, provide \c barrier, \c recover calls,
    //  * "if possible" (and may need special compilation) */
    // class DstormLiveness; // see dstorm.hh (or dstorm.cuh)

    /** TBD -- LiveAlways is a LiveBase impl where no node is reported dead.
     *
     * It might be useful as a default parameter.
     * IB, UDP, ... are expected to have different
     * LiveBase implementations. */
    class LiveAlways;

    // ------- declarations -------------

    namespace detail {

        class LiveBase
        {
        public:
            typedef uint_least16_t Tnode;
            virtual std::vector<Tnode> const& getDead() const = 0;
            virtual void update() = 0;                  // update avoid_list
            virtual bool live(Tnode const r) const = 0; // query current avoid_list
            bool dead( Tnode const r ) const {
                return ! live(r);
            }
        protected:
            LiveBase();
            virtual ~LiveBase() = 0;
        };

    }//detail::

    namespace user {
        class LiveAlways : public dStorm::detail::LiveBase
        {
        public:
            LiveAlways() : LiveBase(), dead() {}
            std::vector<Tnode> const& getDead() const {
                return this->dead;
            }
            bool live( Tnode const /*r*/ ) const {return true;}
            /** LiveAlways \b never declares a node dead (no-op) */
            void update() {}
        private:
            std::vector<Tnode> dead;    ///< always empty
        };
    }//user::

}//dStorm::
#endif // LIVENESS_HPP
