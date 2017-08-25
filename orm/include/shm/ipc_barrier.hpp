/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef IPC_BARRIER_HPP
#define IPC_BARRIER_HPP
/** \file ipc_barrier.hpp
 * simplified from boost/thread/barrier.hpp and switched to use
 * boost::interprocess synchronization types.
 *
 * Issue: locating a barrier in shm is \b not good.  This (and boost)
 * impl has callback functions. This can result in pointers being stored
 * in shm that point to potential process specific addresses! Valgrind
 * detected this.  \c /usr/include/c++/4.8/functional does have some sort
 * of check for whether the function ptr is location-invariant, but this
 * is evidently not the case with \e ipc_barrier.hpp.
 *
 *  - The valgrind complaint seems valid, so:
 *    - I will revert to a pthread barrier for now,
 *    - and restrict \e orm_shm to be \e non-IPC (pure multithreaded).
 */

#include <boost/interprocess/sync/interprocess_mutex.hpp>
// NOT IMPLEMENTED #include <boost/interprocess/interprocess_barrier.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>

//#include <utility>              // std::move
#include <boost/move/move.hpp>          // boost::move

//#include <type_traits>          // std::enable_if
#include <boost/utility/enable_if.hpp>
#include <boost/utility/result_of.hpp>
#include <boost/type_traits/is_void.hpp>

#include <functional>           // std::function
#include <mutex>                // std::unique_lock decorator

namespace shorm { namespace ipc {
    //using std::enable_if;
    using boost::enable_if;
    using boost::result_of;;
    using boost::is_void;;
    using boost::interprocess::interprocess_mutex;
    using boost::interprocess::interprocess_condition;

    // --- unique_lock port ---
#if 0 // --- i will try std::unique_lock ----
    template <typename Mutex>
        class unique_lock
        {
        private:
            Mutex* m;
            bool is_locked;

        private:
            explicit unique_lock(upgrade_lock<Mutex>&);
            unique_lock& operator=(upgrade_lock<Mutex>& other);
        public:
            typedef Mutex mutex_type;
            unique_lock(unique_lock const&) = delete; unique_lock& operator=(unique_lock const&) = delete;
            unique_lock()noexcept :
                m(0),is_locked(false)
                {}

            explicit unique_lock(Mutex& m_) :
                m(&m_), is_locked(false)
            {
                lock();
            }
            unique_lock(Mutex& m_, adopt_lock_t) :
                m(&m_), is_locked(true)
            {}
            unique_lock(Mutex& m_, defer_lock_t)noexcept:
                m(&m_),is_locked(false)
                {}
            unique_lock(Mutex& m_, try_to_lock_t) :
                m(&m_), is_locked(false)
            {
                try_lock();
            }
            unique_lock(unique_lock && other) noexcept:
                m(other.m),is_locked(other.is_locked)
                {
                    other.is_locked=false;
                    other.m=0;
                }

            unique_lock& operator=(unique_lock && other)
            {
                unique_lock temp(::boost::move(other));
                swap(temp);
                return *this;
            }
            void swap(unique_lock& other)noexcept
            {
                std::swap(m,other.m);
                std::swap(is_locked,other.is_locked);
            }

            ~unique_lock()
            {
                if (owns_lock())
                {
                    m->unlock();
                }
            }
            void lock()
            {
                if (m == 0)
                {
                    boost::throw_exception(
                                           boost::lock_error(system::errc::operation_not_permitted, "boost unique_lock has no mutex"));
                }
                if (owns_lock())
                {
                    boost::throw_exception(
                                           boost::lock_error(system::errc::resource_deadlock_would_occur, "boost unique_lock owns already the mutex"));
                }
                m->lock();
                is_locked = true;
            }
            bool try_lock()
            {
                if (m == 0)
                {
                    boost::throw_exception(
                                           boost::lock_error(system::errc::operation_not_permitted, "boost unique_lock has no mutex"));
                }
                if (owns_lock())
                {
                    boost::throw_exception(
                                           boost::lock_error(system::errc::resource_deadlock_would_occur, "boost unique_lock owns already the mutex"));
                }
                is_locked = m->try_lock();
                return is_locked;
            }
        };
#endif

    // ---- barrier.hpp  port ----

    namespace detail
    {

        typedef std::function<void()> void_completion_function;
        typedef std::function<size_t()> size_completion_function;

        struct default_barrier_reseter
        {
            unsigned int size_;
            default_barrier_reseter(unsigned int size) :
                size_(size)
            {
            }
            unsigned int operator()()
            {
                return size_;
            }
        };

        struct void_functor_barrier_reseter
        {
            unsigned int size_;
            void_completion_function fct_;
            template <typename F>

                void_functor_barrier_reseter(unsigned int size, F && funct)
                : size_(size), fct_(boost::move(funct))
                {}
            unsigned int operator()()
            {
                fct_();
                return size_;
            }
        };
        struct void_fct_ptr_barrier_reseter
        {
            unsigned int size_;
            void(*fct_)();
            void_fct_ptr_barrier_reseter(unsigned int size, void(*funct)()) :
                size_(size), fct_(funct)
            {
            }
            unsigned int operator()()
            {
                fct_();
                return size_;
            }
        };
    }//detail::

    class barrier
    {
        static inline unsigned int check_counter(unsigned int count)
        {
            //if (count == 0) boost::throw_exception
            //    (thread_exception(system::errc::invalid_argument, "barrier constructor: count cannot be zero."));
            if (count == 0) throw std::runtime_error("barrier counter cannot be zero");
            return count;
        }
        struct dummy
        {
        };

    public:
        barrier(barrier const&) = delete; barrier& operator=(barrier const&) = delete;

        explicit barrier(unsigned int count) :
            m_count(check_counter(count)), m_generation(0), fct_(detail::default_barrier_reseter(count))
        {
        }

#if 0
        template <typename F>
            barrier( unsigned int count, F && funct,
                     typename enable_if<
                     typename is_void<typename result_of<F>::type>::type, dummy*
                     >::type=0
                   )
            : m_count(check_counter(count)),
            m_generation(0),
            fct_(detail::void_functor_barrier_reseter
                 (count, boost::move(funct) ))
        {}

        template <typename F>
            barrier( unsigned int count, F && funct,
                     typename boost::enable_if<
                     typename is_same<typename result_of<F>::type, unsigned int>::type, dummy*
                     >::type=0
                   )
            : m_count(check_counter(count)),
            m_generation(0),
            fct_( boost::move(funct) )
        {}
#endif

        barrier(unsigned int count, void(*funct)()) :
            m_count(check_counter(count)), m_generation(0),
            fct_(funct
                 ? detail::size_completion_function(detail::void_fct_ptr_barrier_reseter(count, funct))
                 : detail::size_completion_function(detail::default_barrier_reseter(count))
                )
            {
            }
        barrier(unsigned int count, unsigned int(*funct)()) :
            m_count(check_counter(count)), m_generation(0),
            fct_(funct
                 ? detail::size_completion_function(funct)
                 : detail::size_completion_function(detail::default_barrier_reseter(count))
                )
            {
            }

        bool wait()
        {
            // std::unique_lock adds a bunch of nice semantics to any Lockable
            std::unique_lock < interprocess_mutex > lock(m_mutex);
            unsigned int gen = m_generation;

            if (--m_count == 0)
            {
                m_generation++;
                m_count = static_cast<unsigned int>(fct_());
                assert( m_count != 0 );
                //((m_count != 0) ? static_cast<void> (0) : __assert_fail ("m_count != 0", "/opt/rh/devtoolset-2/root/usr/include/boost/thread/barrier.hpp", 187, __PRETTY_FUNCTION__));
                m_cond.notify_all();
                return true;
            }

            while (gen == m_generation)
                m_cond.wait(lock);
            return false;
        }

        void count_down_and_wait()
        {
            wait();
        }

    private:
        interprocess_mutex m_mutex;
        interprocess_condition m_cond;
        unsigned int m_count;
        unsigned int m_generation;
        detail::size_completion_function fct_;
        // if barrier in shm, then doesn't fct_ point to a thread-local.  This would be wrong.
        // valgrind complaint:
        // ==29161==    at 0x4C2B0E0: operator new(unsigned long) (in /usr/lib/valgrind/vgpreload_memcheck-amd64-linux.so)
        // ==29161==    by 0x405C51: std::_Function_base::_Base_manager<shorm::ipc::detail::default_barrier_reseter>::_M_init_functor(std::_Any_data&, shorm::ipc::detail::default_barrier_reseter&&, std::integral_constant<bool, false>) (functional:1987)
        // ==29161==    by 0x405B5D: std::_Function_base::_Base_manager<shorm::ipc::detail::default_barrier_reseter>::_M_init_functor(std::_Any_data&, shorm::ipc::detail::default_barrier_reseter&&) (functional:1958)
        // ==29161==    by 0x405A6F: std::function<unsigned long ()>::function<shorm::ipc::detail::default_barrier_reseter, void>(shorm::ipc::detail::default_barrier_reseter) (functional:2451)
        // ==29161==    by 0x4057E7: shorm::ipc::barrier::barrier(unsigned int) (ipc_barrier.hpp:194)
        // ==29161==    by 0x40450B: shorm_open (shormOps.cpp:90)
        //
    };
}}//shorm::ipc::
#endif // IPC_BARRIER_HPP
