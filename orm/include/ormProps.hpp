/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef ORMPROPS_HPP_
#define ORMPROPS_HPP_
#include <cstdint>
namespace orm {
    // ---- communication properties ---

    /** buffer layout and usage depends on many things.
     * - Hogwild ~ COMM_ASYNWRITE, REDUCE_NONE, BUF_TWO, IOTHREAD_NONE
     * - Multicopy ~ COMM_ASYNCWRITE, REDUCE_SEG, BUF_MULTI, IOTHREAD_NONE
     *
     * - Shm currently implementing only one setting,
     *   - COMM_READ, REDUCE_SEG, BUF_MULTI, IOTHREAD_NONE
     */
    struct Enums
    {
        typedef enum Communication_e {
            COMM_ASYNCWRITE,    ///< like IB, write full data to client rBuf, with hdr and trailer sections
            COMM_READ,          ///< like shared memory (SHM), client rBuf is spsc queue with remote read "addr"
            COMM_WRITENOTIFY    ///< TBD, perhaps easier to implement
                // error check (ex. checksumming) ?
        } Comm;

        /// segments have send buffer, receive buffer[s], and maybe a reduce buffer.
        typedef enum Reduce_e {
            REDUCE_NONE,        ///< no reduce buffer (Hogwild?)
            REDUCE_SEG,         ///< reduce desitnation buffer within segment (average new rBufs)
            REDUCE_SEGADD,      ///< reduce desitnation buffer within segment (sum new rBufs)
            REDUCE_USER,        ///< reduce to user buffer
            REDUCE_USERADD,     ///< reduce to user buffer, add
            REDUCE_USERFUNC     ///< supply raw data ptrs to user function
        } Reduce;

        typedef enum BufLayout_e {
            BUF_ONE,            ///< (useful?) single buf for send and recv (only COMM_ASYNCWRITE?)
            BUF_TWO,            ///< (Hogwild?) one send, one recv buf (overwrite) (only COMM_ASYNCWRITE?)
            /// \name buffer multiplicity
            /// For COMM_ASYNCWRITE, how many rbufs per receiver (point-to-point), but for
            /// COMM_READ, how many sendbufs (and SPSC queue size at receiver is same size)
            //@{
            BUF_MULTI,          ///< (old "multicopy" default) one send, 1 rbuf per recvList, reduce as per Reduce
            BUF_MULTI2,         ///< one send, 2 rbuf per recvList, reduce as per Reduce
            BUF_MULTI4,         ///< one send, 4 rbuf per recvList, reduce as per Reduce
            BUF_MULTI8,         ///< one send, 8 rbuf per recvList, reduce as per Reduce
            BUF_MULTI16,        ///< one send, 16 rbuf per recvList, reduce as per Reduce
            //@}
        } BufLayout;

        typedef enum IOthread_e {
            IOTHREAD_NONE,      ///< use an iothread
            IOTHREAD_REDUCE     ///< TBD, reduce continuously in background thread, calc thread gets "immediate" reduce result (but perhaps without latest and greatest data)
        } IOthread;

    };

    /** Not sure whether these can remain compile-time const. */
    struct Config {
        static uint_least16_t constexpr mask_comm=7;
        static uint_least16_t constexpr shift_reduce=3, mask_reduce=7;
        static uint_least16_t constexpr shift_buflay=6, mask_buflay=7;
        static uint_least16_t constexpr shift_iothread=9, mask_iothread=2;
        uint_least16_t const flags;
        explicit Config( uint_least16_t flags ) : flags(flags) {}
        Config( Enums::Comm        comm, Enums::Reduce   reduce,
                  Enums::BufLayout buflay, Enums::IOthread iothread
                )
            : flags( comm + (reduce<<shift_reduce) + (buflay<<shift_buflay) + (iothread<<shift_iothread) )
        {}
        Enums::Comm      const comm()     const { return static_cast<Enums::Comm     >
            ((flags                )&mask_comm); }
        Enums::Reduce    const reduce()   const { return static_cast<Enums::Reduce   >
            ((flags>>shift_reduce  )&mask_reduce); }
        Enums::BufLayout const buflay()   const { return static_cast<Enums::BufLayout>
            ((flags>>shift_buflay  )&mask_buflay); }
        Enums::IOthread  const iothread() const { return static_cast<Enums::IOthread >
            ((flags>>shift_iothread)&mask_iothread); }
    };

    struct Default {
        // can only define const literals in header.
        static uint_least16_t constexpr ibMulti = Enums::COMM_ASYNCWRITE | (Enums::REDUCE_SEG << 3)
            | (Enums::BUF_MULTI << 6) | (Enums::IOTHREAD_NONE << 9);
        static_assert( ((ibMulti)&Config::mask_comm) == Enums::COMM_ASYNCWRITE, "fix bit-shift/mask issues!");
        static_assert( ((ibMulti>>Config::shift_reduce)&Config::mask_reduce) == Enums::REDUCE_SEG, "fix bit-shift/mask issues!");
        static_assert( ((ibMulti>>Config::shift_buflay)&Config::mask_buflay) == Enums::BUF_MULTI, "fix bit-shift/mask issues!");
        static_assert( ((ibMulti>>Config::shift_iothread)&Config::mask_iothread) == Enums::IOTHREAD_NONE, "fix bit-shift/mask issues!");

        static uint_least16_t constexpr shmMulti = Enums::COMM_READ | (Enums::REDUCE_SEG << 3)
            | (Enums::BUF_MULTI << 6) | (Enums::IOTHREAD_NONE << 9);
    };
}//orm::

#endif // ORMPROPS_HPP_
