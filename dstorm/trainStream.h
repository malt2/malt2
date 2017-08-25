/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
/** \file
 * Bas idea is to overlap some calc with initial read of (compressed)
 * input data. */

/** \defgroup Streamers Data Streamers
 * Data reading may be slow. During data read, we may do some calculation and
 * partitioning of data.  Initial model is for one MASTER file reader that
 * distributes training examples.  This is for non-seekable streams like
 * .gz input files.
 *
 * Later, we can use lz4 to develop a seekable compressed stream with utilities
 * to split input files automatically across network to local disk.
 */
//@{
/** Initial, only MASTER has a file-connected example reader (\c ExFileStreamer)
 * WORKERS all have stream-connected network readers (\c ExGaspiStreamer)
 * MASTER has an \c ExMemStreamer directly attached
 *\verbatim
 * file --> MASTER (fast initial training/partitioning)
 *          ---> GaspiStreamer, destination = self or worker
 *               self:   --> redirect to own MemStreamer
 *               worker: --> worker's GaspiStreamer
 *                             --> worker's MemStreamer
 *\endverbatim
 *
 * - MASTER operation
 *
 *   - MASTER overlaps file-reading with some initial SGD (epoch 0).
 *   - MASTER decides on destination for each EX (self or other rank)
 *   - MASTER streams examples into output buffers, which
 *     dump to WORKERS across network (or remain on self)
 *   - MASTER finishes with EOF across network (or to self)
 *
 * - WORKER operation
 *
 *   - WORKERS read from network and iterate repeatedly within
 *     all known training examples (even during epoch 0)
 *   - After 'cb' trainings, do network output comm as usual.
 *   - Network reading of examples continues until EOF.
 *   - WORKER reader then switches to replay mode over all
 *     examples it has.  (Hopefully this fits in memory)
 *   - EOF read on MemStreamer begins "epoch 1",
 *     which cycles over all examples in memory (hopefully enough mem)
 *
 * - Training
 *
 *   - training may occur during MASTER operation, so MASTER may export
 *     more than just raw examples - perhaps some raw \em model info too.
 *     - Example: matrix factorization may develop crude low-rank
 *       <em>user model</em> info that gets transferred (?)
 *   - Assume ExMemStreamer & ExGaspiStreamer persist for all epochs.
 *   - Mainly ExMemStreamer mainly provides \em vector interface to data.
 *   - Later, may provide:
 *     - ExMemStreamer --> local disk "restreaming" if too much data
 *     - ExMemStreamer --> ExGaspiStreamer to tweak data partitioning
 *                         when more is known
 */
template< typename EX > class ExFileStreamer; // MASTER, rank 0
template< typename EX > class ExGaspiStreamer; // WORKER, rank != 0
template< typename EX > class ExMemStreamer;  // ALL
//@}


/** read from file, partition data via \c ExGaspiStreamer */
template< typename EX > class ExFileStreamer
{
public: // constants
    enum {
        NT_README,              ///< MASTER-->worker
        NT_ACK,                 ///< worker-->MASTER
        NT_EOF                  ///< MASTER-->worker
    };
    /** Ideally, SEG_FULL_TO_ONE of EX[oBufSz] vector of EX training examples.  */
    static const dStorm::SegIdx ExOut = 70U;

    /** When ExOut is full, send batch of training examples, then switch to
     * next buffer */
    static const uint32_t oBufSz;

public: // constructor and functions
    /** read from file, partition to \c selfMem \em or \c ExGaspiStreamer over network */
    ExFileStreamer( char const* filename
                    , typename ExGaspiStreamer<EX> &f
                    //, typename ExMemStreamer<EX>& selfMem
                  );

    /** May destruct and d.delete_segment if our NT_EOF has been NT_ACKed.
     * (Might this run in a thread?) */
    ~ExFileStreamer();

    /** read an example from disk file (MASTER)
     *  and (occasionally) distribute to WORKERS, with NT_README notification.
     *  At EOF from \c filename, we'll send out an NT_EOF notification
     *  through Dstorm to all WORKERS.  
     *  NT_EOF signals workers "begin epoch 1, you won't get more training examples from me".*/
    read( EX* ex, uint32_t n );

private:
    dStorm::Dstorm &d;
    typename ExGaspiStreamer<EX> &selfMem;
    //typename ExMemStreamer<EX> &selfMem;

    /** push an EX to the r'th oBuf containing a vector EX[oBufSz] items.
     * We block if !acked[buf].  */
    push( orm_rank_t r, EX &ex );

    /** We'll pingpong between sending from segment ExOut and ExOut+1, and
     * implement an ACK notification (from destinations notify_waitsome)
     * to really be careful about correctly send out the training examples.
     *
     * Every destination (worker) rank has its own pingpong buffer state,
     * since we might not fill all output buffers uniformly.  For example,
     * class balancing or matrix factorization may not do "round-robin" or
     * randomized distribution of training examples.
     *
     * \p buf[i=0..nProc-1] toggles two values, 0 or 1.  synced via
     * notifications with buf pingpong setting at destination. */
    uint32_t *buf;

    /** when ACKED (or constructed) \c acked[buf]==true. After initiating our
     * write_notify, we set acked[buf]=false. */
    bool acked[2];

}

template< typename EX > class ExGaspiStreamer : ExStreamer<EX>
{
public:
    static const iBufSz = typename ExFileStreamer<EX>::oBufs;   // match seg buffer sizes
    static const dStorm::SegIdx ExIn = typename ExFileStreamer<EX>::ExOut; // may as well match the segment handles too

    /** periodically we notify_waitsome for NT_README or NT_EOF.
     * After NT_README, we push our examples into selfMem.
     * After NT_EOF, we orm_notify NT_ACK to our orm source,
     * and propagate NT_EOF to our selfMem.
     */
    ExGaspiStreamer( dStorm::Dstorm &d, typename ExMemStreamer<EX>& selfMem );

    /** After our NT_EOF to selfMem has been ACKed we can be destroyed.  */
    ~ExGaspiStreamer();

    /** async push completion function */
    typedef void (*push_cb)( EX const* ex, uint32_t const n );

    /** push to \c selfMem or to other ranks via orm_write_notify.
     * \return std::future \c ret , so \c ret.get() only returns when
     * this \c push is guaranteed complete.  \c ex[n] can be overwritten
     * or freed safely after \c ret.get() returns.
     *
     * General idea:
     * \code
     * push:
     *     if( r == iProc ){
     *         pass ex[n] to selfMem immediately;
     *         promise<void> p; p.setvalue();
     *         return p.get_future();
     *     }else{ // BAD impl:
     *        future<void> f = async(launch::async, &ExGaspiStreamer<EX>::io,this,r,ex,n);
     *        return f;
     *     }
     * io (thread):
     *     block on ack of prev sends if nec. (bounded # of ongoing xfers)
     *     reserve nidRead and nidAck notification ids
     *     orm_write_notify( ... nidRead );
     *     orm_notify_waitsome( ... nidAck );
     *     free nidRead and nidAck notification ids
     * \endcode
     * \b BUT \c std::async is SLOW.
     * \todo reorganize ExGaspiStreamer::push to use a thread pool.
     */
    std::future<void> push( orm_rank_t r, EX const* ex, uint32_t const n );

private:

private:
    dStorm::Dstorm &d;
    typename ExMemStreamer<EX>& selfMem;

    /** worker pingpong setting. Ideally worker has just two buffers in the segment */
    uint32_t buf = 0U;

}

/** looks just like a vector of training data, but with a local_push method so it
 * can grow / be modified during use.  Hmmm. This means it needs some multithreading
 * suppport, because local_push may append/resize data from an ExGaspiStreamer
 * \b while the epoch 0 calculation is starting up !
 *
 * \c vector_type is private so we lock around just the required functions.
 * Our \c ex[] memory is non-resizable to allow for unlocked access by local_push
 * and accessors.  Accessors need to recognize that size() may periodically
 * change in value. If handling overflow to disk, our size() might even exceed
 * our \c ex[] capacity!
 */
template< typename EX > class ExMemStreamer
{
public:
    typedef typename std::vector<EX> vector_type;
    typedef uint64_t size_type;

    /** Initialize empty, but allow some memory reservation.  For fast operation,
     * we do not grow in size (but we might be able to return excess memory later).
     * If we bust our size, we throw, but later may wish to overflow to file.
    */
    ExMemStreamer( size_type const EXreserve );

    /** One example source could be a local \c ExFileStreamer who sends us \c ex[n] */
    void local_push( EX const* const ex, uint32_t const n );

    /** overwrite entries in \c idx with random indices.
     * Chooses unique examples if possible.
     * Approx uniform long-term distribution may be tricky if overflowing to disk. */
    void random( std::vector<size_type> &idx );

    /** \name thread-safe vector funcs */
    //@{
    size_type size() const;                          ///< if diskoverflow, can exceed \c ex.size()
    EX const& operator[]( uint64_t const i ) const;
    EX      & operator[]( uint64_t const i )      ;
    //@}
private:
    bool diskoverflow;  ///< \todo local disk overflow of ExMemStreamer
    vector_type ex;     ///< a non-relocatable vector of working memory
};

