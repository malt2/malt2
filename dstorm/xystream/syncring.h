/** \file
 * synchronization ring concept
 */

/** Network graph for eventual consensus */
class SyncRing
{
public:
    enum uint32_t {
        NOPEER=-1U
    };
    enum RingType {
        DEFAULT     ///< A tree-like structure
    };

    /** data attached to each node 0..n-1 */
    struct RingData;

    /** Every \c node has a \c next node, and \em may have a \c peer
     * that is in another ring.
     *
     * - \c node and \c next form disjoint, cyclic, unidirectional
     *   token rings of bounded size.
     *   - Cyclic \e token rings have a single circulating token.
     * - \c node and \c peer are bidirectional connections \b between
     *   rings implying an <em>eventual consensus</em> operation.
     *   - the graph of rings and peers is fully connected and
     *     \b should allow a most central \em root ring to be identified.
     *   - The rings \em may be arranged in a tree to minimize the
     *     number of cross-ring peer connections, but could have more peers
     *     to promote consensus.
     * - The token's owner \b may modify content (the token),
     *   \b must pass it to \c next (overwriting next's version),
     *   and \b may pass it to a consensus peer.
     * 
     * - all transfer operations (pass to next, exchange with peer) \b must
     *   be free of RDMA artifacts (i.e. notify-wait-ack or related protocol
     *   must be designed)
     */
    struct RingConnections {
        uint32_t node;
        uint32_t next;
        uint32_t peer;      ///< optional bidirectional peer (-1U if absent)
        struct RingData d;
    };
    struct RingData
    {
        /** Active [default] nodes are allowed to modify the token before
         * passing it to \c next or \c peer nodes.
         *
         * - \c next nodes \em always accept the incoming token 'verbatim'
         *
         * - Consensus operates only between 2 active nodes or 2 inactive nodes;
         * - otherwise, the inactive partner just accepts the active token.
         */
        bool active;
        uint32_t pskp;      ///< if have peer, > 0 reduces freq of peer consensus
        uint32_t pcnt;      ///< counter to implement \c pskp of consensus
        // Peer consensus may have weights for each partner.
        // Weights might not be needed if there is no central ring! */
        //float prio;
    };

    /** We do \b not now how \c node to \c peer consensus is implemented. */
    SyncRing( uint32_t n                    ///< number of nodes
              , uint32_t max_ring_size      ///< maximum ring size
              , enum RingType type=DEFAULT
            );
    ~SyncRing() {}

private: // data
    enum RingType type;     ///< type of ring structre to construct
    uint32_t const n;       ///< n nodes, 0..n-1
    uint32_t const rmax;    ///< max ring size (rings may be smaller)
};
