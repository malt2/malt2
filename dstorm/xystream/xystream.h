/** \file
 * Toying with doing matrix factorization partition while streaming
 * the training data from gzip file */

typedef uint32_t Tnode;
typedef float value_type;

/** Weighted edge in bipartite \f$x \rightarrow y\f$ graph.
 * \c x and \c y are arbitrary labels in their \c Tnode domain.
 */
struct WEdge
{

    Tnode x; Tnode y; value_type val;
};

/** Initial, only MASTER has a file-connected edge-reader.
 * WORKERS all have stream-connected network readers.
 *
 * MASTER does overlaps some initial SGD with partitioning.
 * MASTER streams examples into output buffers, which
 * dump to WORKERS across network.
 * MASTER finishes with EOF across network.
 *
 * WORKERS read from network and iterate repeatedly within
 * all known training examples.
 * After 'cb' trainings, do network output comm as usual.
 * Network reading of examples continues until EOF.
 * WORKER reader then switches to replay mode over all
 * examples it has.  (Hopefully this fits in memory)
 */
struct WEdgeReader {
    WEdgeReader( char const* fname ) = 0;
    WEdge* readOne() = 0;
};

/** Partitions maintained as linked circular lists of vertices
 *
 * Circular lists can be ordered (?)
 * Head of each partition points to some "first" item of each partition,
 * or is -1 for an empty partition.
 */
struct Pnodes
{
    /** Partition info */
    struct Part {
        Tnode head; ///< a first item in circular fwd list for this partition.
        Tnode sz;   ///< \# of items in this partition.
        // total value?
    };
    /** Partitions. Partition 0 can be used for incoming or unassigned elements. */
    std::vector<Tnode> parts;
    Tnode x;
    Tnode xnext;
    // value?
};

static void rand_unit_vec( float* x, uint32_t n ){
    double sumsq = 0.0;
    do {
        for(uint32_t i=0U; i<n; ++i){
            x[i] = drand48() - 0.5;
            sumsq += x[i] * x[i];
        }
    }while( sumsq < 1.e-8 );
    double const f = 1.0 / sqrt(sumsq);
    for(uint32_t i=0U; i<n; ++i){
        x[i] *= f;
    }
}

/** initialize \c vec1 to 1/sqrt(k) */
void winit1( float dotprod, float* vec1, float* vec2, uint32_t k )
{
    if( k==0 ){
        assert( nullptr == "useless winit1 with k=0" );
        return;
    }
    float const rootkinv = sqrt( dotprod / k );
    if( dotprod >= 1.e-6 ){
        for(uint32_t i=0U; i<k; ++i){
            vec1[i] = rootkinv;
        }
    }else{
        for(uint32_t i=0U; i<k/2; ++i){
            vec1[i] = 1.e-6f;
            vec1[i+1U] = -1.e-6f;
        }
        for(uint32_t i=k/2*2; i<k; ++i){
            vec1[i] = 1.e-6f;
        }
    }
}
/** initialize \c vec1 and \c vec2 such that their dot product is around \c dotprod */
void winit2( float dotprod, float* vec1, float* vec2, uint32_t k )
{
    if( k==0 ){
        assert( nullptr == "useless winit2 with k=0" );
        return;
    }
    float const rootkinv = sqrt( 1.0 / k );
    if( dotprod >= 1.e-6 ){
        for(uint32_t i=0U; i<k; ++i){
            vec1[i] = rootkinv;
            vec2[i] = rootkinv;
        }
    }else{
        for(uint32_t i=0U; i<k/2; ++i){
            vec1[i] = 1.e-6f;
            vec2[i] = 1.e-6f;
            vec1[i+1U] = 1.e-6f;
            vec2[i+1U] = -1.e-6f;
        }
        for(uint32_t i=k/2*2; i<k; ++i){
            vec1[i] = 1.e-6f;
            vec2[i] = 1.e-6f;
        }
    }
}
/** L2-norm of a vector */
float norm2( float const* x, float const n ){
    float sum=0.0f;
    for( uint32_t i=0U; i<n; ++i ) sum += x[i]*x[i];
    return sum;
}
/** squared distance between two vectors */
float dist2( float const* x, float const* y, float const n ){
    float sum=0.0f;
    for( uint32_t i=0U; i<n; ++i ){
        float diff = x[i] - y[i];
        sum += diff * diff;
    }
    return sum;
}
/** squared distance between unit vector in \c x direction and vector \c y.
 * If \c x or \c y has L2-norm < \c eps, use a shorter-length version to
 * avoid overflow. */
float unitxdist2( float const* x, float const* y, float const n, float const eps=1e-8f ){
    float sum=0.0f;
    float s1 = norm2(x,n);
    if( s1 > eps ) s1 = 1.0f / sqrt(s1);
    else           s1 = sqrt(eps);
    float s2 = norm2(y,n);
    if( s2 > eps ) s2 = 1.0f / sqrt(s2);
    else           s2 = sqrt(eps);
    for( uint32_t i=0U; i<n; ++i ){
        float diff = s1 * x[i] - s2 * y[i];
        sum += diff * diff;
    }
    return sum;
}
/** update a single pair of x,y k-vectors such that dot product x·y gets closer to target
 *
 * \p lamregx = lambda * reg_x;
 * \p lamregy = lambda * reg_y;
 */
static inline void wupdate( float const target, float* const wxx, float* const wyy, uint32_t const k,
                            float const lambda, float const lamregx, float const lamregy ){
    err = dot( wxx, wyy, k ) - target;
    //float tmp[k]; memcpy( &tmp, &yk[w->y*k], k*sizeof(float) );
    float const lamerr = lambda*err;
    for( uint32_t kk=0U; kk<k; ++kk ){
        float tmp = wxx[kk];
        wxx[kk] -= lamerr * wyy[kk] + lamregx * tmp    ;
        wyy[kk] -= lamerr * tmp     + lamregy * wyy[kk];
    }
}
void WEdgePart( WEdgeReader *in, uint32_t parts, uint32_t k/*rank of NMF vectors*/ )
{
    assert( k < 1000U );
    assert( k > 0U );
    WEdge* w;
    uint32_t const wsavSz = 1000000U;
    std::vector<WEdge> wsav; wsav.reserve(wsavSz);
    std::vector<WEdge> part[parts];             // tmp, before we distribute them.
    uint32_t xcnt[parts];               // how many x in each part?
    uint32_t xcnt_tot = 0U;             // how many x assigned
    uint32_t xedge[parts];              // how many training examples (x,y,val) in each part?

    // partition centroids initially each gets assigned a "random" NONZERO direction
    float* xcenter = new float[parts * k];     // k-dim average x-vector per partition
    memset( xcenter, 0, sizeof(float)*parts*k );
    //rand_unit_vec( xcenter, k );
    //for(uint32_t p=1U; p<parts; ++p)
    //    memcpy( &xcenter[p*k], &xcenter[0], k*sizeof(float) );

    uint32_t const nxInit = 10000U;
    uint32_t const nyInit = 10000U;
    // x/y observation counts: if either < 2(?), push to wsav
    std::vector<uint_least16_t> xedge;  xedge.reserve(nxInit);
    std::vector<uint_least16_t> yedge;  yedge.reserve(nyInit);
    std::vector<float> xk; xk.reserve(k*nxInit);
    std::vector<float> yk; yk.reserve(k*nyInit);
    std::vector<uint32_t> xpart; xparts.reserve(nxInit);   // to which part has x been assigned? (or -1U)
    // we have x|yedge counters, so don't need fancy signal
    //static_assert( std::numeric_limits<float>::has_quiet_NaN == true );
    //float const unseen = std::numeric_limits<float>::quiet_NaN;
    double const mu = 0.1;

    uint32_t nread = 0U;
    float const lamregx = lambda * reg_x;
    float const lamregy = lambda * reg_y;
    while( (w=in->readOne()) != nullptr )
    {
        ++nread;
        double err;
        if( ++nread < wsavSz/2 ) wsav.push_back(*w);
        // stats update
        if( w->x > xedge.size() ){
            xedge.resize( w->x+1U, 0U );
            xk.resize((w->x+1U)*k);
            xpart.resize(w->x+1U, -1U); // -1U ~ unassigned
        }
        if( w->y > yedge.size() ){
            yedge.resize( w->y+1U, 0U );
            yk.resize((w->y*k+1U)*k);
        }
        //++xedge[w->x]; ++yedge[w->y];

        if( xedge[w->x] == 0U ){
            ++xedge[w->x];
            if( yedge[w->y] == 0U ){ // unseen x, y
                ++yedge[w->y];
                winit2( w->val, &xk[w->x*k], &yk[w->y*k], k );
            }else{ // unseen x, known y
                ++yedge[w->y];
#if 1
                // Hmmm. should really do an exact scaling of x s.t. dot product target exactly satisfied.
                // Oh... do we assume dot product targets are all positive ???
                winit1( w->val, &xk[w->x*k], k );
                wupdate( w->val, &xk[w->x*k], &yk[w->y*k], lambda, lamregx, lamregy );
#else
                //  streamlined to ...
                float const dok = w->val / k;
                float const x1 = (dok > 1.e-6f? dok < 1.e+6f? (float)(std::sqrt(dok))
                                  :                1.e+3f
                                  : 1.e-3f);
                if( dok > 1.e-6f ) { // use x1 for all k entries of x
                    float * const wxx = &xk[w->x*k];
                    float * const wyy = &yk[w->y*k];
                    err = dot( wxx, wyy, k ) - target;
                    float const lamerr = lambda*err;
                    float const regx1  = lamregx * x1;
                    float const lamerrx1 = lamerr * x1;
                    for( uint32_t kk=0U; kk<k; ++kk ){
                        wxx[kk] -= lamerr * wyy[kk] + regx1            ;
                        wyy[kk] -= lamerrx1         + lamregy * wyy[kk];
                    }
                }else{ // alternate x1, -x1 for initial x
                    float const x2 = -x1;
                    float * const wxx = &xk[w->x*k];
                    float * const wyy = &yk[w->y*k];
                    err = dot( wxx, wyy, k ) - target;
                    //float tmp[k]; memcpy( &tmp, &yk[w->y*k], k*sizeof(float) );
                    float const lamerr = lambda*err;
                    float const lamregx1 = lamregx*x1;
                    float const lamregx2 = -lamregx1;
                    float const lamerrx1 = lamerr*x1;
                    float const lamerrx2 = -lamerrx1;
                    float const lamregx1 = lamreg*x1;
                    float const lamerrx1 = lamerr*x1;
                    for( uint32_t kk=0U; kk<k/2; ++ ++kk ){
                        wxx[kk  ] -= lamerr * wyy[kk  ] + lamregx1           ;
                        wyy[kk  ] -= lamerrx1           + lamregy * wyy[kk  ];
                        wxx[kk+1] -= lamerr * wyy[kk+1] + lamregx2           ;
                        wyy[kk+1] -= lamerrx2           + lamregy * wyy[kk+1];
                    }
                    for( uint32_t kk=k/2*2; kk<k; ++kk ){
                        wxx[kk  ] -= lamerr * wyy[kk  ] + lamregx1           ;
                        wyy[kk  ] -= lamerrx1           + lamregy * wyy[kk  ];
                    }
                }
#endif
            }
            wsav.push_back(*w);
            continue;
        }
        if( yedge[w->y] == 0U ){ // known x
            winit1( w->val, &yk[w->y*k], k );
            wupdate( w->val, &xk[w->x*k], &yk[w->y*k], lambda, reg_x, reg_y );
            continue;
        }
        // both x and y have been seen before, no need to initialize, just do an SGD step.
        wupdate( w->val, &xk[w->x*k], &yk[w->y*k], lambda, reg_x, reg_y );
        if( xpart[w->x] == -1U ){ // unassigned
            //    goodness: low unit-vector-distance and low example load
            uint32_t pclose = 0U;
            float pdist = unitxdist2( &xk[w->x*k], &xcenter[0], k );
            float pload = xedge[0] / ((xcnt_tot+1U)/parts);  // uniform load ~ pload==1.0
            float pscore = pdist + pload;   // equal weighting ?
            for(uint32_t p=1U; p<parts; ++p){
                float const d = unitxdist2( &xk[w->x*k], &xcenter[p*k], k );
                float const l = xedge[p] / ((xcnt_tot+1U)/parts);
                if( d+l < pscore ){
                    pscore = d+l;
                    pclose = p;
                }
            }
            // assign x to part pclose
            xpart[w->x] = pclose;
            ++xcnt[pclose];
            ++xcnt_tot;
        }
        uint32_t pclose = xpart[w->x];
        // assign training data (x,y,val) to node of x
        std::vector<WEdge> & ptrain = part[pclose]
        ptrain.push_back(*w);
        if( ptrain.size() >= 100U ){
            // flush ptrain (send these training examples to xpart[w->x])
        }
        // update "central" value of x's assigned partition
        float const m1 = xedge[pclose] / float{xedge[pclose]+1U};
        float const m2 = (1.0-m2);
        for(uint32_t i=0U; i<k; ++i){
            xcenter[pclose*k+i] = m1 * xcenter[pclose*k+i] + m2 * xk[w->x*k+i];
        }
    }// loop over all input
    // loop again over items that might not be so well initialized
    for( auto s: wsav ){
        wupdate( s.val, &xk[s.x*k], &yk[s.y*k], lambda, lamregx, lamregy );
        // cut'n'paste from above loop -- encapsulate in an x-assigner CLASS XXX
        if( xpart[w->x] == -1U ){ // unassigned
            //    goodness: low unit-vector-distance and low example load
            uint32_t pclose = 0U;
            float pdist = unitxdist2( &xk[w->x*k], &xcenter[0], k );
            float pload = xedge[0] / ((xcnt_tot+1U)/parts);  // uniform load ~ pload==1.0
            float pscore = pdist + pload;   // equal weighting ?
            for(uint32_t p=1U; p<parts; ++p){
                float const d = unitxdist2( &xk[w->x*k], &xcenter[p*k], k );
                float const l = xedge[p] / ((xcnt_tot+1U)/parts);
                if( d+l < pscore ){
                    pscore = d+l;
                    pclose = p;
                }
            }
            // assign x to part pclose
            xpart[w->x] = pclose;
            ++xcnt[pclose];
            ++xcnt_tot;
        }
        uint32_t pclose = xpart[w->x];
        // assign training data (x,y,val) to node of x
        std::vector<WEdge> & ptrain = part[pclose]
        ptrain.push_back(*w);
        if( ptrain.size() >= 100U ){
            // flush ptrain (send these training examples to xpart[w->x])
        }
        // update "central" value of x's assigned partition
        float const m1 = xedge[pclose] / float{xedge[pclose]+1U};
        float const m2 = (1.0-m2);
        for(uint32_t i=0U; i<k; ++i){
            xcenter[pclose*k+i] = m1 * xcenter[pclose*k+i] + m2 * xk[w->x*k+i];
        }
    }
    delete[] xcenter;
}
