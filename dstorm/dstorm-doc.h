/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
/** @defgroup Algs Distributed Data Parallelism

\section RW-DDA Gradient averaging with RW-DDA (Cun Mu)

Cun's algorithm used undirected graphs, and the convergence proof assumes
synchronous communication.  I might use \f$|\mathcal{I}_k^T|\f$
in-edge count for \f$\mathrm{deg}(k)\f$, to try to adapt it to async operation.

- On every node k, loop over local training examples
  - Iteration \f$++t\f$: update the dual gradient vector
    - choose example \f$i\f$
    - calculate local gradient \f$\mathbf{g}_i\f$ given current model \f$\mathbf{x}_k^t\f$
  - \em reduce: \b sum \f$|\mathcal{I}_k^t|\f$ available in-edge \b gradient contributions (dual vectors)
    \f$\mathbf{z}_k^{in} \equiv \sum_{j \in \mathcal{I}_k^t}\mathbf{z}_{j\rightarrow k}\f$
  - and incorporate gradient \f$\mathbf{g}_i\f$ into local dual \f$\mathbf{z}_k\f$:
    - \f[ \mathbf{z}_k^{t+1} = \frac{\mathbf{z}_k^{in} + \mathbf{z}_k^t + \mathbf{g}_i}{|\mathcal{I}_k^t|+1} \f]
  - if \c (t\%comm_batch==0)
    - <EM>store,push</EM>: push dual \f$z_k^{t+1}\f$ via out-edges \f$\mathcal{O}_k\f$
  - choose learning rate \f$\eta_k^t = \eta_0/\sqrt{t}\f$ and apply dual gradient \f$\mathbf{z}_k^{t+1}\f$:
    - \f[ \mathbf{x}_k^{t+1} =  - \eta_k^t \cdot \mathbf{z}_k^{t+1} \f]
  - update running average <B>consensus model</B>:
    - \f[ \hat{\mathbf{x}}_k^{t+1} = \sum_{i=1}^{t+1} \mathbf{x}_k^i / (t+1)
      = \frac{t}{t+1}\hat{\mathbf{x}}_k^t + \frac{1}{t+1}\mathbf{x}_k^{t+1} \f]

  - <B>Original algorithm:</B>
    - \c comm_batch is 1 (push every local update of \f$\mathbf{z}_k^t\f$ by \f$\mathbf{g}_i\f$)
    - was synchronous, or with all send rates having distribution (exponential)
      - so it used \f$\mathrm{deg}(k)+1 = |\mathcal{I}_k|\f$ as a \em constant divisor,
        instead of \c nReduce=\f$|\mathcal{I}_k^t|\f$
      - if one async node \f$j\f$ were consistently faster, it would get overrepresented by sending
        out more \f$\mathbf{z}_{j\rightarrow k}\f$, and it is not clear how this could get
        corrected for async RW-DDA
    - disallows losing contributions to the incoming sum \f$z_k^{in}\f$ due to
      async RDMA overwrites.
    - might use RDMA "sum" reduction (if possible for floats!)
    - REDUCE_SUM_RBUF segment policy seems good for now. A new REDUCE_SUM_AVG_RBUF_OBUF
      policy could be supported. There are serious torn read issues with any
      REDUCE_SUM_RBUF_OBUF policy -- don't do that!

  - ? Rewrite in terms of model-deltas (for Leon-GPL, where gradient is not easily provided)

  - Note: x = -(gamma/sqrt(t)) * z is for the proximal projection,
    \f[ \prod_\chi^\psi (\mathbf{z},\alpha):=\mathrm{arg\,min}_{\mathbf{x}\in \chi} \{ <\mathbf{z},\mathbf{x}> + \frac{1}{\alpha}\psi(\mathbf{x}) \} \f]
    where \f$\psi(x)=||x||^2\f$ and the problem being solved is
    \f[ \mathrm{min}_{x\in \chi} \frac{1}{n}\sum_{i=1}^n f_i(x) \mathrm{   subject to } x\in \chi \f]
  - Duchi explains this as:

    - The intuition underlying this algorithm is as follows: given the current
    iterate , the next iterate \f$x^{t+1}\f$ to chosen to minimize an averaged
    first-order approximation to the function \f$f\f$, while the proximal
    function \f$\psi\f$and stepsize \f$\alpha^t>0\f$ enforce that the iterates
    \f${x^t}_{t=0}^{\inf}\f$ do not oscillate wildly. The algorithm is similar
    to the “follow the perturbed leader” algorithms ...

  \subsection rw-dda with comm_batch

Here the idea is that we make local updates and reduce communication time.
We can still \em reduce whenever we notice input

- On every node k, loop over local training examples
  - update the dual gradient vector
  - choose example \f$i\f$
  - calculate local gradient \f$\mathbf{g}_i\f$ given current model \f$\mathbf{x}_k^t\f$
  - \em reduce: \b sum \f$|\mathcal{I}_k^t|\f$ available in-edge \b gradient contributions (dual vectors)
    \f$\mathbf{z}_k^{in} \equiv \sum_{j \in \mathcal{I}_k^t}\mathbf{z}_{j\rightarrow k}\f$
  - and incorporate gradient \f$\mathbf{g}_i\f$ into local dual \f$\mathbf{z}_k\f$:
    - \f[ \mathbf{z}_k^{t+1} = \frac{\mathbf{z}_k^{in} + \mathbf{z}_k^t + \mathbf{g}_i}{|\mathcal{I}_k^t|+1} \f]
  - if (i%comm_batch == 0)
    - <EM>store,push</EM>: push dual \f$z_k^{t+1}\f$ via out-edges \f$\mathcal{O}_k\f$
  - choose learning rate \f$\eta_k^t = \eta_0/\sqrt{t}\f$ and apply dual gradient \f$\mathbf{z}_k^{t+1}\f$:
    - \f[ \mathbf{x}_k^{t+1} =  - \eta_k^t \cdot \mathbf{z}_k^{t+1} \f]
  - update running average <B>consensus model</B>:
    - \f[ \hat{\mathbf{x}}_k^{t+1} = \sum_{i=1}^{t+1} \mathbf{x}_k^i / (t+1)
      = \frac{t}{t+1}\hat{\mathbf{x}}_k^t + \frac{1}{t+1}\mathbf{x}_k^{t+1} \f]

\section MA-SGD Model-averaging SGD

Here, we use data splitting to split examples amongst compute ranks.
Each rank performs the following:

- On every node k, loop over local training examples
  - local =pdate of model \f$\mathbf{x}_k\f$:
    - choose example \f$i\f$
    - calculate gradient \f$\mathbf{g}_i\f$ given current model \f$\mathbf{x}_k^t\f$
    - choose learning rate \f$\eta_k^t\f$ and apply gradient
      - \f$\mathbf{x}_k^{t'} = \mathbf{x}_k^t + \eta_k^t \cdot \mathbf{g}_i\f$
  - \em reduce: \b average with any available in-edge \b models
    - \f[ \mathbf{x}_k^{t+1} = \frac{ \mathbf{x}_k^{t'} + \sum_{j \in \mathcal{I}_k}\mathbf{x}_j }{ |\mathcal{I}_k + 1| } \f]
  - <EM>store,push</EM>: push model via out-edges \f$\mathcal{O}_k\f$

Leon-GPL svm examples also provide a roughly equivalent alternative that transmits
<B>model deltas</B> instead of the models.

Here, we explored rather sparse graphs, and can use various levels of synchronization
(barrier, notify+ack, async).

*/
#include "segInfo.hpp"
#include "segVecDense.hpp"
//#include "segTensor.hpp"
namespace dStorm {
/** \namespace user
 * - TBD code movement:
 *   - Dstorm-supplied example format \ref dStorm::user::Seg_VecDense could be here
 *     - but \ref dStorm::SegImpl, \ref dStorm::SegBase, and \ref dStorm::SegInfo remain \em generic.
 *   - Leon-GPL code, which implements two new data structures for sparse
 *     and dense vectors particular to a specific ASGD (asynchronoous
 *     gradient descent) implementation.  Implementation code for those
 *     vectors could be in the dStorm::user namespace.
 */
}
