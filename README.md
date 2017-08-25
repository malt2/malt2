# MALT-2:Distributed Data-Parallel Learning for Torch

## About

MALT-2 is a distributed data-parallel machine learning system for  [Torch](http://torch.ch).

MALT-2 is a ML parallelization framework to paralleize any existing ML application.
The system is designed to be simple to use and easy to extend, while
maintaining efficiency and state-of-the-art accuracy.

* Easy to add to existing code general-purpose interface, requires only changing optimization type to dstsgd (distributed SGD).
* Support for multi-machine, multi-GPU training with CUDA implementations for distributed parameter averaging.
* Includes C++ and Lua interface to extend existing code. Support for Torch and NEC MiLDE.
* Easily extend your existing Torch code with minimal changes.
* Explore existing distributed GPU apps over Resnets, and large language models.
* Various optimizations such as sparse-reduce, NOTIFY_ACK to accelerate distributed model training

## Building MALT with Torch

### Requirements

* [Torch](http://torch.ch)
* [MPI (OpenMPI or MPICH)](https://www.open-mpi.org/) built with CUDA support. If you are using Ubuntu 16.04, you can use these [packages](https://github.com/asimkadav/ompi).
* [Boost (1.54 or higher)](http://www.boost.org/)

### Setup

### Install Torch, MPI, Boost and CUDA (if using GPU).

* Checkout the latest version of MALT-2 from [github](https://github.com/malt-2)

```
git clone https://github.com/malt-2/malt.git
```

### Setup the environment variables

### Source your torch/cuda/MKL environment:
on some machines, you might need things something like:
```
source [torch-dir]/install/bin/torch-activate
source /opt/intel/mkl/bin/intel64/mklvars.sh intel64
```

If using modules, you can try:

```
module install icc cuda80 luajit
```
### To build everything including dstorm, orm and torch, just type:
````
make
```

### Component-wise build

To build componenet-wise (not required if using make above):

#### Build the dstorm directory, run:
```
./mkit.sh GPU test
```
You should get a `SUCCESS` as the output. Check the log files to ensure the build is successful.

The general format is:
```
./mkit.sh <type> 
```

where TYPE is: 
          or MPI (liborm  + mpi)
          or GPU (liborm + mpi + gpu)
A side effect is to create ../dstorm-env.{mk|cmake} environment files, so lua capabilities
can match the libdstorm compile options.

#### Build the orm


```
./mkorm.sh GPU
```

#### Building Torch packages. With Torch environment setup, install the malt-2 and dstoptim (distributed optimization packages)

```
cd dstorm/src/torch
rm -rf build && VERBOSE=7 luarocks make malt-2-scm-1.rockspec >& mk.log && echo YAY #build and install the malt-2 package
cd dstoptim
rm -rf build && VERBOSE=7 luarocks make dstoptim-scm-1.rockspec >&mk.log && echo YAY # build the dstoptim package
```


### Test

* A very basic test is to run th and then try, by hand,
```
require "malt2"
```

### Run a quick test.


* With MPI, then you'll need to run via mpirun, perhaps something like:
```
mpirun -np 3 `which th` `pwd -P`/test.lua mpi 2>&1 | tee test-mpi.log
```

* if GPU,
```
mpirun -np 3 `which th` `pwd -P`/test.lua gpu 2>&1 | tee test-GPU-gpu.log
```

* NEW: a `WITH_GPU` compile can also run with MPI transport
```
mpirun -np 3 `which th` `pwd -P`/test.lua mpi 2>&1 | tee test-GPU-mpi.log
```

default transport is set to the "highest" built into libdstorm2: GPU > MPI  > SHM
```
mpirun -np 3 `which th` `pwd -P`/test.lua 2>&1 | tee test-best.log
```

### Running over multiple GPUs.
* MPI only sees the hostname. By default, on every host, MPI jobs enumerate the
GPUs and start running the processes. The only way to change this and run on
other GPUs in a round-robin fashion is to change this enumeration for every
rank using `CUDA_VISIBLE_DEVICES`. An example script is in `redirect.sh` file
in the top-level directory.

* To run:
```
mpirun -np 2 ./redirect.sh `which th` `pwd`/test.lua
```
This script assigns available GPUs in a round-robin fashion. Since MPI requires
visibility of all other GPUs to correctly access shared memory, this script only
changes the enumeration order and does not restrict visibility.

## Applications

### Now we can run simple torch demos such as distributed linear-regression or imagenet.
Follow instructions here.

