####                                                                     ####
####   After your 'all:' target, include this to set up things     ####
####   in a standard fashion.  Avoid this fiddly stuff in many Makefiles ####
####                                                                     ####
#
# to work as an include file, need the directory of *this* file, dStorm.mk
#         (not the directory of the parent makefile)
#         Then we define subdirectories relative to "here"
# this file must be able to "make exports" without extra output (for bashrc script use)
#
# There are four main compilation flavors:
# 	1. ORM (liborm, NO notify-ack support, no MPI/GPU)
# 	2. MPI (liborm, NO notify-ack, no GPU)
# 	3. GPU (liborm, NO notify-ack, with cuda GPU code)
# TODO:
# 	extend notify-ack upwards to ORM, MPI and GPU compilations,
# 	by making changing the old IB-dependent functions to orm shims.
# 	May want an orm bool for notify-ack code blocks for transports
# 	like shm that may have stub functions for notify-ack!
##
SHELL:=/bin/bash
THIS_DIR:=
ifdef prj_DIR
# Where am I? for milde compile:
THIS_MKFILE:=$(prj_DIR)/src/dStorm.mk
THIS_DIR:=$(shell cd `dirname $(THIS_MKFILE)` && pwd -P)
endif
ifeq ($(THIS_DIR),)
THIS_MKFILE:=$(CURDIR)/$(lastword $(MAKEFILE_LIST))
THIS_DIR:=$(shell cd `dirname $(THIS_MKFILE)` && pwd -P)
endif
ifeq ($(THIS_DIR),)
# Where am I? for invocation as make -C <absolute path>
THIS_MKFILE:=$(lastword $(MAKEFILE_LIST))
THIS_DIR:=$(shell cd `dirname $(THIS_MKFILE)` && pwd -P)
endif
ifeq ($(THIS_DIR),)
$(info CURDIR $(CURDIR))
$(info THIS_MKFILE $(THIS_MKFILE))
$(info THIS_DIR <$(THIS_DIR)>)
$(error "What directory am I in? (could not set THIS_DIR)")
endif
# -j$(JOBS) to run parallel make on 80% of available CPUs
JOBS:=$(shell lscpu | grep '^CPU(s):' | gawk --source '//{j=int($$2 * 0.8); if(j<1){print 1;}else{print j}}')

ifeq ($(WITH_GPU),1)
.PHONY: settings echo warnings exports check-ssh check-CUDA_aware_MPI exports force gcc-flags
else
.PHONY: settings echo warnings exports check-ssh force gcc-flags
endif


ifeq ($(shell uname -o),Cygwin) # crippled platform, just for compile tests
# disable  infiniband (only can use shm)
CYGWIN:=1
WITH_GPU:=1
WITH_MPI:=1
WITH_LIBORM:=1
COMPILE_HOST := ${shell hostname}
USE_IB:=0
PIC_FLAG:=
FIND:=/usr/bin/find
else
CYGWIN:=0
# WITH_MPI require liborm, so liborm is a req'd common denominator
WITH_LIBORM?=1
WITH_MPI?=1
WITH_GPU?=1
CXX?=g++
CC?=gcc
COMPILE_HOST := ${shell hostname -s}
#USE_IB **used to** mean old transports with IB and TCP support
#                        and not liborm
USE_IB:=0
PIC_FLAG:=-fPIC
FIND:=find
endif

ifdef WITH_LIBORM
ifeq ($(WITH_LIBORM),)
WITH_LIBORM:=0
endif
endif

ifdef WITH_MPI
ifeq ($(WITH_MPI),)
WITH_MPI:=0
endif
endif

ifdef WITH_GPU
ifeq ($(WITH_GPU),)
WITH_GPU:=0
endif
endif

#CXXSTD := -std=c++11 -Wall -Werror
# NOTE: for gcc 4.8.2 must override to c++11 standard
GCC_VERSION := $(shell $(CXX) -dumpversion| awk 'BEGIN{FS="."}//{print $$1*10000+$$2*100+$$3}')
ifeq (1,$(shell if [ $(GCC_VERSION) -lt 40800 ]; then echo 1; fi))
GCC_48_OVERRIDES := -std=c++0x -Wno-attributes
else
GCC_48_OVERRIDES := -std=c++11
endif

DSTORM_DIR :=$(shell cd $(THIS_DIR)/dstorm && pwd -P)
GPU_DIRNAME:=
MPI_DIRNAME:=
MPI_DIR    :=
GPU_DIR    :=
ORM_DIR    :=$(shell cd $(THIS_DIR)/orm && pwd -P)
DSTORM_LIBRARY_TARGET:=
#
# WITH_GPU implies WITH_MPI implies WITH_LIBORM
#    (WITH_LIBORM can be 0/1 even without GPU or MPI enabled)
# After next set of tests, the WITH_* make variables are either 0 or 1
ifneq ($(WITH_GPU),0)
WITH_GPU:=1
WITH_MPI:=1
WITH_LIBORM:=1
GPU_DIRNAME:=/usr/local/cuda/
GPU_DIR    :=$(shell cd  $(GPU_DIRNAME) && pwd -P)
endif
#
ifneq ($(WITH_MPI),0)
WITH_MPI:=1
WITH_LIBORM:=1
MPI_DIRNAME:=/usr/lib/openmpi
MPI_DIR    :=$(shell cd  $(MPI_DIRNAME) && pwd -P)
CXX        :=mpic++
CC         :=mpicc
endif
#
ifneq ($(WITH_LIBORM),0)
WITH_LIBORM:=1
# (above) always need this one: ORM_DIR    :=$(shell cd $(THIS_DIR)/orm && pwd -P)
endif

#
ifeq ($(DSTORM_LIBRARY_TARGET),)
DSTORM_LIBRARY_TARGET:=libdstorm2-pic.a
endif
DSTORM_LIBRARY:=$(DSTORM_DIR)/lib64/$(DSTORM_LIBRARY_TARGET)

# For now, NOTIFY_ACK support is NOT available WITH_LIBORM
# (NOTIFY_ACK support **should** be supported always, eventually)
# TODO : support NOTIFY_ACK for MPI transport too
#ifeq ($(WITH_GASPLIB)$(WITH_MPI),10)
ifeq ($(WITH_LIBORM),0)
WITH_NOTIFYACK:=1
else
WITH_NOTIFYACK:=0
endif

IFBlibs:=
ORMlibs:=
MPIlibs:=
GPUlibs:=
# XXX should we add rpath settings for cuda? Can it differ in locn among machines?
ifeq ($(USE_IB),1)
IFBLIBS:=-libverbs
endif
ifeq ($(WITH_LIBORM),1)
ORMlibs:=-L$(ORM_DIR) -lorm -lrt
# ??
# ORMlibs:=-Wl,--whole-archive -L$(ORM_DIR) -lorm -Wl,--no-whole-archive -lrt
endif
ifeq ($(WITH_MPI),1)
MPIlibs:=-pthread -Wl,-rpath -Wl,/usr/lib/openmpi/lib -Wl,--enable-new-dtags -L/usr/lib/openmpi/lib -lmpi_cxx -lmpi
MPIlibs:=         -Wl,-rpath -Wl,/usr/lib/openmpi/lib -Wl,--enable-new-dtags -L/usr/lib/openmpi/lib -lmpi_cxx -lmpi
endif
ifeq ($(WITH_GPU),1)
GPUlibs:=-L$(GPU_DIR)/lib64 -lcuda -lcudart
endif
#
IFlibs+=-lpthread
ORMlibs+=$(MPIlibs) $(GPUlibs)


# Normally, you will be compiling and initiating tests from the IB "master" node
# Eventually, this should be set to the first entry in IBNET, and this machine:
#   - is the first machine listed in the machines files,

IBMASTER:=$(COMPILE_HOST)

# If we are compiling from a non-cluster machine
# we'll need to get some outputs from the IBMASTER node ... like a list of machines
ifeq ($(COMPILE_HOST),$(IBMASTER))
sshMASTER:= 
else
sshMASTER:=ssh $(IBMASTER)
endif

# Try IB first, then TCP
ifeq ($(USE_IB),1)
ORM_DEVICE=$(shell { if $$(ibaddr >&/dev/null); then echo IB; else echo TCP; fi; })
else
ORM_DEVICE=TCP
endif
ifeq ($(CYGWIN),1)
IBNET0:=
else
IBNET_DISC :=ibnetdiscover -l 2>/dev/null | grep '^Ca' | sed 'sx.*\"\([^ ]\+\).*x\1x'
#       warns if libibverbs ok, but no infiniband master
IBNET0:=${shell ${sshMASTER} ${IBNET_DISC}}
endif
# if (USE_IB) we also insist that ibnetdiscover gave us some machines in the ib cluster
ifeq ($(USE_IB),1)
ifeq ($(IBNET0),)
ORM_DEVICE:=TCP
endif
endif



#    USE_IB 0 means don't try to compile with infiniband
#    IBNET0 empty means tests won't run "here" using iB transport
# Some of our tests aren't scripted carefully so make sure compile host is first for TCP
ifeq ($(ORM_DEVICE),TCP)
ifeq ($(IBNET0),)
IBNET0:=$(COMPILE_HOST)
endif
endif

#
########### Default IBNET ~ all machines
#
# the machine list SHOULD contain the "master" node 0, which is "this machine" if at all possible
# If skipped, you may not see expected output! (except from 'printf' on the master node, if
#        ----- move all occurences of $(HOSTNAME) to the front -----
IBNET :=${shell echo $(IBNET0) \
	| gawk 'BEGIN{RS="[ ]"; m=""; o="";} \
		/[ \n]+$$/{gsub(/[ \n]+/,"",$$0);} \
		/$(COMPILE_HOST)/{if(m!="")m=m " ";m=m $$0; next;} \
		//{if(o!="")o=o " ";o=o $$0;} \
		END{r=(m " " o); gsub(/ +/," ",r); gsub(/ $$/,"",r); print r;}'}
#
########## some users may wish to use a subset of machines by default:
#
#IBNET := ${shell echo $(IBNET0) | gawk 'BEGIN{RS=" ";n=0; o=""} \
#	//{if($$0 == "") next; \
#	   if($$0 == "snake01 "){ \
#      		if(n > 1){o=" " o;} \
#		o=$$0 "GOTCHA" o; ++n;\
#	   }else{ \
#		if(n > 0){o=o " ";} \
#		o=o "<" $$0 ">  "; ++n; \
#	   } \
#	} \
#	END{print o;}'}
# Change the ordering to suit your machine allocation on you IB subnet
# default setup assuming IBMASTER same as COMPILE_HOST (override later)
CLUST4 :=$(shell echo $(IBMASTER) | cut -b 1-4)	# first 4 chars of "master" machine

#
# adjust above default of "use all IB machines"
# to reduced set, since we need to share cluster
#
ifneq (x$(IBNET0),x)	# If we had IB (from ibnetdiscover (on IBMASTER (=COMPILE_HOST)))
ifeq ($(CLUST4),fire)	   # have only a few machines on the shared csacluster...
IBNET :=fire22 fire23
else ifeq ($(USER),kruus)  # I've assigned myself these machines for daily tests
ifeq ($(COMPILE_HOST),snake01)
IBNET:=snake01 horse03 horse04
else
IBNET:=$(COMPILE_HOST)
endif
else ifeq ($(COMPILE_HOST),snake01)
IBNET :=snake01 snake03
else ifneq ($(IBNET),)	# If you are allowed access to full IB subnet, use all machines
# Old way had two separate IB subnets for horse/snake subnet:
# (move these sections upward to activate them)
else ifeq ($(CLUST4),hors)
IBNET :=horse01 horse02 horse03 horse04
else ifeq ($(CLUST4),snak)
IBNET :=snake01 snake02 snake03 snake04
endif
else
#		Some tests might be ok with orm_shm, on COMPILE_HOST
endif

################# standardize above settings
# sanitize (easy to leave trailing blanks etc when editing Makefile
IBNET :=${shell echo '$(IBNET)' | awk '{gsub(/ +/," ",$$0);sub(/  +$$/," ",$$0);print $$0;}'}
IBNET_N :=${shell echo '$(IBNET)' | wc -w}
# Now after any IBNET customizations, redetermine IBMASTER
IBMASTER:=$(shell echo '$(IBNET)' | gawk '//{print $$1}')
# ..... and redetermine if ssh command 'prefix' is needed to run cmd on IBMASTER
# If we are compiling from a non-cluster machine
# we'll need to get some outputs from the IBMASTER node ... like a list of machines
ifeq ($(COMPILE_HOST),$(IBMASTER))
sshMASTER:=
else
sshMASTER:=ssh $(IBMASTER)
endif

# assume compiling for local host, and
# local host reflects all machines in cluster
MACH_FLAGS:=${shell gcc -march=native -Q --help=target | grep -- '\(\(-msse\)\|\(-mssse\)\|\(-mavx\)\).*enabled' | awk '{if(mach!="")mach=mach " ";mach=mach $$1}END{print mach}'}

#$(info entered dStorm.mk CXXINC = <$(CXXINC)>)
#$(info entered dStorm.mk CXXFLAGS = <$(CXXFLAGS)>)
ifdef CXXINC
CXXINC:=$(CXXINC)
else
CXXINC:=
endif
# XXX should ** test ** for proper boost version and location XXX
CXXINC+=-I/opt/boost/include
ifdef CXXDEF
CXXDEF:=$(CXXDEF)
CXXDEF+=-D_FILE_OFFSET_BITS=64
else
CXXDEF:=-D_FILE_OFFSET_BITS=64
endif

# New: everybody needs this dir, if only for orm_fwd.h constants
CXXINC_OTHER:=

ifeq ($(WITH_GPU),1)
CXXINC_OTHER+=-I$(GPU_DIR)/include
CXXINC_OTHER+=-I$(GPU_DIR)/samples/common/inc
# cub: tentative, might not be there in all cuda versions?
#CXXINC_OTHER+=-I$(GPU_DIR)/include/thrust/system/cuda/detail
#  BUT ... thrust cub version is in CUB_NS_PREFIX thresut::system::cuda::detail ...
# XXX why in externals? I guess the one under GPU_DIR might be changing?
#CXXINC_OTHER+=-I$(THIS_DIR)/externals/cub
CXXDEF+=-DWITH_GPU=1
endif
ifeq ($(WITH_MPI),1)
CXXINC_OTHER+=-I$(MPI_DIR)/include
CXXINC_OTHER+=-I$(MPI_DIR)/include/openmpi
CXXDEF+=-DWITH_MPI=1
endif
ifeq ($(WITH_LIBORM),1)
# now always needed... at least for orm_fwd.h:
#CXXINC_OTHER+=-I$(THIS_DIR)/orm/include
#CXXINC+=$(CXXORM)
CXXDEF+=-DWITH_LIBORM=1
endif
INCLUDE_DIR:=$(DSTORM_DIR)/include
CXXINC+=-I$(INCLUDE_DIR) -I.
CXXINC+=-I$(ORM_DIR)/include $(CXXINC_OTHER)

#  warning: ‘used’ attribute does not apply to types [-Wattributes]
#           class DLLP app_State // global Script state
CXXFLAGS += -Wall -Werror -fopenmp -frtti $(GCC_48_OVERRIDES) $(CXXINC) $(CXXDEF)
CCFLAGS  += $(cc_FLAGS) -Wall -Werror -fopenmp $(CXXINC) $(CXXDEF)

GENCODE_FLAGS?=
ifeq ($(WITH_GPU),1)
NVCC      := $(GPU_DIR)/bin/nvcc -ccbin $(CXX)
# architecture
HOST_ARCH := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)
ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 ppc64le))
    TARGET_SIZE := 64
else ifeq ($(TARGET_ARCH),armv7l)
    TARGET_SIZE := 32
else
    $(error ERROR - unsupported value $(TARGET_ARCH) for TARGET_ARCH!)
endif
#
NVCCFLAGS := -m${TARGET_SIZE} $(GCC_48_OVERRIDES) $(CXXINC) $(CXXDEF)
# Gencode arguments XXX 53 60 or 62 for Titan X (Pascal) ???
# as of cuda-8.0, '20' has been deprecated
SMS ?= 30 35 37 50 52
ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
endif
#
ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
#
# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif
#
HIGHCODE_FLAGS := -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif

#CXXINC+=$(CXXINC_NOW)
# IBNET ... can be overridden below if your cluster machines aren't detected properly
# I compile switch to snake01, and want to use only a couple of machines, so
# I added a clause to NOT use the full subset of machines for my local tests

# ------------------- DONE with main settings ----------------------


# no 'C' code involved here
#CCFLAGS  += $(CXXINC) -ggdb3 -Wall -Wno-deprecated -Werror -D_FILE_OFFSET_BITS=64 $(MACH_FLAGS)
#CXXFLAGS += -fdata-sections -ffunction-sections -O0
#   add     -fdata-sections -ffunction-sections   to make tsc with minimal data functions (maybe not working great)
DBGFLAGS := -DVERBOSITY=1
OPTFLAGS := -Ofast -march=native -msse2 -fbuiltin
# -Ofast is -O3 with -ffast-math and some other non-compliant optimizations
RELFLAGS := $(OPTFLAGS) -DNDEBUG -ggdb3
#RELFLAGS := $(OPTFLAGS) $(DBG2FLAGS)
NVRELFLAGS := -DNDEBUG -use_fast_math --compiler-options "-O3 -ffast-math" -use_fast_math 

# This should be the first target:
exports:
	@echo 'export DSTORM_DIR=$(DSTORM_DIR);'
	@echo 'export IBMASTER=$(IBMASTER);'
	@echo 'export IBNET=$(IBNET);'

settings: mm2_echo warnings # gcc-flags
mm2_echo:
	@echo 'pwd -P         <'`pwd -P`'>'
	@echo 'CURDIR         <$(CURDIR)>'
	@echo 'THIS_MKFILE    <$(THIS_MKFILE)>'
	@echo "THIS_DIR       <$(THIS_DIR)>"
	@echo 'DSTORM_DIR     <$(DSTORM_DIR)>'
	@echo 'MPI_DIR        <$(MPI_DIR)>'
	@echo 'GPU_DIR        <$(GPU_DIR)>'
	@echo 'COMPILE_HOST   <$(COMPILE_HOST)>'
	@echo 'IBMASTER       <$(IBMASTER)>'
	@echo 'sshMASTER      <$(sshMASTER)>'
	@echo 'CLUST4         <$(CLUST4)>'
	#@echo 'IBNET_DISC '  <$(IBNET_DISC)
	@echo 'USE_IB         <$(USE_IB)>'
	@echo 'IBNET0         <$(IBNET0)>'
	@echo 'IBNET          <$(IBNET)>'
	@echo 'IBNET_N        <$(IBNET_N)>'
	@echo 'USER           <$(USER)>'
	@echo 'SHELL          <$(SHELL)>'
	@echo 'THIS_DIR       <$(THIS_DIR)>'
	@echo 'MACH_FLAGS     <$(MACH_FLAGS)>'
	@echo 'CMDGOAL        <$(CMDGOAL)>'
	@echo 'CXX            <$(CXX)>'
	@echo 'CXXDEF         <$(CXXDEF)>'
	@echo 'CXXINC_OTHER   <$(CXXINC_OTHER)>'
	@echo 'CXXINC         <$(CXXINC)>'
	@echo 'CXXFLAGS       <$(CXXFLAGS)>'
	@echo 'CC             <$(CC)>'
	@echo 'CCFLAGS        <$(CCFLAGS)>'
	@echo 'DSTORM_LIBRARY <$(DSTORM_LIBRARY)>'
	@echo 'IFBlibs        <$(IFBlibs)>'
	@echo 'ORMlibs        <$(ORMlibs)>'
	@echo 'MPIlibs        <$(MPIlibs)>'
	@echo 'GPUlibs        <$(GPUlibs)>'
	@echo 'WITH_LIBORM    <$(WITH_LIBORM)>'
	@echo 'WITH_MPI       <$(WITH_MPI)>'
	@echo 'WITH_NOTIFYACK <$(WITH_NOTIFYACK)>'
	@echo 'WITH_GPU       <$(WITH_GPU)>'
ifeq ($(WITH_GPU),1)
	@echo 'CUDA_aware_MPI <$(CUDA_aware_MPI)>'
	@echo 'NVCC           <$(NVCC)>'
	@echo 'NVCCFLAGS      <$(NVCCFLAGS)>'
	@echo 'GENCODE_FLAGS  <$(GENCODE_FLAGS)>'
endif

# All sub-makes will inherit these values for sure:
export WITH_LIBORM
export WITH_MPI
export WITH_GPU
export WITH_NOTIFYACK

warnings: | mm2_echo
	@if [ -z "$(IBNET0)" ]; then \
	  echo ""; \
	  echo "WARNING:"; \
	  echo "    We could not detect your infiniband machines."; \
	  echo "    1. Compile this on your 'root' inifiniband machine,"; \
	  echo "       preferably in a common network-accessible directory"; \
	  echo "    2. Edit the Makefile and set up your infiniband hosts so that"; \
	  echo "       [ssh IBMASTER] ibnetdiscover -l returns a set of machines."; \
	  echo "    COMPILE_HOST   = $(COMPILE_HOST)"; \
	  echo "    IBMASTER       = $(IBMASTER)"; \
	  echo "    We will continue compiling anyway..."; \
	  echo "    WIP: some tests MAY work, by running over shared memory"; \
	  echo ""; \
	fi
	# The next location should be MILDE_DIR if we are not running standalone
	@if [ ! -f "$(DSTORM_DIR)/../data/rcv1/rcv1.train.bin.gz" ]; then \
	  echo ""; \
	  echo "WARNING: We will compile, but it seems that"; \
	  echo "     $(DSTORM_DIR)/../data"; \
	  echo "  has not been set up with rcv1 datasets yet (perhaps should"; \
	  echo "  be a link to some network location that you prefer to use)."; \
	  echo ""; \
	fi

# compile directory, for IB, should always be a network-accessible pathname
check-ssh:
	# This can run a long time (or hang) if autofs or ssh has issues)
	@echo ' IB machines detected: IBNET = $(IBNET)'; \
	echo '    Is DSTORM_DIR $(DSTORM_DIR) accessible via ssh?'; \
	echo '    (ssh tests may be lengthy waiting for autofs mounts)'; \
	nwarn=0; \
	for m in $(IBNET); do \
	  false || echo -n "On $$m, is there a DSTORM_DIR $(DSTORM_DIR) ?"; \
	  sshTrial=`ssh $$m test -d $(DSTORM_DIR) && echo YES || echo NO`; \
	  false || echo "sshTrial = <$${sshTrial}>"; \
	  if [ "$${sshTrial}" = "NO" ]; then \
	    true || echo "WARNING: ssh to $$m did not find the DSTORM_DIR"; \
	    nwarn=$$(( $$nwarn + 1 )); \
	  fi; \
	done; \
	if [ "$${nwarn}" = "0" ]; then \
	  echo "GOOD: DSTORM_DIR present when ssh'ing to IB cluster machines"; \
	fi;
gcc-flags:      # print the CPU optimization flags used for CPU on this machine
	gcc -march=native -Q --help=target -v

ifeq ($(WITH_GPU),1)
CUDA_aware_MPI :=$(shell ompi_info --parsable --all | grep mpi_built_with_cuda_support:value)
check-CUDA_aware_MPI:
	@# You can compile and link successfully, with vanilla OpenMPI, but if
	@# it is not CUDA-aware, you segfault when you finally try to run ...
	@# So checking the suffix of the returned flag from mpi-info...
	@case $(CUDA_aware_MPI) in *false) \
	  echo "Please build MPI with cuda support first"; \
	  false ;; \
	esac
endif


# generate a cyclic list of "all machines" of sufficient length,
# then sort the first $* entries,
# so any repeated machines are listed consecutively
test%.conf:
	@rm -f $@
	@echo 'Target $@ with INBET_N=$(IBNET_N) and IBNET=<$(IBNET)>'
	@test $(IBNET_N) -gt 0 || { echo "OHOH, adapt Makefile to get your IBNET cluster machine list"; exit -1; }
	@#echo "Machine list based on sort of first $* entries in cyclic repetition of $(IBNET)"
	@n=$(IBNET_N); cyc='$(IBNET)'; \
	  while test $$n -lt $*; do cyc="$$cyc $$cyc"; n=$$((n + n)); done; \
          echo "n=$$n , cyc=$$cyc"; \
	  noecho(){ true; } ; \
	  noecho "IBNET machine list length $(IBNET_N) --> cyclic list of length n=$$n"; \
	  machlist0=`echo "$$cyc" | cut -d' ' -f1-$* | tr ' ' '\n' | sort | tr '\n' ' '`; \
	  echo "sorted machlist <$$machlist0>"; \
	  machlist=`echo -n "$$machlist0" \
	  | gawk 'BEGIN{RS="[ ]"; m=""; o="";} \
		/[ \n]+$$/{gsub(/[ \n]+/,"",$$0);} \
		/$(IBMASTER)/{if(m!="")m=m " ";m=m $$0; next;} \
		//{if(o!="")o=o " ";o=o $$0;} \
		END{r=m; if(o!=""){r=r " " o;}; gsub(/ +/," ",r); print r;}'`; \
	  echo "%@: $$machlist"; \
	  echo "$$machlist" | tr ' ' '\n' > $@;
here.conf:
	@# perhaps get a better way to do this? It used to be MACH1,
	@# but perhaps we can simply use the compiling machines hostname
	@rm -f here.conf
	@for i in `seq 1 16`; do echo $(IBMASTER) >> $@; done
	@echo "created here.conf with 16 copies of $(IBMASTER)"

