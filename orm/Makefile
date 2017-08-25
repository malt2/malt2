.PHONY: all orm-all clean realclean cygwin-message check dirs libs hdr-compiles

LIB:=.
OBJ:=obj
DEPS:=dep
INCLUDE_DIR:=./include
TARGETS:=liborm.a

#CXXFLAGS:=-ggdb3

all: orm-all
	echo "TARGETS = $(TARGETS)"
	echo "make orm-all DONE"

# New: make liborm compile automatically agree with libdstorm compile options
DSTORM_ENV_MK:=$(wildcard ../dStorm-env.mk)
DSTORM_MK:=../dStorm.mk
ifneq ($(DSTORM_ENV_MK),)
$(warning "DSTORM_ENV_MK is $(DSTORM_ENV_MK)")
include $(DSTORM_ENV_MK)
ifneq ($(WITH_LIBORM),1)
$(error "Please recompile dstorm/ with at least WITH_LIBORM (or WITH_MPI or WITH_GPU) flags")
endif
$(warning "orm/Makefile: WITH_MPI is $(WITH_MPI)")
include $(DSTORM_MK)
$(warning "orm/Makefile: WITH_MPI is $(WITH_MPI)")
else
ifneq (,$(filter clean realclean,$(CMDGOAL)))
# Perhaps force a libdstorm recompile ?  But with what settings?
#WITH_LIBORM:=1
#WITH_MPI:=0
#WITH_GPU:=0
#$(MAKE -C ../dStorm)
#include ../dStorm.mk
$(error "${DSTORM_ENV_MK} not found. Please build libdstorm first: cd ../dstorm && ./mkit.sh {ORM|MPI|GPU} test")
endif
# FIND is usually set in dStorm.mk, but we bypassed dStorm.mk (and we use it for realclean)
include $(DSTORM_MK)
endif

# always also make this, since torch build depends on it (for now)
TARGETS+=liborm-pic.a

ifeq ($(CYGWIN),0)
#TARGETS+=liborm-pic.a
else
TARGETS+=cygwin-message
endif
#TARGETS+=libshm.a

$(TARGETS): | check dirs
orm-all: check dirs $(TARGETS)
	echo "TARGETS = $(TARGETS)"
	echo "make orm-all DONE"

MAIN_SOURCES:=orm_construct.c orm_util.c
MAIN_SOURCES+=orm_shm.c


ifeq ($(WITH_MPI),1)
MAIN_SOURCES+=orm_mpi.c
endif
ifeq ($(WITH_GPU),1)
MAIN_SOURCES+=orm_gpu.cpp
endif
# gradually move stubs from shm_TBD.o into real implementations ...
# Shorm is a C API and a C++ implementation of Orm for shared memory.
# Hopefully the user can mostly ignore the details, except to
# beware that the communication is based on a shared message
# queue, quite unlike the inifiband approach of predefined
# point-to-point fixed-length buffer areas within each segment.
SHM_SOURCES:= shm_TBD.c shm_proc_init.cpp shm_segment.cpp \
		shormOps.cpp shormMgr.cpp orm_any.cpp
# shm_printf .c and .cpp exist. WHICH ONE?	A: .cpp seems stale
SHM_SOURCES+=shm_printf.c

MPI_SOURCES:=
ifeq ($(WITH_MPI), 1)
MPI_SOURCES:= mpiorm.cpp
endif

GPU_SOURCES:=
ifeq ($(WITH_GPU), 1)
GPU_SOURCES:= gpuorm.cpp
#
# unfortunately, some of the cuda-8.0 helper includes generate warnings
#
CXXFLAGS+=-Wno-error
endif

SOURCES:=$(MAIN_SOURCES) $(SHM_SOURCES) $(MPI_SOURCES) $(GPU_SOURCES)

#
# dependency auto-generation
#
ifdef MAKECMDGOALS
CMDGOAL:=$(MAKECMDGOALS)
ifeq (x$(CMDGOAL),x)
CMDGOAL:=all
endif
else
CMDGOAL:=all
endif
#
include ../makefile.std-obj-deps
#
C_SOURCES:=$(filter %.c,$(SOURCES))
CPP_SOURCES:=$(filter %.cpp,$(SOURCES))
DEP_D_TARGETS:= $(patsubst %.c,$(DEPS)/%.d, $(C_SOURCES)) \
		$(patsubst %.cpp,$(DEPS)/%.d, $(CPP_SOURCES))
DEP_LD_TARGETS:=$(patsubst %.c,$(DEPS)/%.ld,$(C_SOURCES)) \
		$(patsubst %.cpp,$(DEPS)/%.ld,$(CPP_SOURCES))
ifeq (,$(filter clean realclean,$(CMDGOAL)))
include $(DEP_D_TARGETS)
include $(DEP_LD_TARGETS)
endif

$(TARGETS): | dirs
libs: dirs $(TARGETS)
dirs: $(DEPS) $(OBJ) $(LIB)
$(DEPS) $(OBJ) $(LIB):; mkdir -p $@

#
# orm library has OBJECTS and SHM_OBJECTS
#
#OBJECTS:=
#ifeq ($(CYGWIN),0)
#endif
#OBJECTS+= orm_shm.o orm_construct.o
##SHM_OBJECTS:=shm_TBD.o shm_printf.o shm_proc_init.o shm_segment.o
#SHM_OBJECTS+= shormOps.o shormMgr.o
OBJECTS:=	$(patsubst %.c,$(OBJ)/%.o,$(C_SOURCES)) \
		$(patsubst %.cpp,$(OBJ)/%.o,$(CPP_SOURCES))
OBJECTS_PIC:=	$(patsubst %.o,%.lo,$(OBJECTS))

liborm.a: $(OBJECTS)
	$(AR) rcs $@ $+
	@echo "TARGETS = $(TARGETS)"
	@echo "build target $@ DONE"
	@echo "OBJECTS_PIC = $(OBJECTS_PIC)"
liborm-pic.a: $(OBJECTS_PIC)
	$(AR) rcs $@ $+
	echo "TARGETS = $(TARGETS)"
	@echo "build target $@ DONE"

check: settings
	@echo 'CCFLAGS         $(CCFLAGS)'
	@echo 'CXXFLAGS        $(CXXFLAGS)'
	@echo 'MAIN_SOURCES    $(MAIN_SOURCES)'
	@echo 'SHM_SOURCES     $(SHM_SOURCES)'
	@echo 'MPI_SOURCES     $(MPI_SOURCES)'
	@echo 'SOURCES         $(SOURCES)'
	@echo 'C_SOURCES       $(C_SOURCES)'
	@echo 'CPP_SOURCES     $(CPP_SOURCES)'
	@echo 'OBJECTS         $(OBJECTS)'
	@echo 'OBJECTS_PIC     $(OBJECTS_PIC)'
	@echo 'WITH_MPI        $(WITH_MPI)'
	@echo 'WITH_GPU        $(WITH_GPU)'

HDRS:=orm.h orm_fwd.h ormProps.hpp orm_any.h
ifeq ($(WITH_GPU),0)
HDRS+=ipc_barrier.hpp layout.hpp shormMgr.hpp shormOps.hpp orm_shm.h
endif

ifeq ($(WITH_MPI),1)
HDRS+=ormConf.hpp
HDRS+=orm_mpi.h
HDRS+=mpiorm.hpp
endif
ifeq ($(WITH_GPU),1)
HDRS+=gpuorm.hpp
endif
HDRCOMPILER:=g++
$(OBJ)/%.gch: %              | $(OBJ)
	$(HDRCOMPILER) $(CXXFLAGS) -Wall -Werror -DTEST_COMPILE $< -o $@ 
$(OBJ)/%.gch: include/%      | $(OBJ)
	$(HDRCOMPILER) $(CXXFLAGS) -Wall -Werror -DTEST_COMPILE $< -o $@ 
$(OBJ)/%.gch: include/shm/%  | $(OBJ)
	$(HDRCOMPILER) $(CXXFLAGS) -Wall -Werror -DTEST_COMPILE $< -o $@
hdr-compiles: $(OBJ) $(patsubst %,$(OBJ)/%.gch,$(HDRS))
	@rm -f *.gch include/*.gch include/detail/*.gch $(filter %.gch,$^)
	@echo "GOOD - standalone header compilation succeeded (.gch files in $(OBJ) removed)"
cygwin-message:
	@echo "    Cygwin : orm_mpi will not be included in liborm.a"
	@echo "    Cygwin : orm_gpu will not be included in liborm.a"

clean:
	rm -f $(OBJECTS) $(SHM_OBJECTS) *.o *.a *.lo a.out *.gch
	${FIND} include -name '*.gch' -exec rm {} \;
realclean: clean
	rm -f $(TARGETS)
	@# remove entirely --- DANGEROUS if they are not "here"
	if [ -d "$(DEPS)" -a "$(DEPS)" != "." ]; then rm -rf "$(DEPS)"; fi
	if [ -d "$(OBJ)"  -a "$(OBJ)"  != "." ]; then rm -rf "$(OBJ)" ; fi
	if [ -d "$(LIB)"  -a "$(LIB)"  != "." ]; then rm -rf "$(LIB)" ; fi
#last line
