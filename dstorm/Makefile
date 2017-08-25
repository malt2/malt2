#
# When building standalone, we build libdstorm2.a without -fPIC
# When building from milde, we build libdstorm2 with -fPIC
# Capabilities: WITH_GPU WITH_LIBORM WITH_MPI are set in dStorm.mk include file
##
LIB:=lib64
OBJ:=obj
DEPS:=dep
INCLUDE_DIR:=./include
LOG:=test
ENVFILE:=$(INCLUDE_DIR)/dstorm_env.h
DSTORM_ENV_MK:=../dStorm-env.mk
DSTORM_ENV_CMAKE:=../dStorm-env.cmake

TARGETS:=
TESTCONF+=test2.conf test4.conf test8.conf test16.conf # err if bad IBNET, so last
LINKS:= test_utils.c test_utils.h one_dies.c

.PHONY: all clean force check-CUDA_aware_MPI run force-run hdr-compiles doc
.PHONY: dirs test gcc-flags lib-PIC dstorm-all env-files

#
######################################### make exports section ###############
# also defines targets test%.conf, settings, echo, warnings, check-ssh
# and sets a number of handy makefile variables
#   such as WITH_GPU, WITH_LIBORM and WITH_GPU (default zero, value set to either 0 or 1)
#             (many viewable by 'make settings')
all: dstorm-all
include ../dStorm.mk

ifdef MAKECMDGOALS
CMDGOAL:=$(MAKECMDGOALS)
else
CMDGOAL:=all
endif
#
######################################### make exports section DONE ###########
ifeq ($(WITH_GPU),1)
TESTLIBS:=
##TARGETS+=$(OBJ)/ionet/userIoNet.o
##TARGETS+=$(OBJ)/ionet/globIoNet.o
##TARGETS+=$(OBJ)/ionet/scalIoNet.o
#TARGETS+=$(OBJ)/dstorm_net.o
#TARGETS+=$(OBJ)/segInfo.o
#TARGETS+=$(OBJ)/segVecGpu.o
#TARGETS+=$(OBJ)/dstorm_fwd.o
#TARGETS+=$(OBJ)/dstorm.o
##TARGETS+=$(OBJ)/ionet/globIoNet.ptx
##TARGETS+=$(OBJ)/ionet/userIoNet.ptx
##TARGETS+=$(OBJ)/ionet/scalIoNet.ptx
##TARGETS+=$(OBJ)/dstorm_net.ptx
##TARGETS+=$(OBJ)/segInfo.ptx
##TARGETS+=$(OBJ)/segVecGpu.ptx
##TARGETS+=$(OBJ)/dstorm_fwd.ptx
##TARGETS+=$(OBJ)/dstorm.ptx
else
TESTLIBS:=$(LIB)/libdstorm2-pg.a
TARGETS+=testUserIoNet
TARGETS+=prtIoNet
TARGETS+=scalTest
endif
TARGETS+=$(LIB)/libdstorm2.a
TARGETS+=$(LIB)/libdstorm2-pic.a

dstorm-all: $(DSTORM_ENV_MK) warnings check-CUDA_aware_MPI dirs $(TARGETS) # test

#
#
# ubuntu might require a -fPIC static library (don't remember why)
# cygwin compiles everything with PIC code, so don't need a -pic library.

# MILDE .make file builds 'all' and 'lib-PIC' targets,
# because milde require PIC code in a ".dll" 
# New: our torch/ code also requires pic code
lib-PIC: $(LIB)/libdstorm2-pic.a

RUN_LOGS:=
RUN_LOGS+=test/prtIoNet-equiv.log
RUN_LOGS+=test/scalTest.log test/scalTest.vg test/prtIoNet.vg
run: warnings check-ssh
	@# simpler than enforcing test sequentiality...
	$(MAKE) -j1 $(RUN_LOGS)
force-run:
	rm -f $(RUN_LOGS)
	$(MAKE) -j1 $(RUN_LOGS)

# XXX add auto dependency tracking to test binaries too !
# test global (nonscalable) io net code in globIoNet* files
testUserIoNet: test/testUserIoNet.cpp ionet/userIoNet.cpp ionet/globIoNet.cpp ionet/scalIoNet.cpp \
		include/ionet/globIoNet.hpp include/ionet/userIoNet.hpp include/ionet/scalIoNet.hpp
	$(CXX) -I. -std=c++11 -Wall -Werror -g $(CXXINC) \
		$(filter %.cpp,$^) -o $@
#	$(VALGRINDX) ./$@ > test/testUserIoNet.vg 2>&1
prtIoNet: test/prtIoNet.cpp ionet/globIoNet.cpp ionet/scalIoNet.cpp ionet/userIoNet.cpp
	$(CXX) -I. -std=c++11 -Wall -Werror -g $(CXXINC) $^ -o $@
prtIoNet-old: test/prtIoNet.cpp ionet/globIoNet.cpp ionet/scalIoNet.cpp ionet/userIoNet.cpp
	$(CXX) -I. -std=c++11 -Wall -Werror -g $(CXXINC) -DOLDIONET=1 $^ -o $@
test/prtIoNet-equiv.log: prtIoNet
	{ ./$< test 20; } > $@ 2>&1 && echo 'OK... $@  new and old IoNet impls equivalent!'
test/prtIoNet.vg: prtIoNet
	$(call VGLOG,./prtIoNet)

# test scalable io net code in mkIoNet* files
scalTest: test/scalTest.cpp ionet/globIoNet.cpp ionet/scalIoNet.cpp ionet/userIoNet.cpp dstorm_net.cpp dstorm_fwd.cpp
	$(CXX) -std=c++11 -Wall -Werror -g $(CXXINC) \
		$^ -o $@
test/scalTest.log: scalTest
	{ ./scalTest old 8 && ./scalTest new 8 && ./scalTest both 8; } > $@ 2>&1 &&  echo "OK... $@"
test/scalTest.vg: scalTest
	$(call VGLOG,./scalTest both 8)

libs: dirs $(TARGETS)
$(TARGETS): | dirs

# 'make all' on borg (non-infiniband) can COMPILE if I supply some IB cluster machine
#IBMASTER:=snake01
# of course borg is Ubuntu 14, while snake01 is some old CentOs
#    so trying to run the binaries created by borg won't work
#    "version `GLIBC_2.14' not found"



# $(TESTCONF)

dirs: $(DEPS) $(DEPS)/ionet $(OBJ) $(LIB)
# dstorm_env.h must be made before dependencies !
$(DEPS): | env-files
$(DEPS) $(OBJ) $(LIB) $(DEPS)/ionet $(OBJ)/ionet $(DEPS)/test $(OBJ)/test:; mkdir -p $@ $(OBJ)/ionet

# SOURCES is used for auto-dependency generation
SOURCES:=
SOURCES+=dstorm_fwd.cpp
SOURCES+=dstorm_net.cpp
SOURCES+=dstorm_dbg.cpp
SOURCES+=ionet/userIoNet.cpp
SOURCES+=ionet/globIoNet.cpp
SOURCES+=ionet/scalIoNet.cpp
SOURCES+=segInfo.cpp
SOURCES+=dstorm_push.cpp
ifeq ($(WITH_GPU),0)
SOURCES+=dstorm.cpp
SOURCES+=segInfo.cpp
SOURCES+=segVecDense.cpp
SOURCES+=segTensor.cpp
SOURCES+=test/testUserIoNet.cpp
SOURCES+=test/prtIoNet.cpp
SOURCES+=test/scalTest.cpp

else # --------------- GPU sources ------------
SOURCES+=dstorm_push.cu
SOURCES+=dstorm.cu
SOURCES+=dstorm.cpp
SOURCES+=dstorm_kernel.cu
SOURCES+=segInfo.cu
SOURCES+=segVecGpu.cu
endif

ifdef MAKECMDGOALS
CMDGOAL:=$(MAKECMDGOALS)
ifeq (x$(CMDGOAL),x)
CMDGOAL:=all
endif
else
CMDGOAL:=all
endif
#
ifeq (,$(filter clean realclean exports,$(CMDGOAL)))
$(info CXX=$(CXX) just before makefile.std-obj-deps)
include ../makefile.std-obj-deps
endif
#
#$(DEPS)/%-pic.d: | deps
#$(DEPS)/ionet/%.d: $(SRC_DIR)/ionet/%.cpp ; $(MAKECXXDEPEND)                    -o - \
#		$< | sed 's!^'`basename $*`'.o:!$(OBJ)/ionet/$*.o:!' > $(DEPS)/$*.d
#$(DEPS)/test/%.d: $(SRC_DIR)/test/%.cpp ; $(MAKECXXDEPEND)                    -o - \
#		$< | sed 's!^'`basename $*`'.o:!$(OBJ)/test/$*.o:!' > $(DEPS)/$*.d
# unfortunately, 'include' directives are NOT processed in order they are encountered.
# They go into an include-list of targets that get made after Makefile is parsed,
# but before any normal targets are made.
# So we make real dependency includes predepend on bogus-predep to generate ENVFILE
# ** before ** make autogenerates dependencies and fails due to 'missing dstorm_env.h'
#  $(patsubst %.cu,$(DEPS)/%.d, $(SOURCES))
$(DEPS)/bogus-predep.d: $(DEPS) env-files
	touch $@
$(patsubst %.cpp,$(DEPS)/%.d, $(filter %.cpp,$(SOURCES))) \
  $(patsubst %.cpp,$(DEPS)/%-pg.d, $(filter %.cpp,$(SOURCES))) \
  $(patsubst %.cpp,$(DEPS)/%.ld,$(filter %.cpp,$(SOURCES))) \
  $(patsubst %.cu,$(DEPS)/%-cu.d, $(SOURCES)) \
  : | $(DEPS)/bogus-predep.d
ifeq (,$(filter clean realclean exports,$(CMDGOAL)))
include $(DEPS)/bogus-predep.d
#include $(patsubst %.cpp,$(DEPS)/%-pic.d,$(STRIP_SOURCE))
include $(patsubst %.cpp,$(DEPS)/%.d, $(filter %.cpp,$(SOURCES)))
include $(patsubst %.cpp,$(DEPS)/%-pg.d, $(filter %.cpp,$(SOURCES)))
include $(patsubst %.cpp,$(DEPS)/%.ld,$(filter %.cpp,$(SOURCES)))
include $(patsubst %.cu,$(DEPS)/%-cu.d,$(filter %.cu,$(SOURCES)))
endif

# LIB_SOURCES is what goes into libdstorm
#	We might want to remove some slow or debug or test stuff
STRIP_SOURCE:=$(filter-out test%,$(STRIP_SOURCE))

doc:
	mkdir -p ../../doc/malt
	doxygen 2>&1 | tee doxygen.log

# hdr-compiles: check that all headers are self-contained and compile without warnings
HDR_COMPILER:=g++
$(OBJ)/%.gch: include/%         | $(OBJ)
	$(HDR_COMPILER) $(CXXFLAGS) -Iinclude -Wall -Werror -DTEST_COMPILE $< -o $@ 
$(OBJ)/%.gch: include/detail/%  | $(OBJ)
	$(HDR_COMPILER) $(CXXFLAGS) -Iinclude -Wall -Werror -DTEST_COMPILE $< -o $@
$(OBJ)/%.gch: include/ionet/%  | $(OBJ)
	$(HDR_COMPILER) $(CXXFLAGS) -Iinclude -Wall -Werror -DTEST_COMPILE $< -o $@
GCH_FILES:=
GCH_FILES+=$(OBJ)/dstorm_common.h.gch
GCH_FILES+=$(OBJ)/dstorm_fwd.hpp.gch
GCH_FILES+=$(OBJ)/dstorm.hpp.gch
GCH_FILES+=$(OBJ)/segVecDense.hpp.gch
GCH_FILES+=$(OBJ)/dstorm_msg.hpp.gch
GCH_FILES+=$(OBJ)/float_ops.hpp.gch
GCH_FILES+=$(OBJ)/liveness.hpp.gch
GCH_FILES+=$(OBJ)/dstorm_msg.hh.gch
GCH_FILES+=$(OBJ)/segInfo.hh.gch
GCH_FILES+=$(OBJ)/segImpl.hh.gch
GCH_FILES+=$(OBJ)/dstorm.hh.gch
GCH_FILES+=$(OBJ)/segVecDense.hh.gch
GCH_FILES+=$(OBJ)/ionet_fwd.hpp.gch
GCH_FILES+=$(OBJ)/scalIoNet.hpp.gch
GCH_FILES+=$(OBJ)/userIoNet.hpp.gch
GCH_FILES+=$(OBJ)/globIoNet.hpp.gch
ifeq ($(WITH_GPU),1)
GCH_FILES+=$(OBJ)/segVecGpu.hpp.gch
endif
hdr-compiles: $(OBJ) $(GCH_FILES)
	@#$(OBJ)/float_ops_omp.hpp.gch
	@rm -f include/*.gch include/detail/*.gch $(filter %.gch,$^)
	@echo "GOOD - standalone header compilation succeeded (.gch files in $(OBJ) removed)"

#$(patsubst %.cu,$(OBJ)/%-cu.ptx,$(filter %.cu,$(STRIP_SOURCE)))
$(LIB)/libdstorm2.a: $(patsubst %.cpp,$(OBJ)/%.o,$(filter %.cpp,$(STRIP_SOURCE))) \
	             $(patsubst %.cu,$(OBJ)/%-cu.o,$(filter %.cu,$(STRIP_SOURCE))) \
		     | dirs
	@echo "----------------"
	@echo "SOURCES          = $(SOURCES)"
	@echo "filtered SOURCES = $(filter %.cpp,$(SOURCES))"
	@echo "STRIP_SOURCE     = $(STRIP_SOURCE)"
	@echo "dependency files $(patsubst %.cpp,$(DEPS)/%.d,$(STRIP_SOURCE))"
	@echo "----------------"
#ifeq ($(WITH_GPU),1)
#	$(AR) rcs $@ $+ $(OBJ)/ionet/userIoNet.o $(OBJ)/ionet/globIoNet.o $(OBJ)/ionet/scalIoNet.o $(OBJ)/segVecGpu-cu.o $(OBJ)/segInfo.o $(OBJ)/dstorm_net.o $(OBJ)/dstorm_fwd.o $(OBJ)/dstorm.o
#else
	$(AR) rcs $@ $+
#endif
	$(STRIP_A)
	@echo "#############-Build $@ DONE."
$(LIB)/libdstorm2-rel.a: $(patsubst %.cpp,$(OBJ)/%-rel.o,$(filter %.cpp,$(STRIP_SOURCE))) \
	                 $(patsubst %.cu,$(OBJ)/%-cu-rel.o,$(filter %.cu,$(STRIP_SOURCE))) \
		         | dirs
	@echo "----------------"
	@echo "SOURCES = $(SOURCES)"
	@echo "filtered SOURCES = $(filter %.cpp,$(SOURCES))"
	@echo "STRIP_SOURCES = $(STRIP_SOURCE)"
	@echo "dependency files $(patsubst %.cpp,$(DEPS)/%.d,$(STRIP_SOURCE))"
	@echo "----------------"
	$(AR) rcs $@ $+
	$(STRIP_A)
	@echo "#############-Build $@ DONE."
$(LIB)/libdstorm2-pg.a: $(patsubst %.cpp,$(OBJ)/%-pg.o,$(filter %.cpp,$(STRIP_SOURCE))) | dirs
	@echo "----------------"
	@echo "SOURCES = $(SOURCES)"
	@echo "filtered SOURCES = $(filter %.cpp,$(SOURCES))"
	@echo "STRIP_SOURCES = $(STRIP_SOURCE)"
	@echo "dependency files $(patsubst %.cpp,$(DEPS)/%-pg.d,$(STRIP_SOURCE))"
	@echo "----------------"
	$(AR) rcs $@ $+
	@$(STRIP_A)
	@echo "#############-Build $@ DONE."
$(LIB)/libdstorm2-pic.a: $(patsubst %.cpp,$(OBJ)/%.lo,$(filter %.cpp,$(STRIP_SOURCE))) \
	                 $(patsubst %.cu,$(OBJ)/%-cu.lo,$(filter %.cu,$(STRIP_SOURCE))) \
		         | dirs
	$(AR) rcs $@ $+
	$(STRIP_A)
	@echo "#############-Build $@ DONE. with -fPIC"

settings2: settings
	@echo "SOURCES = $(SOURCES)"
	@echo "TARGETS = $(TARGETS)"
	@echo " cuda deps = $(patsubst %.cu,$(DEPS)/%-cu.d, $(SOURCES))"

$(RUN_LOGS): | $(TARGETS)
$(TARGETS): | env-files

DSTORM_PRETEST:=
DSTORM_PRETEST+=mk-selftest-0
DSTORM_PRETEST+=check-exports

#mk-selftest-0: | $(TESTLIBS)
#	@{ $(MAKE) -C selftest-0 && echo "YAY, selftest-0 compilation PASSED"; } || { echo "OHOH selftest-0 compilation FAILED"; false; }

# ??? test: | $(TARGETS)
test: $(TESTLIBS) $(TARGETS) $(DSTORM_PRETEST)
	$(MAKE) -j1 $(RUN_LOGS)
	# We should not break Leon's asgd compilation
	@{ $(MAKE) -C Leon-GPL realclean \
		&& $(MAKE) -C Leon-GPL -j8 all \
		&& echo "YAY, Leon's asgd compilation wasn't broken"; } \
		|| { echo "OHOH, we broke Leon-GPL/ compilation"; false; }
	@{ $(MAKE) -C selftest-0 && echo "YAY, selftest-0 compilation PASSED"; } || { echo "OHOH selftest-0 compilation FAILED"; false; }
	@rm -f selftest-0/*.log
	{ $(MAKE) -C selftest-0 -j1 test && echo "YAY, selftest-0 test PASSED"; } || { echo "OHOH selftest-0 test FAILED"; false; }
test-Leon: test | test
	# We should not break Leon's asgd compilation
	@{ $(MAKE) -C Leon-GPL realclean \
		&& $(MAKE) -C Leon-GPL -j8 all \
		&& echo "YAY, Leon's asgd compilation wasn't broken"; } \
		|| { echo "OHOH, we broke Leon-GPL/ compilation"; false; }
	@{ $(MAKE) -C Leon-GPL test && echo "YAY, Leon-GPL 'make test' PASSED"; } || { echo "OHOH Leon-GPL 'make test' FAILED"; false; }

.PRECIOUS: success_or_die.o
success_or_die.o: success_or_die.c success_or_die.h env-files
	$(CC) -O2 -o $@ $(CXXINC) -c $<
#test_utils.o: test_utils.c test_utils.h
#	gcc -O2 -o $@ $(CXXINC) -I. -c $<

# check-exports target should not be necessary.  
env-files: $(ENVFILE) $(DSTORM_ENV_MK) $(DSTORM_ENV_CMAKE)
$(DSTORM_ENV_CMAKE): Makefile
	@{ echo '# Autogenerated by dstorm/Makefile';\
	   echo '#  Records settings used to compile libdstorm';\
	   echo '#  dstorm-related projects should agree with libdstorm compile options';\
	   echo '#  and this file should be included by the boilerplate dStorm.mk file'; \
	   echo 'SET(WITH_LIBORM $(WITH_LIBORM))';\
	   echo 'SET(WITH_MPI $(WITH_MPI))';\
	   echo 'SET(WITH_GPU $(WITH_GPU))';\
	   echo 'SET(LIBDSTORM_FLAGS $(CXXDEF))';\
	} > $@
$(DSTORM_ENV_MK): Makefile
	@{ echo '# Autogenerated by dstorm/Makefile';\
	   echo '#  Records settings used to compile libdstorm';\
	   echo '#  dstorm-related projects should agree with libdstorm compile options';\
	   echo '#  and this file should be included by the boilerplate dStorm.mk file'; \
	   echo 'WITH_LIBORM:=$(WITH_LIBORM)';\
	   echo 'WITH_MPI:=$(WITH_MPI)';\
	   echo 'WITH_GPU:=$(WITH_GPU)';\
	} > $@
$(ENVFILE): Makefile $(DSTORM_ENV_MK)
	@{ echo '#ifndef DSTORM_ENV_H_';\
	   echo '#define DSTORM_ENV_H_';\
	   echo '';\
	   echo '/* This is an AUTO-GENERATED FILE */';\
	   echo '/*   Edit the Makefile instead    */';\
	   echo '';\
	   echo '#define DSTORM_DIR "$(DSTORM_DIR)"';\
	   echo '#define MPI_DIR "$(MPI_DIR)"';\
	   echo '#define GPU_DIR "$(GPU_DIR)"';\
	   echo "#define COMPILE_DATE \"`date +%Y-%m-%d`\"";\
	   echo '#define COMPILE_HOST "$(COMPILE_HOST)"';\
	   echo '#define IBMASTER "$(IBMASTER)"';\
	   echo '#define IBNET "$(IBNET)"';\
	   echo '#ifndef WITH_LIBORM';\
	   echo '#define WITH_LIBORM $(WITH_LIBORM)';\
	   echo '#endif';\
	   echo '#ifndef WITH_MPI';\
	   echo '#define WITH_MPI $(WITH_MPI)';\
	   echo '#endif';\
	   echo '#ifndef WITH_GPU';\
	   echo '#define WITH_GPU $(WITH_GPU)';\
	   echo '#endif';\
	   echo '#ifndef WITH_NOTIFYACK';\
	   echo '#define WITH_NOTIFYACK $(WITH_NOTIFYACK)'; \
	   echo '// see dStorm.mk'; \
	   echo '#endif';\
	   echo '';\
	   echo '#endif // DSTORM_ENV_H_';\
	} > $@
# ? $(INCLUDE_DIR)/dStorm.hpp: $(ENVFILE)

clean:
	rm -f a.out *.o $(TARGETS) $(TESTS) \
		$(RUN_LOGS) test/*.vg $(RUN_LOGS)/*.log \
		dstorm_env.h test*.conf test_utils.c test_utils.h one_dies.c \
		$(ENVFILE) $(OBJ)/*.gch ./*.exe ./core
realclean: clean
	rm -f $(ENVFILE) $(DSTORM_ENV_MK) $(DSTORM_ENV_CMAKE)
	@# remove entirely --- DANGEROUS if they are not "here"
	@#  "." is OK because rm refuses to remove "."
	if [ -d "$(DEPS)" -a "$(DEPS)" != "." ]; then rm -rf "$(DEPS)"; fi
	if [ -d "$(OBJ)"  -a "$(OBJ)"  != "." ]; then rm -rf "$(OBJ)" ; fi
	if [ -d "$(LIB)"  -a "$(LIB)"  != "." ]; then rm -rf "$(LIB)" ; fi
	if [ -d "$(ORM_DIR)"  ]; then $(MAKE) -C $(ORM_DIR) realclean; fi
# last line
