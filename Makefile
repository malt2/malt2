TH:=$(shell which th)
SMS:='30 50 60'
CMAKE:=cmake

all: torch

malt: dstorm orm

dstorm:
	@echo "Note: compiling for reduced set of GPU target architectures, SMS=$(SMS)"
	echo "mk-GPU-dstorm.log"
	WITH_GPU=1 SMS=${SMS} $(MAKE) -C dstorm 2>&1 | tee mk-GPU-dstorm.log && echo YAY || OOPS=mk-GPU-dstorm.log

orm:
	echo "Trying mk-GPU-orm.log"
	WITH_GPU=1 $(MAKE) -C orm 2>&1 | tee mk-GPU-orm.log && echo YAY || OOPS=mk-GPU-orm.log

torch: malt
	@if test "$(TH)" = ""; then \
		echo "Torch not found."; \
		echo "Source the torch environment"; \
		false; \
	fi
	$(CMAKE) -C malt2.torch
	$(MAKE) -C malt2.torch
	luarocks install malt2.torch/malt-2-scm-1.rockspec  
	luarocks install dstoptim/dstoptim-scm-1.rockspec 

clean:
	rm -rf malt2.torch/build
	rm -rf dstoptim/build
	$(MAKE) -C dstorm realclean
	$(MAKE) -C orm clean
	$(MAKE) -C torch clean
	rm *.log
	
.PHONY: all dstorm orm torch clean 
