
TESTS :=
TESTS += layout-demo
TESTS += layout-optdemo

CXXFLAGS := -std=c++11 -ggdb3 -I.. -I../include

all: $(TESTS) run

run:
	@for t in $(TESTS); do \
	    echo -e "\\n\\n=========== Test $$t ===============\\n"; \
	    $${t}; \
	done

layout-demo: layout-demo.cpp
	$(CXX) -o $@ $(CXXFLAGS) $<
layout-optdemo: layout-demo.cpp
	# now include the optimized impl for layout::ptr, but with debug assertions...	
	$(CXX) -o $@ -DTEST_OPT $(CXXFLAGS) $<

clean:
	rm -f $(TESTS)
