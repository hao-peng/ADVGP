GP_SRC = $(wildcard gp/*.cc)
GP = $(patsubst gp/%.cc, gp/%, $(GP_SRC))

# -ltcmalloc_and_profiler
LDFLAGS = -Wl,-rpath,$(DEPS_PATH)/lib $(PS_LDFLAGS_SO) -pthread
gp/% : gp/%.cc build/libps.a
	$(CXX) -std=c++0x $(CFLAGS) -MM -MT gp/$* $< >gp/$*.d
	$(CXX) -std=c++0x $(CFLAGS) -o $@ $(filter %.cc %.a, $^) $(LDFLAGS)

-include gp/*.d
