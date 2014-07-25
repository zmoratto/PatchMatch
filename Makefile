VWDIR=$(HOME)/projects/VisionWorkbench/build
ASPDIR=$(HOME)/projects/StereoPipeline/build
GTEST=$(PWD)/gtest-1.7.0
BDIR=$(HOME)/packages/base_system

CXXFLAGS += -O2 -g -I$(BDIR)/include -I$(BDIR)/include/boost-1_55 -I$(VWDIR)/include -I$(PWD) -I$(ASPDIR)/include -I$(BDIR)/include/eigen3 -Wall -Wno-unused-local-typedefs -DTRILIBRARY #-DDISABLE_RL #-DVW_ENABLE_BOUNDS_CHECK=1 -DDEBUG
CFLAGS += -DTRILIBRARY

LDFLAGS += -L$(BDIR)/lib -lceres -lboost_system-mt-1_55 -lboost_thread-mt-1_55 -lboost_filesystem-mt-1_55 -lboost_program_options-mt-1_55 -L$(VWDIR)/lib -lvwCore -lvwMath -lvwFileIO -lvwImage -lvwStereo  -L$(ASPDIR)/lib -laspCore  -L$(GTEST)/lib -L$(PWD) -lpthread -Wl,-rpath,$(BDIR)/lib -Wl,-rpath,$(VWDIR)/lib -Wl,-rpath,$(ASPDIR)/lib

EXECS = fancy_correlate transform_by_disparity testing

%.o : %.cc
	$(CXX) -c -o $@ $(CXXFLAGS) $^

all: $(EXECS)

fancy_correlate : fancy_correlate.o PatchMatch2.o SurfaceFitView.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)

transform_by_disparity : transform_by_disparity.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)

testing : testing.o PatchMatch2.o SurfaceFitView.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -f *.o *~ $(EXECS) effect*png my*png *.tif *.xml

