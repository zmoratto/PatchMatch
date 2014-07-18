VWDIR=$(HOME)/projects/VisionWorkbench/build
ASPDIR=$(HOME)/projects/StereoPipeline/build
GTEST=$(PWD)/gtest-1.7.0
BDIR=$(HOME)/packages/base_system

CXXFLAGS += -g -O3 -I$(BDIR)/include -I$(BDIR)/include/boost-1_55 -I$(VWDIR)/include -I$(PWD) -I$(ASPDIR)/include -Wall -Wno-unused-local-typedefs -DTRILIBRARY -I$(BDIR)/include/eigen3 #-DVW_ENABLE_BOUNDS_CHECK=1
CFLAGS += -DTRILIBRARY

LDFLAGS += -L$(BDIR)/lib -lceres -lglog -lboost_system-mt-1_55 -lboost_thread-mt-1_55 -lboost_filesystem-mt-1_55 -lboost_program_options-mt-1_55 -L$(VWDIR)/lib -lvwCore -lvwMath -lvwFileIO -lvwImage -lvwStereo  -L$(ASPDIR)/lib  -L$(GTEST)/lib -L$(PWD) -lpthread -Wl,-rpath,$(BDIR)/lib -Wl,-rpath,$(VWDIR)/lib

%.o : %.cc
	$(CXX) -c -o $@ $(CXXFLAGS) $^

%.o : %.cxx
	$(CXX) -c -o $@ $(CXXFLAGS) -I$(GTEST)/include $^

EXECS = TestSurfaceFit

all: $(EXECS)

TestSurfaceFit : main.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -f *.o *~ $(EXECS) effect*png my*png *.tif *.xml

