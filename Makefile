VWDIR=$(HOME)/projects/visionworkbench/build
ASPDIR=$(HOME)/projects/StereoPipeline/build
GTEST=$(PWD)/gtest-1.7.0
BDIR=$(HOME)/packages/base_system

CXXFLAGS += -O2 -g -I$(BDIR)/include -I$(BDIR)/include/boost-1_55 -I$(VWDIR)/include -I$(PWD) -I$(ASPDIR)/include -I$(BDIR)/include/eigen3 -Wall -Wno-unused-local-typedefs -DTRILIBRARY #-DDEBUG #-DVW_ENABLE_BOUNDS_CHECK=1 -DDEBUG
CFLAGS += -DTRILIBRARY

LDFLAGS += -L$(BDIR)/lib -lceres -lglog -lboost_system-mt-1_55 -lboost_thread-mt-1_55 -lboost_filesystem-mt-1_55 -lboost_program_options-mt-1_55 -L$(VWDIR)/lib -lvwCore -lvwMath -lvwFileIO -lvwImage -lvwStereo  -L$(ASPDIR)/lib  -L$(GTEST)/lib -L$(PWD) -lpthread -Wl,-rpath,$(BDIR)/lib -Wl,-rpath,$(VWDIR)/lib -Wl,-rpath,$(ASPDIR)/lib

EXECS = fancy_correlate transform_by_disparity testing testing_patchtvmin ground_truth_gen testing_ARAP testing_volumefilter

%.o : %.cc
	$(CXX) -c -o $@ $(CXXFLAGS) $^

all: $(EXECS)

fancy_correlate : fancy_correlate.o PatchMatch2Heise.o SurfaceFitView.o PatchMatch2NCC.o TVMin3.o SurfaceFitWCostView.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)

transform_by_disparity : transform_by_disparity.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)

testing : testing.o PatchMatch2NCC.o SurfaceFitView.o PatchMatch2.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)

testing_patchtvmin : testing_patchtvmin.o TVMin2.o PatchMatch2NCC.o SurfaceFitView.o TVMin3.o PatchMatch2Heise.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)

ground_truth_gen : ground_truth_gen.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)

testing_ARAP : testing_ARAP.o ARAPDataTerm.o ARAPSmoothTerm.o PatchMatch2NCC.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)

testing_volumefilter : testing_volumefilter.cc

clean:
	rm -f *.o *~ $(EXECS) effect*png my*png *.tif *.xml

