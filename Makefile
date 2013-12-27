VWDIR=$(HOME)/projects/visionworkbench/build
GTEST=$(PWD)/gtest-1.7.0
BDIR=$(HOME)/packages/base_system

CXXFLAGS += -g -Ofast -DNDEBUG -I$(BDIR)/include -I$(BDIR)/include/boost-1_55 -I$(VWDIR)/include -I$(PWD) -ffast-math -Wall -Wno-unused-local-typedefs #-DVW_ENABLE_BOUNDS_CHECK=1

LDFLAGS += -L$(BDIR)/lib -lboost_system-mt-1_55 -lboost_thread-mt-1_55 -lboost_filesystem-mt-1_55 -L$(VWDIR)/lib -lvwCore -lvwMath -lvwFileIO -lvwImage -L$(GTEST)/lib -L$(PWD) -lgtest -lpthread -Wl,-rpath,$(BDIR)/lib -Wl,-rpath,$(VWDIR)/lib

%.o : %.cc
	$(CXX) -c -o $@ $(CXXFLAGS) $^

%.o : %.cxx
	$(CXX) -c -o $@ $(CXXFLAGS) -I$(GTEST)/include $^

EXECS = TestPatchMatch TestMoc

all: $(EXECS)

TestPatchMatch : TestPatchMatch.o PatchMatch.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)

TestMoc : TestMoc.o PatchMatch.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -f *.o *~ $(EXECS) effect*png my*png *.tif *.xml

checkall: $(EXECS)
	./TestPatchMatch
	parallel gdal_translate -b 1 -scale -ot Byte {} {.}.b-H.tif ::: *-D.tif
	parallel gdal_translate -b 2 -scale -ot Byte {} {.}.b-V.tif ::: *-D.tif

check: $(EXECS)
	./TestPatchMatch --gtest_filter=PatchMatch.PatchMatchView
	parallel gdal_translate -b 1 -scale -ot Byte {} {.}.b-H.tif ::: *-D.tif
	parallel gdal_translate -b 2 -scale -ot Byte {} {.}.b-V.tif ::: *-D.tif

profile:
	LD_PRELOAD=/usr/lib/libprofiler.so CPUPROFILE=`pwd`/patchmatch.prof ./TestPatchMatch
	google-pprof --pdf $(PWD)/TestPatchMatch patchmatch.prof > patchmatch.pdf
