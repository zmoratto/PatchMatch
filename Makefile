VWDIR=$(HOME)/projects/visionworkbench/build
GTEST=$(PWD)/gtest-1.7.0
BDIR=$(HOME)/packages/base_system

CXXFLAGS += -g -O1 -I$(BDIR)/include -I$(BDIR)/include/boost-1_55 -I$(VWDIR)/include -I$(PWD) -ffast-math -Wall -Wno-unused-local-typedefs -DVW_ENABLE_BOUNDS_CHECK=1

LDFLAGS += -L$(BDIR)/lib -lboost_system-mt-1_55 -lboost_thread-mt-1_55 -lboost_filesystem-mt-1_55 -L$(VWDIR)/lib -lvwCore -lvwMath -lvwFileIO -lvwImage -L$(GTEST)/lib -L$(PWD) -lgtest -lpthread -Wl,-rpath,$(BDIR)/lib -Wl,-rpath,$(VWDIR)/lib

%.o : %.cc
	$(CXX) -c -o $@ $(CXXFLAGS) $^

%.o : %.cxx
	$(CXX) -c -o $@ $(CXXFLAGS) -I$(GTEST)/include $^

EXECS = TestPatchMatch

all: $(EXECS)

TestPatchMatch : TestPatchMatch.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -f *.o *~ $(EXECS) effect*png my*png *.tif

check: $(EXECS)
	./TestPatchMatch
	parallel gdal_translate -b 1 -scale -ot Byte {} {.}.b-H.tif ::: lr_?.tif rl_?.tif lr_??.tif rl_??.tif
	parallel gdal_translate -b 2 -scale -ot Byte {} {.}.b-V.tif ::: lr_?.tif rl_?.tif lr_??.tif rl_??.tif
	parallel gdal_translate -b 3 -scale -ot Byte {} {.}.b-Nx.tif ::: lr_?.tif rl_?.tif lr_??.tif rl_??.tif
	parallel gdal_translate -b 4 -scale -ot Byte {} {.}.b-Ny.tif ::: lr_?.tif rl_?.tif lr_??.tif rl_??.tif

profile:
	LD_PRELOAD=/usr/lib/libprofiler.so CPUPROFILE=`pwd`/patchmatch.prof ./TestPatchMatch
	google-pprof --pdf $(PWD)/TestPatchMatch patchmatch.prof > patchmatch.pdf
