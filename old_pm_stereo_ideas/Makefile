VWDIR=$(HOME)/projects/VisionWorkbench/build
ASPDIR=$(HOME)/projects/StereoPipeline/build
GTEST=$(PWD)/gtest-1.7.0
BDIR=$(HOME)/packages/base_system

CXXFLAGS += -g -O1 -I$(BDIR)/include -I$(BDIR)/include/boost-1_55 -I$(VWDIR)/include -I$(PWD) -I$(ASPDIR)/include -ffast-math -Wall -Wno-unused-local-typedefs -DTRILIBRARY #-DVW_ENABLE_BOUNDS_CHECK=1 
CFLAGS += -DTRILIBRARY

LDFLAGS += -L$(BDIR)/lib -lboost_system-mt-1_55 -lboost_thread-mt-1_55 -lboost_filesystem-mt-1_55 -lboost_program_options-mt-1_55 -L$(VWDIR)/lib -lvwCore -lvwMath -lvwFileIO -lvwImage -lvwStereo -lvwInterestPoint -L$(ASPDIR)/lib -laspCore -L$(GTEST)/lib -L$(PWD) -lgtest -lpthread -Wl,-rpath,$(BDIR)/lib -Wl,-rpath,$(VWDIR)/lib -Wl,-rpath,$(ASPDIR)/lib -Wl,-rpath,$(GTEST)/lib

%.o : %.cc
	$(CXX) -c -o $@ $(CXXFLAGS) $^

%.o : %.cxx
	$(CXX) -c -o $@ $(CXXFLAGS) -I$(GTEST)/include $^

EXECS = TestPatchMatch TestMoc patch_match TestPatchMatchView TestPatchMatchHeise render_disparity_guess tricall TestPatchMatchHeiseNormal draw_normals draw_disparity_angle

all: $(EXECS)

tricall : tricall.o triangle.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)

draw_normals : draw_normals.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)

draw_disparity_angle : draw_disparity_angle.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)

render_disparity_guess : render_disparity_guess.o DisparityFromIP.o triangle.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)

patch_match : PatchMatch.o patch_match.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)

TestPatchMatch : TestPatchMatch.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)

TestPatchMatchView : TestPatchMatchView.o PatchMatch.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)

TestPatchMatchHeise : TestPatchMatchHeise.o PatchMatchSimple.o TVMin.o triangle.o DisparityFromIP.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)

TestPatchMatchHeiseNormal : TestPatchMatchHeiseNormal.o PatchMatchSimple.o TVMin.o triangle.o DisparityFromIP.o ScanlineOpt.o
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
