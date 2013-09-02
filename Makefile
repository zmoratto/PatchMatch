VWDIR=$(HOME)/projects/VisionWorkbench/build
MACPORTS=$(HOME)/macports
BDIR=$(HOME)/packages/base_system

CXXFLAGS += -g -O3 -I$(BDIR)/include -I$(BDIR)/include/boost-1_54 -I$(VWDIR)/include -I$(PWD)

LDFLAGS += -L$(BDIR)/lib -lboost_system-mt-1_54 -lboost_thread-mt-1_54 -lboost_filesystem-mt-1_54 -L$(VWDIR)/lib -lvwCore -lvwMath -lvwFileIO -lvwImage -L$(MACPORTS)/lib -lgtest

%.o : %.cc
	$(CXX) -c -o $@ $(CXXFLAGS) $^

%.o : %.cxx
	$(CXX) -c -o $@ $(CXXFLAGS) -I$(MACPORTS)/include $^

EXECS = TestSemiGlobalMatching

all: $(EXECS)

TestSemiGlobalMatching : TestSemiGlobalMatching.o SemiGlobalMatching.o
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $^

clean:
	rm -f *.o *~ $(EXECS) effect*png my*png

check: $(EXECS)
	./TestSemiGlobalMatching
