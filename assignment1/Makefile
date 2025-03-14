CXX       = g++ -std=c++17

OPTFLAGS  = -O3  
AVXFLAGS  = -mavx -mfma 
AUTOFLAGS = -march=native 


CXXFLAGS  = -Wall 
INCLUDES  = -I. -I./include 
LIBS      = #-pthread -fopenmp 


SOURCES   = $(wildcard *.cpp)
TARGETS   = $(SOURCES:.cpp=)

.PHONY: all clean cleanall


%: %.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTFLAGS) -o $@ $< $(LIBS)

# For files with _avx in their name, append flags to CXXFLAGS
%_avx: CXXFLAGS += $(AVXFLAGS)

# For files with _auto in their name, append flags to CXXFLAGS
%_auto: CXXFLAGS += $(AUTOFLAGS)


all: $(TARGETS)


clean: 
	-rm -f *.o *~

cleanall: clean
	-rm -f $(TARGETS)