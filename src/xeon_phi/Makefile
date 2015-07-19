#    Copyright 2014-2015 Dake Feng, Peri LLC, dakefeng@gmail.com
#    This file is part of TomograPeri.
#    TomograPeri is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    TomograPeri is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    You should have received a copy of the GNU Lesser General Public License
#    along with TomograPeri.  If not, see <http://www.gnu.org/licenses/>.



CXX      = icpc

# for survey and suitablity, and run
#CXXFLAGS = -gdwarf-3 -V -O3 -fno-inline-functions 
#LNKFLAGS = -ldl

# for correctnexx
#CXXFLAGS = -g -V -O0
#LNKFLAGS = -Bdynamic -ldl
 
# for check vec optimization
#CXXFLAG -qopt-report=3 -qopt-report-phase=vec 

# for run
CXXFLAGS = -O3 -fPIC
LNKFLAGS = -ldl -shared

CCFLAGS = -fPIC
 
LIBS     =
CXXOBJS     = utils_cilk.o pml_quad_cilk.o pml_hybrid_cilk.o ospml_quad_cilk.o ospml_hybrid_cilk.o
CXXSRCS     = utils_cilk.cpp pml_quad_cilk.cpp pml_hybrid_cilk.cpp ospml_quad_cilk.cpp ospml_hybrid_cilk.cpp
CCOBJS     = recon.o 
CCSRCS     = recon.cpp



TARGET   = libtomoperi_phi.so

$(TARGET):	$(CXXOBJS) $(CCOBJS)
	$(CXX) -o $(TARGET) $(CXXOBJS) $(CCOBJS) $(LIBS) $(LNKFLAGS) 

$(CCOBJS):	$(CCSRCS)
	$(CC) -c $(CCFLAGS) $(CCSRCS)

$(CXXOBJS):	$(CXXSRCS)
	$(CXX) -c $(CXXFLAGS) $(CXXSRCS)

all:	$(TARGET)

clean:
	rm -f $(CCOBJS) $(CXXOBJS) $(TARGET)
