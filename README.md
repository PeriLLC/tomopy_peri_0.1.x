# tomopy_peri_0.1.x
The hardware accellerated implementaion for tomopy reconstruction algorithms.

The support algorithms and hardware platforms:

1. pml_quad, pml_hybrid, ospml_quad, ospml_hybrid algorithms
2. Intel Xeon Phi Co-processor.

How TO INSTALL:

0. Make sure you have Intel Xeon Phi Co-processor mounted and MPSS, Intel parallel studio installed and configured.

1. Install tomopy, this package is for > 0.1.11 version

2. Install tomopy_peri:

use two commands as belowing to build and install 
>>python setup.py build

>>python setup.py install 

3. Test the package:
Change to test directory:
>>cd test

Test with small data set:
>>python testSmallTestData.py

For large data set, first create the test data set by:
>>python genLargeTestData.py

Test with large data set:
>>python testLargeTestData.py

A example on recon and view result
>>python testRecon.py

How TO USE:

1. No extra code needed. You many just simply use recon_accelerated function to replace recon function in your code.
2. Some extra parameters are for different hardware platform. Currently only Xeon_Phi is supported.


8.2.2015
