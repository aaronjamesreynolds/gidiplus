/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

/*
Brief: Material initialization
Author: Aaron James Reynolds 
(RNG functions credited to XSBench)
*/

#include "Mini-app.cuh"

/*
===============================================================================
Sample a random number (Taken from XSBench)
*/
MCGIDI_HOST_DEVICE double LCG_random_double(uint64_t * seed)
{
  // LCG parameters
  const uint64_t m = 9223372036854775808ULL; // 2^63
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return (double) (*seed) / (double) m;
}	

/*
===============================================================================
Fast forward random number generator (Taken from XSBench)
*/
MCGIDI_HOST_DEVICE uint64_t fast_forward_LCG(uint64_t seed, uint64_t n)
{
  // LCG parameters
  const uint64_t m = 9223372036854775808ULL; // 2^63
  uint64_t       a = 2806196910506780709ULL;
  uint64_t       c = 1ULL;

  n = n % m;

  uint64_t a_new = 1;
  uint64_t c_new = 0;

  while(n > 0) 
  {
    if(n & 1)
    {
      a_new *= a;
      c_new = c_new * a + c;
    }
    c *= (a + 1);
    a *= a;

    n >>= 1;
  }

  return (a_new * seed + c_new) % m;
}

/*
===============================================================================
Check if two numbers are equal within a tolerance
*/
bool approximatelyEqual(double a, double b, double epsilon)
{
  return (double) fabs(a - b) <= ( ((double) fabs(a) > (double) fabs(b) ? (double) fabs(b) : (double) fabs(a)) * epsilon);
}

/*
===============================================================================
Element-wise check if two vectors are euqal within a tolerance
*/
bool approximatelyEqual(double *a,  double *b, int size, double epsilon)
{

  bool vectorsEqual = true;

  for (int i = 0; i < size; i++)
  {
    vectorsEqual = vectorsEqual and approximatelyEqual(a[i], b[i], epsilon);
  }

  return vectorsEqual;
}

/*
===============================================================================
Get current time
*/
double get_time()
{

  unsigned long us_since_epoch = 
    std::chrono::high_resolution_clock::now().time_since_epoch() 
    / std::chrono::microseconds(1);
  return (double) us_since_epoch / 1.0e6;

}

/*
===============================================================================
Generate LookupRate object from XS lookup timing and sampling parameters
*/
LookupRate_t calcLookupRate(
    std::vector<double> edgeTimes, 
    int                 numLookups,
    int                 numBatches,
    std::string         tag)
{
  
  std::vector<double> runtimes;
  LookupRate_t        lookupRate;
  double              variance = 0;

  lookupRate.totalTime = edgeTimes.back() - edgeTimes[0];

  for (int iTime = 1; iTime < edgeTimes.size(); iTime++)
  {
    runtimes.push_back(edgeTimes[iTime] - edgeTimes[iTime - 1]);
    lookupRate.meanTime += runtimes.back();
  }

  lookupRate.meanTime   = lookupRate.meanTime / runtimes.size();
  lookupRate.lookupRate = (double) numLookups / lookupRate.meanTime;

  for (int iTime = 0; iTime < runtimes.size(); iTime++)
  {
    variance += (runtimes[iTime] - lookupRate.meanTime) 
    * (runtimes[iTime] - lookupRate.meanTime);
  }

  variance = variance / runtimes.size();
  lookupRate.meanTimeStdDev = sqrt(variance);

  lookupRate.tag        = tag;
  lookupRate.numLookups = numLookups;
  lookupRate.numBatches = numBatches;

  return lookupRate;

}

/*
===============================================================================
Set CUDA options that were present in MCGIDI/Test/gpuTest/gpuTest.cpp
*/
void setCudaOptions()
{

  // Set and verify CUDA limits
  size_t my_size;
  cudaDeviceSetLimit( cudaLimitStackSize, 80 * 1024 );
  cudaDeviceGetLimit( &my_size, cudaLimitStackSize ) ;
  printf( "cudaLimitStackSize =  %luk\n", my_size / 1024 );
  cudaDeviceSetLimit( cudaLimitMallocHeapSize, 100 * 1024 * 1024 );
  cudaDeviceGetLimit( &my_size, cudaLimitMallocHeapSize ) ;
  printf( "cudaLimitMallocHeapSize =  %luM\n", my_size / ( 1024 * 1024 ) );
  cudaDeviceSetLimit( cudaLimitPrintfFifoSize, 40 * 1024 * 1024 );
  cudaDeviceGetLimit( &my_size, cudaLimitPrintfFifoSize );
  printf( "cudaLimitPrintfFifoSize =  %luM\n", my_size / ( 1024 * 1024 ) );

}
