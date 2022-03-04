
#ifndef __XSBENCH_HEADER_H__
#define __XSBENCH_HEADER_H__

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>  
#include <math.h>
#include "MCGIDI.hpp"
#include <sys/time.h>
#include <chrono>
#include <thrust/reduce.h>
#include <omp.h>

#define STARTING_SEED 1070

enum HM_size {large, small};

template <class myType>
void unwrapFrom2Dto1D(std::vector<std::vector<myType>> a, myType *b, int rows, int cols) 
{

  for (int iRow = 0; iRow < rows; iRow++)
  {
    for (int iCol = 0; iCol < cols; iCol++)
    {
      b[iRow * cols + iCol] = a[iRow][iCol];
    }
  }

}


// XSCalc.cu functions
__global__ void calcScatterMacroXSs(
    char   **deviceProtares,
    int    *materialComposition,
    double *numberDensities,
    double *verification,
    int     maxNumberIsotopes,
    int     numCollisions);
__global__ void calcTotalMacroXSs(
    char   **deviceProtares,
    int    *materialComposition,
    double *numberDensities,
    double *verification,
    int     maxNumberIsotopes,
    int     numCollisions);
int calcScatterMacroXSs(
    std::vector<MCGIDI::Protare *> protares,
    int    *materialComposition,
    double *numberDensities,
    int     maxNumberIsotopes,
    int     numCollisions);
bool calcScatterMacroXSs(
    std::vector<MCGIDI::Protare *> protares,
    int    *materialComposition,
    double *numberDensities,
    double *verification_gpu,
    int     maxNumberIsotopes,
    int     verifyStart,
    int     numVerify);

// ProtareInit.cu functions
std::vector<MCGIDI::Protare *> initMCProtares(
    int         numIsotopes, 
    const char *isotopeNames[]);
void printReactionData(std::vector<MCGIDI::Protare *> protares);
std::vector<char *> copyProtaresFromHostToDevice(
    std::vector<MCGIDI::Protare *> protares);
__global__ void setUp( int a_numIsotopes, MCGIDI::DataBuffer **a_buf );

// Materials.cu functions
MCGIDI_HOST_DEVICE int pick_mat(uint64_t * seed);
std::vector< std::vector<int> > initMaterialCompositions(HM_size size);
std::vector< std::vector<double> > initNumberDensities(
    std::vector< std::vector<int> > materialCompositions);

// Utils.cu functions
MCGIDI_HOST_DEVICE double myRNG( uint64_t *seed );
MCGIDI_HOST_DEVICE double LCG_random_double(uint64_t * seed);
MCGIDI_HOST_DEVICE uint64_t fast_forward_LCG(uint64_t seed, uint64_t n);
bool approximatelyEqual(double a, double b, double epsilon);
double get_time();
void setCudaOptions();

// TestFunc.cu functions
__global__ void testMicroXS(MCGIDI::ProtareSingle *MCProtare);
__global__ void testRandomMicroXSs(char **deviceProtares, int numIsotopes);
__global__ void testRandomMacroXSs(char **deviceProtares, int numIsotopes);
#endif
