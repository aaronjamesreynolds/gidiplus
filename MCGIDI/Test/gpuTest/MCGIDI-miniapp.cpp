/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>  
#include <math.h>
#include "MCGIDI.hpp"
#include <sys/time.h>
#include <chrono>
#include <thrust/reduce.h>

#define STARTING_SEED 1070

enum HM_size {large, small};

double get_time()
{

       // If using C++, we can do this:
       unsigned long us_since_epoch = std::chrono::high_resolution_clock::now().time_since_epoch() / std::chrono::microseconds(1);
       return (double) us_since_epoch / 1.0e6;

}


MCGIDI_HOST_DEVICE double myRNG( uint64_t *state );
MCGIDI_HOST_DEVICE double LCG_random_double(uint64_t * seed);
__device__ uint64_t fast_forward_LCG(uint64_t seed, uint64_t n);

/*
=========================================================
*/
__global__ void setUp( int a_numIsotopes, MCGIDI::DataBuffer **a_buf ) {  // Call this each isotope per block and one warp only (i.e. <<< number_isotopes, 32>>>)

    int isotopeIndex = blockIdx.x;

    MCGIDI::DataBuffer *buf = a_buf[isotopeIndex];
    MCGIDI::ProtareSingle *MCProtare = new(buf->m_placementStart) MCGIDI::ProtareSingle( );

    buf->zeroIndexes( );
    buf->m_placement = buf->m_placementStart + sizeof( MCGIDI::ProtareSingle );
    buf->m_maxPlacementSize = sizeof( *a_buf[isotopeIndex] ) + sizeof( MCGIDI::ProtareSingle );

    MCProtare->serialize( *buf, MCGIDI::DataBuffer::Mode::Unpack );                 // This line causes a "nvlink warning".
    buf->m_placement = buf->m_placementStart + sizeof( MCGIDI::ProtareSingle );
}

/*
=========================================================
*/
__device__ int pick_mat(uint64_t * seed) 
{  
  double dist[12];
  dist[0]  = 0.140;       // fuel
  dist[1]  = 0.052;       // cladding
  dist[2]  = 0.275;       // cold, borated water
  dist[3]  = 0.134;       // hot, borated water
  dist[4]  = 0.154;       // RPV
  dist[5]  = 0.064;       // Lower, radial reflector
  dist[6]  = 0.066;       // Upper reflector / top plate
  dist[7]  = 0.055;       // bottom plate
  dist[8]  = 0.008;       // bottom nozzle
  dist[9]  = 0.015;       // top nozzle
  dist[10] = 0.025;       // top of fuel assemblies
  dist[11] = 0.013;       // bottom of fuel assemblies

  double roll = LCG_random_double(seed);

  for( int i = 0; i < 12; i++ )
  {
    double running = 0;
    for( int j = i; j > 0; j-- )
      running += dist[j];
    if( roll < running )
      return i;
  }

  return 0;

}
/*
=========================================================
*/
// Calculate microscopic XSs for a given protare
// 
// Called on blocks only.

__global__ void testMicroXS(MCGIDI::ProtareSingle *MCProtare) 
{       

  // Data used to evaluate XS
  MCGIDI::DomainHash       domainHash(4000, 1e-8, 10);
  int                      collisionIndex = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t                 seed           = STARTING_SEED + collisionIndex;
  double                   energy         = pow( 10.0, myRNG( &seed ) * 1.3 );
  double                   temperature    = 2.58522e-8;
  int                      hashIndex      = domainHash.index(energy);
  MCGIDI::URR_protareInfos urr;

  // Evaluate scattering and total XS
  double scatteringCrossSection = MCProtare->reactionCrossSection(
      0, urr, hashIndex, temperature, energy);
  double totalCrossSection      = MCProtare->crossSection(
      urr, hashIndex, temperature, energy);

  // Print cross sections
  printf("Thread %d Isotope %d - total cross section: %g, scattering cross section: %g\n", 
      collisionIndex, 0, totalCrossSection, scatteringCrossSection);

}

/*
=========================================================
*/
// Calculate microscopic XSs for a random protare
// 
// Called on blocks only.

__global__ void testRandomMicroXSs(char **deviceProtares, int numIsotopes) 
{       

  // Data used to evaluate XS
  MCGIDI::DomainHash       domainHash(4000, 1e-8, 10);
  int                      collisionIndex = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t                 seed           = STARTING_SEED + collisionIndex;
  double                   energy         = pow( 10.0, myRNG( &seed ) * 1.3 );
  double                   temperature    = 2.58522e-8;
  int                      hashIndex      = domainHash.index(energy);
  int                      isoIndex       = myRNG(&seed) * numIsotopes;
  MCGIDI::URR_protareInfos urr;

  MCGIDI::ProtareSingle *MCProtare  = reinterpret_cast<MCGIDI::ProtareSingle *>(deviceProtares[isoIndex]);

  // Evaluate scattering and total XS
  double scatteringCrossSection = MCProtare->reactionCrossSection(
      0, urr, hashIndex, temperature, energy);
  double totalCrossSection      = MCProtare->crossSection(
      urr, hashIndex, temperature, energy);

  // Print cross sections
  printf("Thread %d Isotope %d - total cross section: %g, scattering cross section: %g\n", 
      collisionIndex, isoIndex, totalCrossSection, scatteringCrossSection);

}

/*
=========================================================
*/
// Calculate macroscopic cross sections for a random protare
// 
// Called on blocks only.

__global__ void testRandomMacroXSs(char **deviceProtares, int numIsotopes) 
{       

  // Data used to evaluate XS
  MCGIDI::DomainHash       domainHash(4000, 1e-8, 10);
  int                      collisionIndex = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t                 seed           = STARTING_SEED + collisionIndex;
  double                   energy         = pow( 10.0, myRNG( &seed ) * 1.3 );
  double                   temperature    = 2.58522e-8;
  int                      hashIndex      = domainHash.index(energy);
  MCGIDI::URR_protareInfos urr;

  // Set up dummy material compositions
  int numMaterials = 5;

  // Each entry corresponds to an isoIndex
  int materialComposition[5][10] =
  {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
   {10, 11, 12, 13, -1, -1, -1, -1, -1, -1},
   {14, 15, 16, -1, -1, -1, -1, -1, -1, -1},
   {17, 18, -1, -1, -1, -1, -1, -1, -1, -1},
   {19, -1, -1, -1, -1, -1, -1, -1, -1, -1}};

  // Dummy number densities
  double numberDensities[5][10] =
  {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0}, 
   {10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0},
   {10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0},
   {10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0},
   {10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0}};

  // Sample material
  int matIndex = myRNG(&seed) * numMaterials;
  
  // Initialize accumulators and loop variables
  double scatteringCrossSection = 0;
  double totalCrossSection = 0;
  double numberDensity = -1;
  int isoIndex = -1;

  // Evaluate scattering and total XS
  for (int iConstituent = 0; materialComposition[matIndex][iConstituent] >= 0; iConstituent++)
  {
  
    isoIndex = materialComposition[matIndex][iConstituent];
    numberDensity = numberDensities[matIndex][iConstituent];

    MCGIDI::ProtareSingle *MCProtare  = reinterpret_cast<MCGIDI::ProtareSingle *>(deviceProtares[isoIndex]);

    scatteringCrossSection += numberDensity * MCProtare->reactionCrossSection(
        0, urr, hashIndex, temperature, energy);
    totalCrossSection += numberDensity * MCProtare->crossSection(
        urr, hashIndex, temperature, energy);

  }

  // Print cross sections
  printf("Thread %d Material %d Energy %g - total cross section: %g, scattering cross section: %g\n", 
      collisionIndex, matIndex, energy, totalCrossSection, scatteringCrossSection);

}

/*
=========================================================
*/
// Calculate scatter cross section for a random protare. 
// Material compositions and number densities pre-initialized.
// 

__global__ void calcScatterMacroXSs(
    char   **deviceProtares,
    int    *materialComposition,
    double *numberDensities,
    double *verification,
    int     numMaterials,
    int     maxNumberIsotopes,
    int     numCollisions) 
{       

  int collisionIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if( collisionIndex >= numCollisions ) return;

  // Data used to evaluate XS
  MCGIDI::DomainHash       domainHash(4000, 1e-8, 10);
  uint64_t                 seed           = STARTING_SEED;
  double                   temperature    = 2.58522e-8;
  MCGIDI::URR_protareInfos urr;

  // Fast-forward random number seed
  seed = fast_forward_LCG(seed, 2*collisionIndex);
  
  // Sample material and energy
  double energy    = pow(10.0, LCG_random_double(&seed) * 1.3);
  int    matIndex  = pick_mat(&seed);
  int    hashIndex = domainHash.index(energy);
  
  // Initialize accumulators and loop variables
  double scatteringCrossSection = 0;
  double numberDensity = -1;
  int isoIndex = -1;

  // Evaluate scattering and total XS
  for (int iConstituent = 0; 
       materialComposition[matIndex * maxNumberIsotopes + iConstituent] >= 0 
       && iConstituent < maxNumberIsotopes; 
       iConstituent++)
  {
    isoIndex      = materialComposition[matIndex * maxNumberIsotopes + iConstituent];
    numberDensity = numberDensities[matIndex * maxNumberIsotopes + iConstituent];

    MCGIDI::ProtareSingle *MCProtare  = reinterpret_cast<MCGIDI::ProtareSingle *>(deviceProtares[isoIndex]);

    scatteringCrossSection += numberDensity * MCProtare->reactionCrossSection(
        0, urr, hashIndex, temperature, energy);
  }

  // Calculate verification entry
  verification[collisionIndex] = scatteringCrossSection / numCollisions;
}

/*
=========================================================
*/
// Calculate total cross section for a random protare. 
// Material compositions and number densities pre-initialized.
// 

__global__ void calcTotalMacroXSs(
    char   **deviceProtares,
    int    *materialComposition,
    double *numberDensities,
    double *verification,
    int     numMaterials,
    int     maxNumberIsotopes,
    int     numCollisions) 
{       

  int collisionIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if( collisionIndex >= numCollisions ) return;

  // Data used to evaluate XS
  MCGIDI::DomainHash       domainHash(4000, 1e-8, 10);
  uint64_t                 seed           = STARTING_SEED;
  double                   temperature    = 2.58522e-8;
  MCGIDI::URR_protareInfos urr;

  // Fast-forward random number seed
  seed = fast_forward_LCG(seed, 2*collisionIndex);
  
  // Sample material and energy
  double energy    = pow(10.0, LCG_random_double(&seed) * 1.3);
  int    matIndex  = pick_mat(&seed);
  int    hashIndex = domainHash.index(energy);
  
  // Initialize accumulators and loop variables
  double totalCrossSection = 0;
  double numberDensity = -1;
  int isoIndex = -1;

  // Evaluate scattering and total XS
  for (int iConstituent = 0; 
       materialComposition[matIndex * maxNumberIsotopes + iConstituent] >= 0 
       && iConstituent < maxNumberIsotopes; 
       iConstituent++)
  {
    isoIndex      = materialComposition[matIndex * maxNumberIsotopes + iConstituent];
    numberDensity = numberDensities[matIndex * maxNumberIsotopes + iConstituent];

    MCGIDI::ProtareSingle *MCProtare  = reinterpret_cast<MCGIDI::ProtareSingle *>(deviceProtares[isoIndex]);

    totalCrossSection += numberDensity * MCProtare->crossSection(
        urr, hashIndex, temperature, energy);
  }

  // Calculate verification entry
  verification[collisionIndex] = totalCrossSection / numCollisions;
}

/*
=========================================================
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

/*
=========================================================
*/
std::vector<MCGIDI::Protare *> initMCProtares(int numIsotopes, const char *isotopeNames[])
{

    // Initialize protares and nuclear data maps
    std::vector<MCGIDI::Protare *>protares(numIsotopes);
    std::string mapFilename( "/usr/gapps/Mercury/data/nuclear/endl/2009.3_gp3.17/gnd/all.map" );
    PoPI::Database pops( "/usr/gapps/Mercury/data/nuclear/endl/2009.3/gnd/pops.xml" );
    std::ifstream meta_stream( "/usr/gapps/data/nuclear/development/GIDI3/Versions/V10/metastables_alias.xml" );
    std::string metastable_string( ( std::istreambuf_iterator<char>( meta_stream ) ), 
                                     std::istreambuf_iterator<char>( ) );
    pops.addDatabase( metastable_string, false );
    GIDI::Map::Map map( mapFilename, pops );
    MCGIDI::DomainHash domainHash( 4000, 1e-8, 10 );

    // Timing variables
    timeval tv1, tv2;
    double elapsed_time;

    // For each isotope referenced in isotopeNames, construct a GIDI::protare. Then, initialize a MCGIDI::protare
    // from the GIDI::protare object. 
    gettimeofday( &tv1, nullptr );

    for( int isoIndex = 0; isoIndex < numIsotopes; isoIndex++ ) 
    {
        std::string protareFilename( map.protareFilename( PoPI::IDs::neutron, isotopeNames[isoIndex] ) );

        // Initialize GIDI::protare
        GIDI::Protare                *protare;
        GIDI::Construction::Settings construction( 
            GIDI::Construction::ParseMode::excludeProductMatrices, 
            GIDI::Construction::PhotoMode::nuclearAndAtomic );
        protare = map.protare( construction, pops, PoPI::IDs::neutron, isotopeNames[isoIndex] );

        // Initialize arguments needed by the MCGIDI:protare GIDI-copy constructor
        // Note: only initializing MCGIDI protares with one temperature
        GIDI::Styles::TemperatureInfos temperature = {protare->temperatures()[0]};
        std::string                    label( temperature[0].griddedCrossSection( ) );
        MCGIDI::Transporting::MC       MC( 
            pops, 
            PoPI::IDs::neutron, 
            &protare->styles( ), 
            label, 
            GIDI::Transporting::DelayedNeutrons::on, 
            20.0 );
        GIDI::Transporting::Particles  particleList;
        GIDI::Transporting::MultiGroup continuous_energy_multigroup;
        GIDI::Transporting::Particle   projectile( "n", continuous_energy_multigroup );
        std::set<int>                  exclusionSet;
        particleList.add( projectile );

        // Construct MCGIDI::protare from GIDI::protare
        protares[isoIndex] = MCGIDI::protareFromGIDIProtare(
            *protare, 
            pops, 
            MC, 
            particleList, 
            domainHash, 
            temperature, 
            exclusionSet);

    }
    gettimeofday(&tv2, nullptr);
    elapsed_time = ((tv2.tv_usec - tv1.tv_usec) / 100000.0) + (tv2.tv_sec - tv1.tv_sec);

    printf("Initialized %lu MCGIDI protares in %f seconds.\n", numIsotopes, elapsed_time);

    return protares;

}

/*
=========================================================
*/

void printReactionData(std::vector<MCGIDI::Protare *> protares)
{

    int numIsotopes = protares.size();

    // For the each  protare, print out the possible reactions and their thresholds
    for( int isoIndex = 0; isoIndex < numIsotopes; isoIndex++ ) 
    {

      MCGIDI::Protare *MCProtare = protares[isoIndex];
      int numberOfReactions = MCProtare->numberOfReactions( );
      MCGIDI::Sampling::Input input( true, MCGIDI::Sampling::Upscatter::Model::B );
      MCGIDI::Sampling::MCGIDIVectorProductHandler products;

      for( int iReaction = 0; iReaction < numberOfReactions; ++iReaction ) 
      {
        MCGIDI::Reaction const *reaction = MCProtare->reaction( iReaction );
        double                 threshold = MCProtare->threshold( iReaction );

        printf( "HO: reaction(%d) = %s threshold = %g ENDF_MT = %d\n" , 
            iReaction, reaction->label( ).c_str( ), threshold, reaction->ENDF_MT());
      }
    }

}

/*
=========================================================
*/

std::vector<char *> copyProtaresFromHostToDevice(std::vector<MCGIDI::Protare *> protares)
{

    int numIsotopes = protares.size();

    // Build data buffer to copy host MCGIDI::protares to device
    std::vector<MCGIDI::DataBuffer *>deviceBuffers_h( numIsotopes );
    std::vector<char *>deviceProtares( numIsotopes );
    for( int isoIndex = 0; isoIndex < numIsotopes; isoIndex++ ) 
    {
        MCGIDI::DataBuffer buf_h;

        protares[isoIndex]->serialize( buf_h, MCGIDI::DataBuffer::Mode::Count );

        buf_h.allocateBuffers( );
        buf_h.zeroIndexes( );
        protares[isoIndex]->serialize( buf_h, MCGIDI::DataBuffer::Mode::Pack );

        size_t cpuSize = protares[isoIndex]->memorySize( );
        deviceBuffers_h[isoIndex] = buf_h.copyToDevice( cpuSize, deviceProtares[isoIndex] );
    }

    // Copy data buffer from host to device
    MCGIDI::DataBuffer **deviceBuffers_d = nullptr;
    cudaMalloc( (void **) &deviceBuffers_d, sizeof( MCGIDI::DataBuffer * ) * numIsotopes );
    cudaMemcpy( deviceBuffers_d, &deviceBuffers_h[0], sizeof( MCGIDI::DataBuffer * ) * numIsotopes, cudaMemcpyHostToDevice );
    
    printf("Copied %lu buffered MCGIDI protares from host to device.\n", numIsotopes);

    setUp<<< numIsotopes, 32 >>>( numIsotopes, deviceBuffers_d );

    gpuErrchk( cudaPeekAtLastError( ) );
    gpuErrchk( cudaDeviceSynchronize( ) );

    return deviceProtares;

}

/*
=========================================================
*/
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

/*
=========================================================
*/
std::vector< std::vector<int> > initMaterialCompositions(HM_size size)
{

  std::vector<int> numNucs;

  // Depending on size, define the number of nuclides in each of the 12 materials
  switch(size)
  {
    case large:
      numNucs.insert(numNucs.end(), 
          {321, 5, 4, 4, 26, 21, 21, 21, 21, 21, 9, 9});
      break;
    case small:
      numNucs.insert(numNucs.end(), 
          {34, 5, 4, 4, 26, 21, 21, 21, 21, 21, 9, 9});
      break;
  }

  int numMats    = numNucs.size();
  int maxNumNucs = numNucs[0];

  // Initialize array containing material compositions with -1's
  std::vector< std::vector<int> > materialCompositions(numMats, std::vector<int>(maxNumNucs, -1));
  
  // Assign material compositions with indices identical to those XSBench
  materialCompositions[0] =  { 58, 59, 60, 61, 40, 42, 43, 44, 45, 46, 1, 2, 3, 7,
    8, 9, 10, 29, 57, 47, 48, 0, 62, 15, 33, 34, 52, 53, 
    54, 55, 56, 18, 23, 41 }; // fuel base composition
  for( int iNuc = materialCompositions[0].size(); iNuc < maxNumNucs; iNuc++ )
    materialCompositions[0].push_back(34 + iNuc); // nuclides in large problem variant

  // Non-fuel materials
  materialCompositions[1] =  { 63, 64, 65, 66, 67 }; // cladding
  materialCompositions[2] =  { 24, 41, 4, 5 }; // cold borated water
  materialCompositions[3] =  { 24, 41, 4, 5 }; // hot borated water
  materialCompositions[4] =  { 19, 20, 21, 22, 35, 36, 37, 38, 39, 25, 27, 28, 29,
    30, 31, 32, 26, 49, 50, 51, 11, 12, 13, 14, 6, 16,
    17 }; // RPV
  materialCompositions[5] =  { 24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25,
    49, 50, 51, 11, 12, 13, 14 }; // lower radial reflector
  materialCompositions[6] =  { 24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25,
    49, 50, 51, 11, 12, 13, 14 }; // top reflector / plate
  materialCompositions[7] =  { 24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25,
    49, 50, 51, 11, 12, 13, 14 }; // bottom plate
  materialCompositions[8] =  { 24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25,
    49, 50, 51, 11, 12, 13, 14 }; // bottom nozzle
  materialCompositions[9] =  { 24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25,
    49, 50, 51, 11, 12, 13, 14 }; // top nozzle
  materialCompositions[10] = { 24, 41, 4, 5, 63, 64, 65, 66, 67 }; // top of FA's
  materialCompositions[11] = { 24, 41, 4, 5, 63, 64, 65, 66, 67 }; // bottom FA's

  // Pad material compositions
   for (int iMat = 0; iMat < numMats; iMat++)
  {
    for (int iNuc = materialCompositions[iMat].size(); materialCompositions[iMat].size() < maxNumNucs; iNuc++)
    {
      materialCompositions[iMat].push_back(-1);
    }
  }

  //for (int iMat = 0; iMat < numMats; iMat++)
  //{
  //  for (int iNuc = 0; iNuc < maxNumNucs; iNuc++)
  //  {
  //    std::cout << materialCompositions.at(iMat).at(iNuc) << " ";
  //  }
  //  std::cout << std::endl;
  //}

  std::cout << "Initialized materials." << std::endl;

  return materialCompositions;
}

/*
=========================================================
*/
std::vector< std::vector<double> > initNumberDensities(
    std::vector< std::vector<int> > materialCompositions)
{

  int numMats    = materialCompositions.size();
  int maxNumNucs = materialCompositions[0].size();

  // Initialize array containing number densities with 10.0's
  std::vector< std::vector<double> > numberDensities(numMats, std::vector<double>(maxNumNucs, 10.0));
  
  return numberDensities;
}


/*
=========================================================
*/
int main( int argc, char **argv ) 
{

    // Default options for command line arguments
    int doPrint       = 0;                    // doPrint == 0 means do not print out results from unpacked data
    int numCollisions = 100 * 1000;           // Number of sample reactions
    int numIsotopes   = 1;                    // Number of isotopes
    int numBatches    = 1;                    // Number of batches
    int numThreads    = 256;                  // Number of threads in kernel launch
    int doCompare     = 0;                    // Compare the bytes of gidi data. 0 - no comparison, 1 - no compare, 
                                             // write out data, 2 - Read in data and compare
    HM_size problem_size = small;                                             
 
    // Initialize vector containing isotope names
    const char *isotopeNames[] = {
            "U235",  "H1",    "U238",  "He4",   "Pu239", "C12",   "Pb209", "Hg200", "W185", "Gd156",
            "Sm148", "Nd145", "Cs135", "Xe128", "As73",  "Zn69",  "Br80",  "Fe56",  "Cr51", "Sc46",
            "Ar40",  "Al27",  "O16",   "Li6",   "Li7",   "Be9",   "H3",    "He3",   "Na23", "Mg25",
            "Am242", "Cf250", "Np238", "Er166", "Pm147", "Ce142", "Ba133", "Xe136", "I126", "Cd106",
            "Mo95",  "Zr90",  "Kr80",  "Sr84",  "Ni62",  "Co58",  "V51",   "Ca44",  "Ti45", "Bk248",
            "U235",  "H1",    "U238",  "He4",   "Pu239", "C12",   "Pb209", "Hg200", "W185", "Gd156",
            "Sm148", "Nd145", "Cs135", "Xe128", "As73",  "Zn69",  "Br80",  "Fe56",  "Cr51", "Sc46",
            "Ar40",  "Al27",  "O16",   "Li6",   "Li7",   "Be9",   "H3",    "He3",   "Na23", "Mg25",
            "Am242", "Cf250", "Np238", "Er166", "Pm147", "Ce142", "Ba133", "Xe136", "I126", "Cd106",
            "Mo95",  "Zr90",  "Kr80",  "Sr84",  "Ni62",  "Co58",  "V51",   "Ca44",  "Ti45", "Bk248", 
            "U235",  "H1",    "U238",  "He4",   "Pu239", "C12",   "Pb209", "Hg200", "W185", "Gd156",
            "Sm148", "Nd145", "Cs135", "Xe128", "As73",  "Zn69",  "Br80",  "Fe56",  "Cr51", "Sc46",
            "Ar40",  "Al27",  "O16",   "Li6",   "Li7",   "Be9",   "H3",    "He3",   "Na23", "Mg25",
            "Am242", "Cf250", "Np238", "Er166", "Pm147", "Ce142", "Ba133", "Xe136", "I126", "Cd106",
            "Mo95",  "Zr90",  "Kr80",  "Sr84",  "Ni62",  "Co58",  "V51",   "Ca44",  "Ti45", "Bk248",
            "U235",  "H1",    "U238",  "He4",   "Pu239", "C12",   "Pb209", "Hg200", "W185", "Gd156",
            "Sm148", "Nd145", "Cs135", "Xe128", "As73",  "Zn69",  "Br80",  "Fe56",  "Cr51", "Sc46",
            "Ar40",  "Al27",  "O16",   "Li6",   "Li7",   "Be9",   "H3",    "He3",   "Na23", "Mg25",
            "Am242", "Cf250", "Np238", "Er166", "Pm147", "Ce142", "Ba133", "Xe136", "I126", "Cd106",
            "Mo95",  "Zr90",  "Kr80",  "Sr84",  "Ni62",  "Co58",  "V51",   "Ca44",  "Ti45", "Bk248",
            "U235",  "H1",    "U238",  "He4",   "Pu239", "C12",   "Pb209", "Hg200", "W185", "Gd156",
            "Sm148", "Nd145", "Cs135", "Xe128", "As73",  "Zn69",  "Br80",  "Fe56",  "Cr51", "Sc46",
            "Ar40",  "Al27",  "O16",   "Li6",   "Li7",   "Be9",   "H3",    "He3",   "Na23", "Mg25",
            "Am242", "Cf250", "Np238", "Er166", "Pm147", "Ce142", "Ba133", "Xe136", "I126", "Cd106",
            "Mo95",  "Zr90",  "Kr80",  "Sr84",  "Ni62",  "Co58",  "V51",   "Ca44",  "Ti45", "Bk248",
            "U235",  "H1",    "U238",  "He4",   "Pu239", "C12",   "Pb209", "Hg200", "W185", "Gd156",
            "Sm148", "Nd145", "Cs135", "Xe128", "As73",  "Zn69",  "Br80",  "Fe56",  "Cr51", "Sc46",
            "Ar40",  "Al27",  "O16",   "Li6",   "Li7",   "Be9",   "H3",    "He3",   "Na23", "Mg25",
            "Am242", "Cf250", "Np238", "Er166", "Pm147", "Ce142", "Ba133", "Xe136", "I126", "Cd106",
            "Mo95",  "Zr90",  "Kr80",  "Sr84",  "Ni62",  "Co58",  "V51",   "Ca44",  "Ti45", "Bk248",
            "U235",  "H1",    "U238",  "He4",   "Pu239", "C12",   "Pb209", "Hg200", "W185", "Gd156",
            "Sm148", "Nd145", "Cs135", "Xe128", "As73",  "Zn69",  "Br80",  "Fe56",  "Cr51", "Sc46",
            "Ar40",  "Al27",  "O16",   "Li6",   "Li7",   "Be9",   "H3",    "He3",   "Na23", "Mg25",
            "Am242", "Cf250", "Np238", "Er166", "Pm147", "Ce142", "Ba133", "Xe136", "I126", "Cd106",
            "Mo95",  "Zr90",  "Kr80",  "Sr84",  "Ni62",  "Co58",  "V51",   "Ca44",  "Ti45", "Bk248",
            "U235",  "H1",    "U238",  "He4",   "Pu239", "C12",   "Pb209", "Hg200", "W185", "Gd156",
            "Sm148", "Nd145", "Cs135", "Xe128", "As73",  "Zn69",  "Br80",  "Fe56",  "Cr51", "Sc46",
            "Ar40",  "Al27",  "O16",   "Li6",   "Li7",   "Be9",   "H3",    "He3",   "Na23", "Mg25",
            "Am242", "Cf250", "Np238", "Er166", "Pm147", "Ce142", "Ba133", "Xe136", "I126", "Cd106",
            "Mo95",  "Zr90",  "Kr80",  "Sr84",  "Ni62",  "Co58",  "V51",   "Ca44",  "Ti45", "Bk248",
            "U235",  "H1",    "U238",  "He4",   "Pu239", "C12",   "Pb209", "Hg200", "W185", "Gd156",
            "Sm148", "Nd145", "Cs135", "Xe128", "As73",  "Zn69",  "Br80",  "Fe56",  "Cr51", "Sc46",
            "Ar40",  "Al27",  "O16",   "Li6",   "Li7",   "Be9",   "H3",    "He3",   "Na23", "Mg25",
            "Am242", "Cf250", "Np238", "Er166", "Pm147", "Ce142", "Ba133", "Xe136", "I126", "Cd106",
            "Mo95",  "Zr90",  "Kr80",  "Sr84",  "Ni62",  "Co58",  "V51",   "Ca44",  "Ti45", "Bk248",
            "U235",  "H1",    "U238",  "He4",   "Pu239", "C12",   "Pb209", "Hg200", "W185", "Gd156",
            "Sm148", "Nd145", "Cs135", "Xe128", "As73",  "Zn69",  "Br80",  "Fe56",  "Cr51", "Sc46",
            "Ar40",  "Al27",  "O16",   "Li6",   "Li7",   "Be9",   "H3",    "He3",   "Na23", "Mg25",
            "Am242", "Cf250", "Np238", "Er166", "Pm147", "Ce142", "Ba133", "Xe136", "I126", "Cd106",
            "Mo95",  "Zr90",  "Kr80",  "Sr84",  "Ni62",  "Co58",  "V51",   "Ca44",  "Ti45", "Bk248"};
    int numberOfIsotopes = sizeof( isotopeNames ) / sizeof( isotopeNames[0] );

    // Read in command line arguments
    if( argc > 1 ) doPrint = atoi( argv[1] );
    if( argc > 2 ) numCollisions = atol( argv[2] );
    if( argc > 3 ) numIsotopes = atoi( argv[3] );
    if( numIsotopes > numberOfIsotopes ) numIsotopes = numberOfIsotopes;
    if( argc > 4 ) numBatches  = atoi( argv[4] );
    if( argc > 5 ) numThreads  = atoi( argv[5] );
    if( argc > 6 ) doCompare = atoi( argv[6] );
    if( argc > 7 )
    {
      char problem_size_arg = *(argv[7]);
      if (std::tolower(problem_size_arg) == 'l')
      {
        problem_size = large;
        numIsotopes  = 355; 
      }
      else if (std::tolower(problem_size_arg) == 's')
      {
        problem_size = small;
        numIsotopes  = 68; 
      }
    }
    
    // Set up material compositions and number densities
    std::vector<std::vector<int>> materialCompositions2D = initMaterialCompositions(problem_size);
    std::vector<std::vector<double>> numberDensities2D = initNumberDensities(materialCompositions2D);

    int    numMats = materialCompositions2D.size(), 
           maxNumIsotopes = materialCompositions2D[0].size(),
           numEntries = numMats * maxNumIsotopes;
    int    *materialCompositions;
    double *numberDensities;
    double *verification;

    int deviceId;
    cudaGetDevice(&deviceId);

    // Allocate memory for material data
    size_t sizeMatComp      = numEntries * sizeof(int);
    size_t sizeNumDens      = numEntries * sizeof(double);
    size_t sizeVerification = numCollisions * sizeof(double);
    cudaMallocManaged(&materialCompositions, sizeMatComp);
    cudaMallocManaged(&numberDensities,      sizeNumDens); 
    cudaMallocManaged(&verification,         sizeVerification); 

    // Initialize 1D material composition and number density vectors
    unwrapFrom2Dto1D(materialCompositions2D, materialCompositions, numMats, maxNumIsotopes);
    unwrapFrom2Dto1D(numberDensities2D, numberDensities, numMats, maxNumIsotopes);

    // Copy material compositions and number densities to device
    cudaMemPrefetchAsync(materialCompositions, sizeMatComp,      deviceId);
    cudaMemPrefetchAsync(numberDensities,      sizeNumDens,      deviceId);
    cudaMemPrefetchAsync(verification,         sizeVerification, deviceId);

    // Print runtime options
    printf( "doPrint = %d, numCollisions = %g, numIsotopes = %d, numBatches = %d , numThreads = %d, doCompare = %d\n", doPrint, static_cast<double>( numCollisions ), numIsotopes, numBatches, numThreads, doCompare);

    // Set and verify CUDA limits
    // These options were in gpuTest. If I use them, I run out of device memory, so I'm not using them.
    //setCudaOptions();

    // Initialize protares and nuclear data maps
    std::vector<MCGIDI::Protare *> protares = initMCProtares(numIsotopes, isotopeNames);

    // Print reaction data
    if (doPrint) printReactionData(protares);
  
    // Serialize protares, then copy them from host to device
    std::vector<char *> deviceProtares = copyProtaresFromHostToDevice(protares);

    if( doPrint ) 
    {
      // Sample and print microscopic cross sections from the last isotope initialized
        testMicroXS<<<1, 10>>>( reinterpret_cast<MCGIDI::ProtareSingle *>( deviceProtares[numIsotopes-1] ) );
        gpuErrchk( cudaPeekAtLastError( ) );
        gpuErrchk( cudaDeviceSynchronize( ) );
        
        // Sample and print microscopic cross sections randomly from number of isotopes initialized
        testRandomMicroXSs<<<1, 10>>>(&deviceProtares[0], numIsotopes);
        gpuErrchk( cudaPeekAtLastError( ) );
        gpuErrchk( cudaDeviceSynchronize( ) );
        
        // Sample and print microscopic cross sections randomly from number of isotopes initialized
        testRandomMacroXSs<<<1, 10>>>(&deviceProtares[0], numIsotopes);
        gpuErrchk( cudaPeekAtLastError( ) );
        gpuErrchk( cudaDeviceSynchronize( ) );
    }
    
    // Calculate number of blocks in execution configuration
    int numBlocks = (numCollisions + numThreads - 1) / numThreads;

    // Launch and time macroscopic total XS sampling kernel 
    double startTime = get_time();
    for (int iBatch = 0; iBatch < numBatches; iBatch++)
    {
      calcTotalMacroXSs<<<numBlocks, numThreads>>>(
          &deviceProtares[0], 
          materialCompositions, 
          numberDensities,
          verification,
          numMats,
          maxNumIsotopes,
          numCollisions);
      gpuErrchk( cudaPeekAtLastError( ) );
      gpuErrchk( cudaDeviceSynchronize( ) );
    }
    double endTime  = get_time();

    // Calculate verification hash
    double verification_hash = thrust::reduce(
        thrust::device, 
        verification, 
        verification + numCollisions);
    gpuErrchk( cudaPeekAtLastError( ) );
    gpuErrchk( cudaDeviceSynchronize( ) );
    
    printf("Total XS verification hash: %f\n", verification_hash);

    // Get XS calculation rate
    double elapsedTime = endTime - startTime;
    double xs_rate = (double) numBatches * numCollisions / elapsedTime;

    // Print out look-up rate
    printf("Looked up %d * %g XSs in %g seconds \n", numBatches, static_cast<double>(numCollisions), elapsedTime);
    printf("Total XS look-up rate: %g cross sections per second \n", xs_rate);

    // Launch and time macroscopic scattering XS sampling kernel 
    startTime = get_time();
    for (int iBatch = 0; iBatch < numBatches; iBatch++)
    {
      calcScatterMacroXSs<<<numBlocks, numThreads>>>(
          &deviceProtares[0], 
          materialCompositions, 
          numberDensities,
          verification,
          numMats,
          maxNumIsotopes,
          numCollisions);
      gpuErrchk( cudaPeekAtLastError( ) );
      gpuErrchk( cudaDeviceSynchronize( ) );
    }
    endTime  = get_time();
    
    // Calculate verification hash
    verification_hash = thrust::reduce(
        thrust::device, 
        verification, 
        verification + numCollisions);
    gpuErrchk( cudaPeekAtLastError( ) );
    gpuErrchk( cudaDeviceSynchronize( ) );
    
    printf("Scattering XS verification hash: %f\n", verification_hash);

    // Get XS calculation rate
    elapsedTime = endTime - startTime;
    xs_rate = (double) numBatches * numCollisions / elapsedTime;

    // Print out look-up rate
    printf("Looked up %d * %g XSs in %g seconds \n", numBatches, static_cast<double>(numCollisions), elapsedTime);
    printf("Scatter XS look-up rate: %g cross sections per second \n", xs_rate);


    return( EXIT_SUCCESS );
}

/*
=========================================================
*/
MCGIDI_HOST_DEVICE double myRNG( uint64_t *seed ) {

   *seed = 2862933555777941757ULL * ( *seed ) + 3037000493ULL;      // Update state from the previous value.
   
   return 5.42101086242752157e-20 * ( *seed );                      // Map state from [0,2**64) to double [0,1).
}

/*
=========================================================
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
=========================================================
*/
__device__ uint64_t fast_forward_LCG(uint64_t seed, uint64_t n)
{
	// LCG parameters
	const uint64_t m = 9223372036854775808ULL; // 2^63
	uint64_t a = 2806196910506780709ULL;
	uint64_t c = 1ULL;

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

