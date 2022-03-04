#include "Mini-app.cuh"

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
  double                   energy         = pow( 10.0, LCG_random_double( &seed ) * 1.3 );
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
  double                   energy         = pow( 10.0, LCG_random_double( &seed ) * 1.3 );
  double                   temperature    = 2.58522e-8;
  int                      hashIndex      = domainHash.index(energy);
  int                      isoIndex       = LCG_random_double(&seed) * numIsotopes;
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
  double                   energy         = pow( 10.0, LCG_random_double( &seed ) * 1.3 );
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
  int matIndex = LCG_random_double(&seed) * numMaterials;
  
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
