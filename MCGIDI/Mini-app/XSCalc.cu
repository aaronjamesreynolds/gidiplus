#include "Mini-app.cuh"

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
  verification[collisionIndex] = scatteringCrossSection;
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
  verification[collisionIndex] = totalCrossSection;
}

bool calcScatterMacroXSs(
    std::vector<MCGIDI::Protare *> protares,
    int    *materialComposition,
    double *numberDensities,
    double *verification_gpu,
    int     maxNumberIsotopes,
    int     verifyStart,
    int     numVerify) 
{       

  // Data used to evaluate XS
  MCGIDI::DomainHash       domainHash(4000, 1e-8, 10);
  double                   temperature    = 2.58522e-8;
  MCGIDI::URR_protareInfos urr;
  
  // Declare loop variables
  //double energy, scatteringCrossSection, numberDensity;
  //int    matIndex, hashIndex, isoIndex;
  bool verification = true;
 
#if defined(_OPENMP)
  #pragma omp parallel for schedule(dynamic, 100) reduction(&&:verification)
#endif
  for (int iXS = 0; iXS < numVerify; iXS++)
  {
    
    uint64_t                 seed           = STARTING_SEED;
    seed = fast_forward_LCG(seed, 2 * (verifyStart + iXS));

    // Sample material and energy
    double energy    = pow(10.0, LCG_random_double(&seed) * 1.3);
    int matIndex  = pick_mat(&seed);
    int hashIndex = domainHash.index(energy);
    
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

      MCGIDI::Protare *MCProtare  = protares[isoIndex];

      scatteringCrossSection += numberDensity * MCProtare->reactionCrossSection(
          0, urr, hashIndex, temperature, energy);
    }

    // Check that calculated value matches the reference
    bool isMatch = approximatelyEqual(scatteringCrossSection, 
        verification_gpu[verifyStart + iXS], 
        std::numeric_limits<float>::epsilon());
    verification = verification and isMatch;
    if (not isMatch) 
    {
      printf("Mismatch at XS lookup %d \n", verifyStart + iXS);
      printf("GPU XS: %.17g \n", scatteringCrossSection);
      printf("CPU XS: %.17g \n", verification_gpu[verifyStart + iXS]);
      printf("diff: %.17g \n\n", 
          (double) fabs(scatteringCrossSection - verification_gpu[verifyStart + iXS]));
    }
  }

  return verification;
}

bool calcTotalMacroXSs(
    std::vector<MCGIDI::Protare *> protares,
    int    *materialComposition,
    double *numberDensities,
    double *verification_gpu,
    int     maxNumberIsotopes,
    int     verifyStart,
    int     numVerify) 
{       

  // Data used to evaluate XS
  MCGIDI::DomainHash       domainHash(4000, 1e-8, 10);
  double                   temperature    = 2.58522e-8;
  MCGIDI::URR_protareInfos urr;
  
  // Declare loop variables
  //double energy, scatteringCrossSection, numberDensity;
  //int    matIndex, hashIndex, isoIndex;
  bool verification = true;
 
#if defined(_OPENMP)
  #pragma omp parallel for schedule(dynamic, 100) reduction(&&:verification)
#endif
  for (int iXS = 0; iXS < numVerify; iXS++)
  {
    
    uint64_t                 seed           = STARTING_SEED;
    seed = fast_forward_LCG(seed, 2 * (verifyStart + iXS));

    // Sample material and energy
    double energy    = pow(10.0, LCG_random_double(&seed) * 1.3);
    int matIndex  = pick_mat(&seed);
    int hashIndex = domainHash.index(energy);
    
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

      MCGIDI::Protare *MCProtare  = protares[isoIndex];

      totalCrossSection += numberDensity * MCProtare->crossSection(
          urr, hashIndex, temperature, energy);
    }

    // Check that calculated value matches the reference
    bool isMatch = approximatelyEqual(totalCrossSection, 
        verification_gpu[verifyStart + iXS], 
        std::numeric_limits<float>::epsilon());
    verification = verification and isMatch;
    if (not isMatch) 
    {
      printf("Mismatch at XS lookup %d \n", verifyStart + iXS);
      printf("GPU XS: %.17g \n", totalCrossSection);
      printf("CPU XS: %.17g \n", verification_gpu[verifyStart + iXS]);
      printf("diff: %.17g \n\n", 
          (double) fabs(totalCrossSection - verification_gpu[verifyStart + iXS]));
    }
  }

  return verification;
}



