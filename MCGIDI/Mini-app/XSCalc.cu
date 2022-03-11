#include "Mini-app.cuh"

/*
=========================================================
*/
// Calculate scatter cross section for a random protare. 
// Material compositions and number densities pre-initialized.
// 

template <typename T>
__global__ void calcScatterMacroXSs(
    char   **deviceProtares,
    T      *domainHash,
    int    *materialComposition,
    double *numberDensities,
    double *verification,
    int     maxNumberIsotopes,
    int     numCollisions,
    bool    sampleProduct)
{       

  int collisionIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if( collisionIndex >= numCollisions ) return;

  // Data used to evaluate XS
  uint64_t                 seed           = STARTING_SEED;
  double                   temperature    = 2.58522e-8;
  MCGIDI::URR_protareInfos urr;

  // Fast-forward random number seed
  seed = fast_forward_LCG(seed, 2*collisionIndex);
  
  // Sample material and energy
  double energy    = pow(10.0, LCG_random_double(&seed) * 1.3);
  int    matIndex  = pick_mat(&seed);
  int    hashIndex = domainHash->index(energy);
  
  // Initialize accumulators and loop variables
  double microCrossSection, scatteringCrossSection = 0;
  double numberDensity = -1;
  int isoIndex = -1;
  MCGIDI::ProtareSingle *MCProtare; 

  // Evaluate scattering and total XS
  for (int iConstituent = 0; 
       materialComposition[matIndex * maxNumberIsotopes + iConstituent] >= 0 
       && iConstituent < maxNumberIsotopes; 
       iConstituent++)
  {
    isoIndex      = materialComposition[matIndex * maxNumberIsotopes + iConstituent];
    numberDensity = numberDensities[matIndex * maxNumberIsotopes + iConstituent];

    MCProtare  = reinterpret_cast<MCGIDI::ProtareSingle *>(deviceProtares[isoIndex]);

    microCrossSection = MCProtare->reactionCrossSection(
        0, urr, hashIndex, temperature, energy);
    scatteringCrossSection += numberDensity * microCrossSection;
  }

  // Sample reaction and product
  if (sampleProduct)
  {
    TallyProductHandler product = sampleProducts(
        MCProtare,
        hashIndex,
        temperature,
        energy, 
        microCrossSection,
        &seed);
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

template <typename T>
__global__ void calcTotalMacroXSs(
    char   **deviceProtares,
    T      *domainHash,
    int    *materialComposition,
    double *numberDensities,
    double *verification,
    int     maxNumberIsotopes,
    int     numCollisions, 
    bool    sampleProduct) 
{       

  int collisionIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if( collisionIndex >= numCollisions ) return;

  // Data used to evaluate XS
  uint64_t                 seed           = STARTING_SEED;
  double                   temperature    = 2.58522e-8;
  MCGIDI::URR_protareInfos urr;

  // Fast-forward random number seed
  seed = fast_forward_LCG(seed, 2*collisionIndex);
  
  // Sample material and energy
  double energy    = pow(10.0, LCG_random_double(&seed) * 1.3);
  int    matIndex  = pick_mat(&seed);
  int    hashIndex = domainHash->index(energy);
  
  // Initialize accumulators and loop variables
  double microCrossSection, totalCrossSection = 0;
  double numberDensity = -1;
  int isoIndex = -1;
  MCGIDI::ProtareSingle *MCProtare; 

  // Evaluate scattering and total XS
  for (int iConstituent = 0; 
       materialComposition[matIndex * maxNumberIsotopes + iConstituent] >= 0 
       && iConstituent < maxNumberIsotopes; 
       iConstituent++)
  {
    isoIndex      = materialComposition[matIndex * maxNumberIsotopes + iConstituent];
    numberDensity = numberDensities[matIndex * maxNumberIsotopes + iConstituent];

    MCProtare  = reinterpret_cast<MCGIDI::ProtareSingle *>(deviceProtares[isoIndex]);

    microCrossSection = MCProtare->crossSection(urr, hashIndex, temperature, energy);
    totalCrossSection += numberDensity * microCrossSection;
  }
  
  // Sample reaction and product
  if (sampleProduct)
  {
    TallyProductHandler product = sampleProducts(
        MCProtare,
        hashIndex,
        temperature,
        energy, 
        microCrossSection,
        &seed);
  }

  verification[collisionIndex] = totalCrossSection;
}

template <typename T>
void calcScatterMacroXSs(
    std::vector<MCGIDI::Protare *> protares,
    T      *domainHash,
    int    *materialComposition,
    double *numberDensities,
    double *verification,
    int     maxNumberIsotopes,
    int     numLookups, 
    bool    sampleProduct) 

{       

  // Data used to evaluate XS
  double                   temperature    = 2.58522e-8;
  MCGIDI::URR_protareInfos urr;

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 100)
#endif
  for (int iXS = 0; iXS < numLookups; iXS++)
  {

    uint64_t                 seed           = STARTING_SEED;
    seed = fast_forward_LCG(seed, 2 * iXS);

    // Sample material and energy
    double energy    = pow(10.0, LCG_random_double(&seed) * 1.3);
    int matIndex  = pick_mat(&seed);
    int hashIndex = domainHash->index(energy);

    // Initialize accumulators and loop variables
    double microCrossSection, scatteringCrossSection = 0;
    double numberDensity = -1;
    int isoIndex = -1;
    MCGIDI::Protare * MCProtare; 


    // Evaluate scattering and total XS
    for (int iConstituent = 0; 
        materialComposition[matIndex * maxNumberIsotopes + iConstituent] >= 0 
        && iConstituent < maxNumberIsotopes; 
        iConstituent++)
    {
      isoIndex      = materialComposition[matIndex * maxNumberIsotopes + iConstituent];
      numberDensity = numberDensities[matIndex * maxNumberIsotopes + iConstituent];

      MCProtare = protares[isoIndex];

      microCrossSection = MCProtare->reactionCrossSection(
          0, urr, hashIndex, temperature, energy);
      scatteringCrossSection += numberDensity * microCrossSection;
    }

    // Sample reaction and product
    if (sampleProduct)
    {
      TallyProductHandler product = sampleProducts(
          MCProtare,
          hashIndex,
          temperature,
          energy, 
          microCrossSection,
          &seed);
    }

    verification[iXS] = scatteringCrossSection;

  }

}

  template <typename T>
void calcTotalMacroXSs(
    std::vector<MCGIDI::Protare *> protares,
    T      *domainHash,
    int    *materialComposition,
    double *numberDensities,
    double *verification,
    int     maxNumberIsotopes,
    int     numLookups,
    bool    sampleProduct) 
{       

  // Data used to evaluate XS
  double                   temperature    = 2.58522e-8;
  MCGIDI::URR_protareInfos urr;

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 100)
#endif
  for (int iXS = 0; iXS < numLookups; iXS++)
  {

    uint64_t                 seed           = STARTING_SEED;
    seed = fast_forward_LCG(seed, 2 * iXS);

    // Sample material and energy
    double energy    = pow(10.0, LCG_random_double(&seed) * 1.3);
    int matIndex  = pick_mat(&seed);
    int hashIndex = domainHash->index(energy);

    // Initialize accumulators and loop variables
    double microCrossSection, totalCrossSection = 0;
    double numberDensity = -1;
    int isoIndex = -1;
    MCGIDI::Protare * MCProtare; 

    // Evaluate scattering and total XS
    for (int iConstituent = 0; 
        materialComposition[matIndex * maxNumberIsotopes + iConstituent] >= 0 
        && iConstituent < maxNumberIsotopes; 
        iConstituent++)
    {
      isoIndex      = materialComposition[matIndex * maxNumberIsotopes + iConstituent];
      numberDensity = numberDensities[matIndex * maxNumberIsotopes + iConstituent];

      MCProtare = protares[isoIndex];

      microCrossSection = MCProtare->crossSection(
          urr, hashIndex, temperature, energy);
      totalCrossSection += numberDensity * microCrossSection; 

    }

    // Sample reaction and product
    if (sampleProduct)
    {
      TallyProductHandler product = sampleProducts(
          MCProtare,
          hashIndex,
          temperature,
          energy, 
          microCrossSection,
          &seed);
    }

    verification[iXS] = totalCrossSection;

  }

}

MCGIDI_HOST_DEVICE TallyProductHandler sampleProducts(
    MCGIDI::ProtareSingle *MCProtare,
    int hashIndex,
    double temperature,
    double energy, 
    double crossSection,
    uint64_t * seed)
{

  // Declare sampling variables 
  MCGIDI::Sampling::Input input( true, MCGIDI::Sampling::Upscatter::Model::B );
  TallyProductHandler products;
  MCGIDI::URR_protareInfos urr;

  // Sample a reaction
  int reactionIndex = MCProtare->sampleReaction( 
      urr, 
      hashIndex, 
      temperature, 
      energy, 
      crossSection, 
      (double (*)(void *)) LCG_random_double, 
      &seed );
  MCGIDI::Reaction const *reaction = MCProtare->reaction( reactionIndex );

  // Sample products of reaction
  reaction->sampleProducts(
      MCProtare, 
      energy, 
      input, 
      (double (*)( void * )) LCG_random_double, 
      &seed, 
      products);

  return products;
}

MCGIDI_HOST_DEVICE TallyProductHandler sampleProducts(
    MCGIDI::Protare *MCProtare,
    int hashIndex,
    double temperature,
    double energy, 
    double crossSection,
    uint64_t * seed)
{

  // Declare sampling variables 
  MCGIDI::Sampling::Input input( true, MCGIDI::Sampling::Upscatter::Model::B );
  TallyProductHandler products;
  MCGIDI::URR_protareInfos urr;

  // Sample a reaction
  int reactionIndex = MCProtare->sampleReaction( 
      urr, 
      hashIndex, 
      temperature, 
      energy, 
      crossSection, 
      (double (*)(void *)) LCG_random_double, 
      &seed );
  MCGIDI::Reaction const *reaction = MCProtare->reaction( reactionIndex );

  // Sample products of reaction
  reaction->sampleProducts(
      MCProtare, 
      energy, 
      input, 
      (double (*)( void * )) LCG_random_double, 
      &seed, 
      products);

  return products;
}


template
__global__ void calcScatterMacroXSs(
    char   **deviceProtares,
    MCGIDI::DomainHash      *domainHash,
    int    *materialComposition,
    double *numberDensities,
    double *verification,
    int     maxNumberIsotopes,
    int     numCollisions,
    bool    sampleProduct);
template
__global__ void calcTotalMacroXSs(
    char   **deviceProtares,
    MCGIDI::DomainHash      *domainHash,
    int    *materialComposition,
    double *numberDensities,
    double *verification,
    int     maxNumberIsotopes,
    int     numCollisions,
    bool    sampleProduct);
template 
void calcScatterMacroXSs(
    std::vector<MCGIDI::Protare *> protares,
    MCGIDI::DomainHash      *domainHash,
    int    *materialComposition,
    double *numberDensities,
    double *verification,
    int     maxNumberIsotopes,
    int     numLookups,
    bool    sampleProduct);
template 
void calcTotalMacroXSs(
    std::vector<MCGIDI::Protare *> protares,
    MCGIDI::DomainHash      *domainHash,
    int    *materialComposition,
    double *numberDensities,
    double *verification,
    int     maxNumberIsotopes,
    int     numLookups,
    bool    sampleProduct);
template
__global__ void calcScatterMacroXSs(
    char   **deviceProtares,
    MCGIDI::MultiGroupHash      *domainHash,
    int    *materialComposition,
    double *numberDensities,
    double *verification,
    int     maxNumberIsotopes,
    int     numCollisions,
    bool    sampleProduct); 
template
__global__ void calcTotalMacroXSs(
    char   **deviceProtares,
    MCGIDI::MultiGroupHash      *domainHash,
    int    *materialComposition,
    double *numberDensities,
    double *verification,
    int     maxNumberIsotopes,
    int     numCollision,
    bool    sampleProduct);
template 
void calcScatterMacroXSs(
    std::vector<MCGIDI::Protare *> protares,
    MCGIDI::MultiGroupHash      *domainHash,
    int    *materialComposition,
    double *numberDensities,
    double *verification,
    int     maxNumberIsotopes,
    int     numLookups,
    bool    sampleProduct);
template 
void calcTotalMacroXSs(
    std::vector<MCGIDI::Protare *> protares,
    MCGIDI::MultiGroupHash      *domainHash,
    int    *materialComposition,
    double *numberDensities,
    double *verification,
    int     maxNumberIsotopes,
    int     numLookups,
    bool    sampleProduct); 
