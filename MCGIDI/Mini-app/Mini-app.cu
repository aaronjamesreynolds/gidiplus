#include "Mini-app.cuh"

/*
=========================================================
*/
int main( int argc, char **argv ) 
{

    Input in = Input(argc, argv);

    // Print logo
    in.printLogo();

    // Print runtime options
    printf("=== RUNTIME OPTIONS ===\n\n");
    in.printInputOptions();

#if defined(_OPENMP)
    // Set number of OMP threads for CPU hash calc
    omp_set_num_threads(in.numOMPThreads);
#endif

    printf("=== INITIALIZING PROTARES ===\n\n");
    
    // Set up material compositions and number densities
    std::vector<std::vector<int>> materialCompositions2D = initMaterialCompositions(in.problem_size);
    std::vector<std::vector<double>> numberDensities2D = initNumberDensities(materialCompositions2D);

    int    numMats = materialCompositions2D.size(), 
           maxNumIsotopes = materialCompositions2D[0].size(),
           numEntries = numMats * maxNumIsotopes;
    int    *materialCompositions;
    double *numberDensities;
    double *verification_host, *verification_device;

    int deviceId;
    cudaGetDevice(&deviceId);

    // Allocate memory for material data
    size_t sizeMatComp      = numEntries * sizeof(int);
    size_t sizeNumDens      = numEntries * sizeof(double);
    size_t sizeVerification = in.numLookups * sizeof(double);
    cudaMallocManaged(&materialCompositions, sizeMatComp);
    cudaMallocManaged(&numberDensities,      sizeNumDens); 
    cudaMallocManaged(&verification_host,     sizeVerification); 
    cudaMallocManaged(&verification_device,   sizeVerification); 

    // Initialize 1D material composition and number density vectors
    unwrapFrom2Dto1D(materialCompositions2D, materialCompositions, numMats, maxNumIsotopes);
    unwrapFrom2Dto1D(numberDensities2D, numberDensities, numMats, maxNumIsotopes);

    // Copy material compositions and number densities to device
    cudaMemPrefetchAsync(materialCompositions, sizeMatComp,      deviceId);
    cudaMemPrefetchAsync(numberDensities,      sizeNumDens,      deviceId);
    cudaMemPrefetchAsync(verification_device,  sizeVerification, deviceId);

    // Set and verify CUDA limits
    // These options were in gpuTest. If I use them, I run out of device memory, so I'm not using them.
    //setCudaOptions();

    // Initialize protares and nuclear data maps
    std::vector<MCGIDI::Protare *> protares = initMCProtares(
        in.numIsotopes, 
        in.isotopeNames, 
        in.mode, 
        in.numHashBins);

    // Ideally, there would only be a single domain hash variable. 
    // Unfortunately, the CE and MG domain hash don't share a parent class,
    // so we have to declare both types, and use switch cases for the XS 
    // calculations calls.
    MCGIDI::DomainHash * ceDomainHash = NULL;
    MCGIDI::MultiGroupHash * mgDomainHash= NULL;
    switch (in.mode)
    {
      case ce:
        ceDomainHash = getCEHash(in.numHashBins);
        break;
      case mg:
        mgDomainHash= getMGHash(in.isotopeNames);
        break;
    }

    // Print reaction data
    if (in.printData) printReactionData(protares);
  
    // Serialize protares, then copy them from host to device
    std::vector<char *> deviceProtares = copyProtaresFromHostToDevice(protares);

    // Calculate number of blocks in execution configuration
    int numBlocks = (in.numLookups + in.numThreads - 1) / in.numThreads;

    // Timing variable
    std::vector<double> edgeTimes;

    // Sampling results
    std::vector<LookupRate_t> lookupRates;

    // Sample XSs
    printf("\n=== XS CALCULATION ===\n\n");

    printf("TOTAL XS\n");
    printf("========\n");
    printf("Sampling total XSs on device...\n");

    // Launch and time macroscopic total XS sampling kernel 
    edgeTimes.push_back(get_time());
    switch (in.mode)
    {
      case ce:
        for (int iBatch = 0; iBatch < in.numBatches; iBatch++)
        {
          calcTotalMacroXSs<<<numBlocks, in.numThreads>>>(
              &deviceProtares[0], 
              ceDomainHash,
              materialCompositions, 
              numberDensities,
              verification_device,
              maxNumIsotopes,
              in.numLookups,
              in.sampleProduct);
          gpuErrchk( cudaPeekAtLastError( ) );
          gpuErrchk( cudaDeviceSynchronize( ) );
          edgeTimes.push_back(get_time());
        }
        break;
      case mg:
        for (int iBatch = 0; iBatch < in.numBatches; iBatch++)
        {
          calcTotalMacroXSs<<<numBlocks, in.numThreads>>>(
              &deviceProtares[0], 
              mgDomainHash,
              materialCompositions, 
              numberDensities,
              verification_device,
              maxNumIsotopes,
              in.numLookups,
              in.sampleProduct);
          gpuErrchk( cudaPeekAtLastError( ) );
          gpuErrchk( cudaDeviceSynchronize( ) );
          edgeTimes.push_back(get_time());
        }
        break;
    }

    // Get XS calculation rate
    lookupRates.push_back(calcLookupRate(edgeTimes, in.numLookups, in.numBatches, "Total"));
    lookupRates.back().print();
    edgeTimes.clear();

    printf("Sampling total XSs on host...\n");

    // Launch and time macroscopic total XS on the host
    edgeTimes.push_back(get_time());
    switch (in.mode)
    {
      case ce:    
        for (int iBatch = 0; iBatch < in.numBatches; iBatch++)
        {
          calcTotalMacroXSs(
              protares,
              ceDomainHash,
              materialCompositions,
              numberDensities,
              verification_host,
              maxNumIsotopes,
              in.numLookups,
              in.sampleProduct);
          edgeTimes.push_back(get_time());
        }
        break;
      case mg:    
        for (int iBatch = 0; iBatch < in.numBatches; iBatch++)
        {
          calcTotalMacroXSs(
              protares,
              mgDomainHash,
              materialCompositions,
              numberDensities,
              verification_host,
              maxNumIsotopes,
              in.numLookups,
              in.sampleProduct);
          edgeTimes.push_back(get_time());
        }
        break;
    }

    // Get XS calculation rate
    lookupRates.push_back(calcLookupRate(edgeTimes, in.numLookups, in.numBatches, "Total"));
    lookupRates.back().print();
    edgeTimes.clear();

    // Verify that host and device XSs match
    bool verification_match = approximatelyEqual(verification_host, 
        verification_device, 
        in.numLookups, 
        std::numeric_limits<float>::epsilon());

    if (verification_match)
      printf("Success! GPU and CPU total XSs lookups match!\n\n");
    else
      printf("Failure! GPU and CPU total XSs lookups DO NOT match!.\n\n");

    printf("SCATTER XS\n");
    printf("==========\n");

    printf("Sampling scatter XSs on device... \n");

    // Launch and time macroscopic scattering XS sampling kernel 
    edgeTimes.push_back(get_time());
    switch (in.mode)
    {
      case ce:    
        for (int iBatch = 0; iBatch < in.numBatches; iBatch++)
        {
          calcScatterMacroXSs<<<numBlocks, in.numThreads>>>(
              &deviceProtares[0], 
              ceDomainHash,
              materialCompositions, 
              numberDensities,
              verification_device,
              maxNumIsotopes,
              in.numLookups,
              in.sampleProduct);
          gpuErrchk( cudaPeekAtLastError( ) );
          gpuErrchk( cudaDeviceSynchronize( ) );
          edgeTimes.push_back(get_time());
        }
        break;
      case mg:
        for (int iBatch = 0; iBatch < in.numBatches; iBatch++)
        {
          calcScatterMacroXSs<<<numBlocks, in.numThreads>>>(
              &deviceProtares[0], 
              mgDomainHash,
              materialCompositions, 
              numberDensities,
              verification_device,
              maxNumIsotopes,
              in.numLookups,
              in.sampleProduct);
          gpuErrchk( cudaPeekAtLastError( ) );
          gpuErrchk( cudaDeviceSynchronize( ) );
          edgeTimes.push_back(get_time());
        }
        break;
    }

    // Get XS calculation rate
    lookupRates.push_back(calcLookupRate(edgeTimes, in.numLookups, in.numBatches, "Scatter"));
    lookupRates.back().print();
    edgeTimes.clear();

    printf("Sampling scatter XSs on host...\n");

    // Launch and time macroscopic scattering XS on the host
    edgeTimes.push_back(get_time());
    switch (in.mode)
    {
      case ce:
        for (int iBatch = 0; iBatch < in.numBatches; iBatch++)
        {
          calcScatterMacroXSs(
              protares,
              ceDomainHash,
              materialCompositions,
              numberDensities,
              verification_host,
              maxNumIsotopes,
              in.numLookups,
              in.sampleProduct);
          edgeTimes.push_back(get_time());
        }
        break;
      case mg:
        for (int iBatch = 0; iBatch < in.numBatches; iBatch++)
        {
          calcScatterMacroXSs(
              protares,
              mgDomainHash,
              materialCompositions,
              numberDensities,
              verification_host,
              maxNumIsotopes,
              in.numLookups,
              in.sampleProduct);
          edgeTimes.push_back(get_time());
        }
        break;
    }

    // Get XS calculation rate
    lookupRates.push_back(calcLookupRate(edgeTimes, in.numLookups, in.numBatches, "Scatter"));
    lookupRates.back().print();
    edgeTimes.clear();

    // Verify that host and device XSs match
    verification_match = approximatelyEqual(verification_host, 
        verification_device, 
        in.numLookups, 
        std::numeric_limits<float>::epsilon());

    if (verification_match)
      printf("Success! GPU and CPU scatter XSs lookups match!\n\n");
    else
      printf("Failure! GPU and CPU scatter XSs lookups DO NOT match!.\n\n");

    return( EXIT_SUCCESS );
}


