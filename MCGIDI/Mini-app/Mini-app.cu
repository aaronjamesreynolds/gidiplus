/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

/*
Brief: Driver for XS lookups
Author: Aaron James Reynolds
*/

#include "Mini-app.cuh"

/*
===============================================================================
*/
int main( int argc, char **argv ) 
{

    Input in = Input(argc, argv);

    // Print logo
    in.printLogo();

    // Print runtime options
    printf("=== RUNTIME OPTIONS ===\n\n");
    in.printInputOptions();

    printf("=== INITIALIZING PROTARES ===\n\n");
    
    // Set up material compositions and number densities
    std::vector<std::vector<int>> materialCompositions2D 
      = initMaterialCompositions(in.problem_size);
    std::vector<std::vector<double>> numberDensities2D 
      = initNumberDensities(materialCompositions2D);

    int      numMats = materialCompositions2D.size(), 
             maxNumIsotopes = materialCompositions2D[0].size(),
             numEntries = numMats * maxNumIsotopes;
    int    * materialCompositions;
    double * numberDensities;
    double * verification_host, *verification_device;

    int deviceId;
    cudaGetDevice(&deviceId);

    // Allocate memory for material data
    size_t sizeMatComp            = numEntries * sizeof(int);
    size_t sizeNumDens            = numEntries * sizeof(double);
    size_t sizeVerificationDevice = in.numDeviceLookups * sizeof(double);
    size_t sizeVerificationHost   = in.numHostLookups * sizeof(double);

    if (in.hostOnly)
    {
      materialCompositions = (int * ) malloc(sizeMatComp);
      numberDensities = (double * ) malloc(sizeNumDens);
      verification_host = (double * ) malloc(sizeVerificationHost);
    }
    else
    {
      cudaMallocManaged(&materialCompositions, sizeMatComp);
      cudaMallocManaged(&numberDensities,      sizeNumDens); 
      cudaMallocManaged(&verification_host,    sizeVerificationHost); 
      cudaMallocManaged(&verification_device,  sizeVerificationDevice); 
    }

    // Initialize 1D material composition and number density vectors
    unwrapFrom2Dto1D(materialCompositions2D, 
        materialCompositions, 
        numMats, 
        maxNumIsotopes);
    unwrapFrom2Dto1D(numberDensities2D, 
        numberDensities, 
        numMats, 
        maxNumIsotopes);

    // Copy material compositions and number densities to device
    if (!in.hostOnly)
    {
      cudaMemPrefetchAsync(materialCompositions, sizeMatComp, deviceId);
      cudaMemPrefetchAsync(numberDensities,      sizeNumDens, deviceId);
      cudaMemPrefetchAsync(verification_device,  
          sizeVerificationDevice, 
          deviceId);
    }

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
    MCGIDI::DomainHash     * ceDomainHash = NULL;
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
    
    printf("\n=== XS CALCULATION ===\n\n");

    printf("TOTAL XS\n");
    printf("========\n");
    
    // Timing variable
    std::vector<double> edgeTimes;

    // Sampling results
    std::vector<LookupRate_t> lookupRates;
 
    if (!in.hostOnly) 
    {
      // Serialize protares, then copy them from host to device
      std::vector<char *> deviceProtares = copyProtaresFromHostToDevice(protares);

      // Calculate number of blocks in execution configuration
      int numBlocks = (in.numDeviceLookups + in.numThreads - 1) / in.numThreads;

      // Sample XSs
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
                in.numDeviceLookups,
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
                in.numDeviceLookups,
                in.sampleProduct);
            gpuErrchk( cudaPeekAtLastError( ) );
            gpuErrchk( cudaDeviceSynchronize( ) );
            edgeTimes.push_back(get_time());
          }
          break;
      }

      // Get XS calculation rate
      lookupRates.push_back(calcLookupRate(
            edgeTimes, 
            in.numDeviceLookups, 
            in.numBatches, 
            "Total"));
      lookupRates.back().print();
      edgeTimes.clear();
    }

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
              in.numHostLookups,
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
              in.numHostLookups,
              in.sampleProduct);
          edgeTimes.push_back(get_time());
        }
        break;
    }

    // Get XS calculation rate
    lookupRates.push_back(calcLookupRate(
          edgeTimes, 
          in.numHostLookups, 
          in.numBatches, 
          "Total"));
    lookupRates.back().print();
    edgeTimes.clear();

    if (!in.hostOnly)
    {
      // Verify that host and device XSs match
      int verificationSize = (in.numDeviceLookups < in.numHostLookups 
          ? in.numDeviceLookups : in.numHostLookups);
      bool verification_match = approximatelyEqual(
          verification_host, 
          verification_device, 
          verificationSize,         
          std::numeric_limits<float>::epsilon());

      if (verification_match)
        printf("Success! GPU and CPU total XSs lookups match!\n\n");
      else
        printf("Failure! GPU and CPU total XSs lookups DO NOT match!.\n\n");
    }

    // Free pointers
    delete(ceDomainHash);
    delete(mgDomainHash);

    return( EXIT_SUCCESS );
}
