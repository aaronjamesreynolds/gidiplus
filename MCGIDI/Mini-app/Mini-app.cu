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
    std::vector<MCGIDI::Protare *> protares = initMCProtares(in.numIsotopes, in.isotopeNames);

    // Print reaction data
    if (in.printData) printReactionData(protares);
  
    // Serialize protares, then copy them from host to device
    std::vector<char *> deviceProtares = copyProtaresFromHostToDevice(protares);

    // Calculate number of blocks in execution configuration
    int numBlocks = (in.numLookups + in.numThreads - 1) / in.numThreads;

    printf("\n=== XS CALCULATION ===\n\n");

    printf("TOTAL XS\n");
    printf("========\n");
    printf("Sampling total XSs on device...\n");

    // Launch and time macroscopic total XS sampling kernel 
    double startTime = get_time();
    for (int iBatch = 0; iBatch < in.numBatches; iBatch++)
    {
      calcTotalMacroXSs<<<numBlocks, in.numThreads>>>(
          &deviceProtares[0], 
          materialCompositions, 
          numberDensities,
          verification_device,
          maxNumIsotopes,
          in.numLookups);
      gpuErrchk( cudaPeekAtLastError( ) );
      gpuErrchk( cudaDeviceSynchronize( ) );
    }
    double endTime  = get_time();

    // Get XS calculation rate
    double elapsedTime = endTime - startTime;
    double xs_rate = (double) in.numBatches * in.numLookups / elapsedTime;

    // Print out look-up rate
    printf("Looked up %d * %g XSs in %g seconds \n", in.numBatches, static_cast<double>(in.numLookups), elapsedTime);
    printf("Total XS look-up rate: %g cross sections per second \n\n", xs_rate);
    
    printf("Sampling total XSs on host...\n");

    // Launch and time macroscopic total XS on the host
    startTime = get_time();
    for (int iBatch = 0; iBatch < in.numBatches; iBatch++)
    {
      calcTotalMacroXSs(
          protares,
          materialCompositions,
          numberDensities,
          verification_host,
          maxNumIsotopes,
          in.numLookups);
    }
    endTime  = get_time();
    
    // Get XS calculation rate
    elapsedTime = endTime - startTime;
    xs_rate = (double) in.numBatches * in.numLookups / elapsedTime;

    // Print out look-up rate
    printf("Looked up %d * %g XSs in %g seconds \n", in.numBatches, static_cast<double>(in.numLookups), elapsedTime);
    printf("Total XS look-up rate: %g cross sections per second \n\n", xs_rate);

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
    startTime = get_time();
    for (int iBatch = 0; iBatch < in.numBatches; iBatch++)
    {
      calcScatterMacroXSs<<<numBlocks, in.numThreads>>>(
          &deviceProtares[0], 
          materialCompositions, 
          numberDensities,
          verification_device,
          maxNumIsotopes,
          in.numLookups);
      gpuErrchk( cudaPeekAtLastError( ) );
      gpuErrchk( cudaDeviceSynchronize( ) );
    }
    endTime  = get_time();

    // Get XS calculation rate
    elapsedTime = endTime - startTime;
    xs_rate = (double) in.numBatches * in.numLookups / elapsedTime;

    // Print out look-up rate
    printf("Looked up %d * %g XSs in %g seconds \n", in.numBatches, static_cast<double>(in.numLookups), elapsedTime);
    printf("Scatter XS look-up rate: %g cross sections per second \n\n", xs_rate);

    printf("Sampling scatter XSs on host...\n");

    // Launch and time macroscopic scattering XS on the host
    startTime = get_time();
    for (int iBatch = 0; iBatch < in.numBatches; iBatch++)
    {
      calcScatterMacroXSs(
          protares,
          materialCompositions,
          numberDensities,
          verification_host,
          maxNumIsotopes,
          in.numLookups);
    }
    endTime  = get_time();

    // Get XS calculation rate
    elapsedTime = endTime - startTime;
    xs_rate = (double) in.numBatches * in.numLookups / elapsedTime;

    // Print out look-up rate
    printf("Looked up %d * %g XSs in %g seconds \n", in.numBatches, static_cast<double>(in.numLookups), elapsedTime);
    printf("Scatter XS look-up rate: %g cross sections per second \n\n", xs_rate);

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


