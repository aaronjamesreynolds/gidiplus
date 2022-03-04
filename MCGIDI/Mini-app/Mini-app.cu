#include "Mini-app.cuh"

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
    HM_size problem_size = small;             // small has 34 fuel nuclides, large has ~300 fuel nuclides                                
    int numOMPThreads = 1;                    // default number of OMP threads
    int numToVerify   = 100;                  // default number of XS to compare as calculate on host and device
 
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
    if (argc > 8) numOMPThreads  = atoi( argv[8] );
    if (argc > 9) numToVerify    = atoi( argv[9] );
    if (numToVerify > numCollisions) numToVerify = numCollisions;

    // Print runtime options
    printf("=== RUNTIME OPTIONS ===\n\n");
    printf( "doPrint = %d, numCollisions = %g, numIsotopes = %d, numBatches = %d, \n", doPrint, static_cast<double>( numCollisions ), numIsotopes, numBatches);
    printf( "numThreads = %d, doCompare = %d, numOMPThreads = %d \n\n",numThreads, doCompare, numOMPThreads);

    // Set number of OMP threads for CPU hash calc
    omp_set_num_threads(numOMPThreads);

    printf("=== INITIALIZING PROTARES  ===\n\n");
    
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

    printf("\n=== XS CALCULATION ===\n\n");

    printf("Calculating total XSs on GPU...\n");
    // Launch and time macroscopic total XS sampling kernel 
    double startTime = get_time();
    for (int iBatch = 0; iBatch < numBatches; iBatch++)
    {
      calcTotalMacroXSs<<<numBlocks, numThreads>>>(
          &deviceProtares[0], 
          materialCompositions, 
          numberDensities,
          verification,
          maxNumIsotopes,
          numCollisions);
      gpuErrchk( cudaPeekAtLastError( ) );
      gpuErrchk( cudaDeviceSynchronize( ) );
    }
    double endTime  = get_time();

    // Get XS calculation rate
    double elapsedTime = endTime - startTime;
    double xs_rate = (double) numBatches * numCollisions / elapsedTime;

    // Print out look-up rate
    printf("Looked up %d * %g XSs in %g seconds \n", numBatches, static_cast<double>(numCollisions), elapsedTime);
    printf("Total XS look-up rate: %g cross sections per second \n\n", xs_rate);
    
    printf("Calculating scatter XSs on GPU... \n");

    // Launch and time macroscopic scattering XS sampling kernel 
    startTime = get_time();
    for (int iBatch = 0; iBatch < numBatches; iBatch++)
    {
      calcScatterMacroXSs<<<numBlocks, numThreads>>>(
          &deviceProtares[0], 
          materialCompositions, 
          numberDensities,
          verification,
          maxNumIsotopes,
          numCollisions);
      gpuErrchk( cudaPeekAtLastError( ) );
      gpuErrchk( cudaDeviceSynchronize( ) );
    }
    endTime  = get_time();
    
    // Get XS calculation rate
    elapsedTime = endTime - startTime;
    xs_rate = (double) numBatches * numCollisions / elapsedTime;

    // Print out look-up rate
    printf("Looked up %d * %g XSs in %g seconds \n", numBatches, static_cast<double>(numCollisions), elapsedTime);
    printf("Scatter XS look-up rate: %g cross sections per second \n\n", xs_rate);

    if (doCompare == 1)
    {
      uint64_t seed = STARTING_SEED;
      int verifyStart = LCG_random_double(&seed) * (numCollisions - numToVerify);
    
      printf("=== CHECKING CPU/GPU CONSISTENCY ===\n\n");
      printf("Calculating scatter XSs on CPU... \n");

      // Get CPU comparison hash
      startTime = get_time();
      bool verification_match = calcScatterMacroXSs(
          protares,
          materialCompositions,
          numberDensities,
          verification,
          maxNumIsotopes,
          verifyStart,
          numToVerify);
      endTime  = get_time();
      elapsedTime = endTime - startTime;

      printf("CPU verification completed in %g seconds.\n\n", elapsedTime);
      if (verification_match)
        printf("Success! GPU and CPU scatter XSs for lookups %d through %d match!.\n",verifyStart, verifyStart + numToVerify - 1);
      else
      {
        printf("Failure! GPU and CPU scatter XSs for lookups %d through %d DO NOT match!.\n",verifyStart, verifyStart + numToVerify - 1);
      }
    }
    else
      printf("To verify consistency between host and device execution, set doCompare = 1.\n");

    return( EXIT_SUCCESS );
}


