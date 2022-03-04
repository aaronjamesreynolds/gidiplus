#include "Mini-app.cuh"

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

    printf("Initialized %d MCGIDI protares in %f seconds.\n", numIsotopes, elapsed_time);

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
    
    printf("Copied %d buffered MCGIDI protares from host to device.\n", numIsotopes);

    setUp<<< numIsotopes, 32 >>>( numIsotopes, deviceBuffers_d );

    gpuErrchk( cudaPeekAtLastError( ) );
    gpuErrchk( cudaDeviceSynchronize( ) );

    return deviceProtares;

}

