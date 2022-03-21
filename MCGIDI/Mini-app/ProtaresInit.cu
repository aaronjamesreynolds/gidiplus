/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

/*
Brief: Protare initialization and H-to-D memory operations
Author: Aaron James Reynolds 
(processes adapted from those in gpuTest.cpp)
*/

#include "Mini-app.cuh"

/*
===============================================================================
Serialize a protare (Taken from MCGICI/Test/gpuTest/gpuTest.cpp)

Call this each isotope per block and one warp only 
(i.e. <<< number_isotopes, 32>>>)
*/
__global__ void setUp( int a_numIsotopes, MCGIDI::DataBuffer **a_buf ) {  

    int isotopeIndex = blockIdx.x;

    MCGIDI::DataBuffer *buf = a_buf[isotopeIndex];
    MCGIDI::ProtareSingle *MCProtare = new(buf->m_placementStart) MCGIDI::ProtareSingle( );

    buf->zeroIndexes( );
    buf->m_placement = buf->m_placementStart + sizeof( MCGIDI::ProtareSingle );
    buf->m_maxPlacementSize = sizeof( *a_buf[isotopeIndex] ) + sizeof( MCGIDI::ProtareSingle );

    // This line causes a "nvlink warning".
    MCProtare->serialize( *buf, MCGIDI::DataBuffer::Mode::Unpack );                 
    buf->m_placement = buf->m_placementStart + sizeof( MCGIDI::ProtareSingle );
}

/*
===============================================================================
Initialize MG or CE protares for XS lookup rate calculation 
(Adapted from MCGICI/Test/gpuTest/gpuTest.cpp)

Using paths that were in gpuTest.cpp. Ideally, these paths would be defined in
the make file. 
*/
std::vector<MCGIDI::Protare *> initMCProtares(
    int numIsotopes, 
    const char *isotopeNames[], 
    energyMode mode, 
    int numHashBins)
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
    MCGIDI::DomainHash domainHash( numHashBins, 1e-8, 10 );
    std::set<int>                  exclusionSet;

    // Initialize progress message
    std::string progress_str = "";

    // For each isotope referenced in isotopeNames, construct a GIDI::protare. 
    // Then, initialize a MCGIDI::protare from the GIDI::protare object. 
    double startTime = get_time();
    for( int isoIndex = 0; isoIndex < numIsotopes; isoIndex++ ) 
    {
      std::string protareFilename( map.protareFilename( PoPI::IDs::neutron, isotopeNames[isoIndex] ) );

      // Initialize GIDI::protare
      GIDI::Protare                * protare;
      GIDI::Construction::Settings   construction( 
          GIDI::Construction::ParseMode::excludeProductMatrices, 
          GIDI::Construction::PhotoMode::nuclearAndAtomic );
      protare = map.protare(
          construction, 
          pops, 
          PoPI::IDs::neutron, 
          isotopeNames[isoIndex] );

      // Initialize a CE or MG protare depending on mode
      // Note: only initializing MCGIDI protares with one temperature
      switch (mode)
      {
        case(ce):
          {
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
            GIDI::Transporting::Particle   projectile( "n", continuous_energy_multigroup);
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
            break;
          }
        case(mg):
          {
            GIDI::Transporting::Particles  particleList;
            GIDI::Styles::TemperatureInfos temperature = {protare->temperatures()[0]};
            std::string label( temperature[0].heatedMultiGroup( ) );

            //GIDI::Transporting::Groups_from_bdfls groups_from_bdfls( "../../GIDI/Test/bdfls" );
            //GIDI::Transporting::Fluxes_from_bdfls fluxes_from_bdfls( "../../GIDI/Test/bdfls", 0.0 );
            GIDI::Transporting::Groups_from_bdfls groups_from_bdfls( "/collab/usr/gdata/nuclear/endl_official/endl2009.3/bdfls" );
            GIDI::Transporting::Fluxes_from_bdfls fluxes_from_bdfls( "/collab/usr/gdata/nuclear/endl_official/endl2009.3/bdfls", 0.0 );

            std::string gid( "LLNL_gid_7" );
            GIDI::Transporting::MultiGroup multi_group = groups_from_bdfls.viaLabel( gid );
            GIDI::Transporting::Particle projectile("n", multi_group );
            projectile.appendFlux( fluxes_from_bdfls.getViaFID( 1 ) );
            particleList.add( projectile );
            particleList.process( *protare, label );
            MCGIDI::Transporting::MC MC( 
                pops, 
                PoPI::IDs::neutron, 
                &protare->styles( ), 
                label, 
                GIDI::Transporting::DelayedNeutrons::on, 
                20.0 );
            MC.crossSectionLookupMode( MCGIDI::Transporting::LookupMode::Data1d::multiGroup );

            // Construct MCGIDI::protare from GIDI::protare
            protares[isoIndex] = MCGIDI::protareFromGIDIProtare( 
                *protare, 
                pops, 
                MC, 
                particleList, 
                domainHash, 
                temperature, 
                exclusionSet ); 
            break;
          }
      }

      // Clear out last progress message and print the new one
      std::cout << std::string(progress_str.length(),'\b');
      progress_str = "Initialized " + std::to_string(isoIndex) 
        + " / " + std::to_string(numIsotopes) + " protares";
      std::cout << progress_str;

    }
    double endTime = get_time();
    double elapsedTime = endTime - startTime;

    // Print protare intialization time
    std::cout << std::string(progress_str.length(),'\b');
    printf("Initialized %d MCGIDI protares in %f seconds.\n", numIsotopes, elapsedTime);

    return protares;

}

/*
===============================================================================
Print reaction data for a list of protares 
(Taken from MCGICI/Test/gpuTest/gpuTest.cpp)
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
===============================================================================
Copy protares from host to device (Taken from MCGICI/Test/gpuTest/gpuTest.cpp)
*/
std::vector<char *> copyProtaresFromHostToDevice(
    std::vector<MCGIDI::Protare *> protares)
{

  int numIsotopes = protares.size();
  size_t totalSize = 0;

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
    totalSize += cpuSize;
    deviceBuffers_h[isoIndex] = buf_h.copyToDevice( cpuSize, deviceProtares[isoIndex] );
  }

  // Copy data buffer from host to device
  MCGIDI::DataBuffer **deviceBuffers_d = nullptr;
  cudaMalloc( (void **) &deviceBuffers_d, sizeof( MCGIDI::DataBuffer * ) * numIsotopes );
  cudaMemcpy( deviceBuffers_d, &deviceBuffers_h[0], sizeof( MCGIDI::DataBuffer * ) * numIsotopes, cudaMemcpyHostToDevice );

  printf("Copied %d buffered MCGIDI protares (%d MB) from host to device.\n", 
      numIsotopes,
      int(totalSize / (1024.0 * 1024.0)));

  setUp<<< numIsotopes, 32 >>>( numIsotopes, deviceBuffers_d );

  gpuErrchk( cudaPeekAtLastError( ) );
  gpuErrchk( cudaDeviceSynchronize( ) );

  return deviceProtares;

}

/*
===============================================================================
Initialize a multigroup hash and return a pointer to it
*/
MCGIDI::MultiGroupHash * getMGHash(const char *isotopeNames[])
{

  // Initialize protares and nuclear data maps
  std::vector<MCGIDI::Protare *> protares(1);
  std::string mapFilename( "/usr/gapps/Mercury/data/nuclear/endl/2009.3_gp3.17/gnd/all.map" );
  PoPI::Database pops( "/usr/gapps/Mercury/data/nuclear/endl/2009.3/gnd/pops.xml" );
  std::ifstream meta_stream( "/usr/gapps/data/nuclear/development/GIDI3/Versions/V10/metastables_alias.xml" );
  std::string metastable_string( ( std::istreambuf_iterator<char>( meta_stream ) ), 
      std::istreambuf_iterator<char>( ) );
  pops.addDatabase( metastable_string, false );
  GIDI::Map::Map map( mapFilename, pops );
  MCGIDI::DomainHash domainHash( 4000, 1e-8, 10 );

  std::string protareFilename( map.protareFilename( PoPI::IDs::neutron, isotopeNames[0] ) );

  // Initialize GIDI::protare
  GIDI::Protare                *protare;
  GIDI::Construction::Settings construction( 
      GIDI::Construction::ParseMode::excludeProductMatrices, 
      GIDI::Construction::PhotoMode::nuclearAndAtomic );
  protare = map.protare( construction, pops, PoPI::IDs::neutron, isotopeNames[0] );

  GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
  MCGIDI::MultiGroupHash * mgDomainHash = new MCGIDI::MultiGroupHash( *protare, temperatures[0] );

  return mgDomainHash;
}

/*
===============================================================================
Initialize a continuous energy hash and return a pointer to it
*/
MCGIDI::DomainHash * getCEHash(const int nBins)
{

  MCGIDI::DomainHash * ceDomainHash = new MCGIDI::DomainHash(nBins, 1e-8, 10 );

  return ceDomainHash;

}

