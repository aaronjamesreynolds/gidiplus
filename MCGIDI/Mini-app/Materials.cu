/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

/*
Brief: Material initialization
Author: Aaron James Reynolds 
(pick_mat function and nuclide indices credited to XSBench)
*/

#include "Mini-app.cuh"

/*
===============================================================================
Sample a material from the H-M reactor (Taken from XSBench)

Sampling distribution taken from material volume fractions of the H-M reactor
*/
MCGIDI_HOST_DEVICE int pick_mat(uint64_t * seed) 
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
===============================================================================
Initialize pseudo material compositions from H-M reactor. 

The protares that compose each material are not physical. Rather, each 
material is composed of the correct number of constituent protares. which
is expected to reproduce the cost of XS lookup.
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
  // (the XSBench indices are based on the alphabetic order of the nuclide
  // identifiers used in the H-M reactor)
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

  switch(size)
  {
    case large:
      printf("Loaded material compositions for large variant of H-M reactor. \n");
      break;
    case small:
      printf("Loaded material compositions for small variant of H-M reactor. \n");
      break;
  }

  return materialCompositions;
}

/*
===============================================================================
Initialize the number densities for each constituent in each material

For now, just assigning dummy values of 10.0 for every constituent.
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
===============================================================================
Calculate average number of energy gridpoints per cross section lookup

*/
double calcAvgNumberOfGridpointsPerLookup(
    std::vector<MCGIDI::Protare *> protares,
    int  *  materialComposition,
    int     maxNumberIsotopes)
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

  double materialAvgNumberOfGridpoints[12];
  int numberOfGridpoints, numIsotopes;

  // Calculate average number of gridpoints for protare in a material
  for (int matIndex = 0; matIndex < 12; matIndex++)
  {

    // Initialize accumulators and loop variables
    numberOfGridpoints = 0;
    numIsotopes = 0;
    int isoIndex = -1;
    MCGIDI::ProtareSingle * MCProtare; 

    // Accumulate total number of gridpoints in each constituent
    for (int iConstituent = 0; 
        materialComposition[matIndex * maxNumberIsotopes + iConstituent] >= 0 
        && iConstituent < maxNumberIsotopes; 
        iConstituent++)
    {

      numIsotopes += 1;
      isoIndex      = materialComposition[matIndex * maxNumberIsotopes + iConstituent];
      MCProtare = reinterpret_cast<MCGIDI::ProtareSingle *>(protares[isoIndex]);
      MCGIDI::HeatedCrossSectionContinuousEnergy * xs = MCProtare->heatedCrossSections().heatedCrossSections()[0];
      numberOfGridpoints += xs->energies().size();

    } // iConstituent

    materialAvgNumberOfGridpoints[matIndex] = double(numberOfGridpoints) / double(numIsotopes);

  } // matIndex

  // Calculate average number of gridpoints per lookup
  double avgNumberGridpointsPerLookup = 0;
  for (int matIndex = 0; matIndex < 12; matIndex++)
  {

    avgNumberGridpointsPerLookup += dist[matIndex] * materialAvgNumberOfGridpoints[matIndex];

  } // matIndex

  return avgNumberGridpointsPerLookup;

}

