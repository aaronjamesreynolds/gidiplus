
#ifndef __XSBENCH_HEADER_H__
#define __XSBENCH_HEADER_H__

#include <iostream>
#include <variant>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>  
#include <math.h>
#include "MCGIDI.hpp"
#include <sys/time.h>
#include <chrono>
#include <thrust/reduce.h>
#include <boost/variant.hpp>
#if defined(_OPENMP)
  #include <omp.h>
#endif

#define STARTING_SEED 1070

// Initialize vector containing isotope names

enum HM_size {large, small};
enum energyMode {ce, mg};
struct LookupRate_t
{
  double meanTime = 0;
  double meanTimeStdDev = 0;
  double totalTime= 0;
  double lookupRate = 0;
  int numBatches = 0;
  int numLookups = 0;
  std::string tag = "";

  void print()
  {
    printf("Looked up %d * %g XSs in %g seconds \n", 
        numBatches, 
        static_cast<double>(numLookups), 
        totalTime);
    if (numBatches == 1)
      printf("%s look-up rate: %g XSs per second \n\n", 
          tag.c_str(),
          lookupRate);
    else
    {
      printf("%s look-up rate: %g (+/- %g%%) XSs per second \n\n", 
          tag.c_str(),
          lookupRate,
          100 * meanTimeStdDev / meanTime );
    }
  }
};

// Input.cu
class Input
{
  public:

    // Constructor
    Input(int argc, char **argv);

    // Functions
    void printLogo();
    void printInputOptions();
    void printUsage();

    // Default options for command line arguments
    int numLookups    = 1E9;                  // n Number of XSs to lookup
    int numBatches    = 1;                    // b Number of batches
    int numThreads    = 512;                  // t Number of threads in kernel launch
    // write out data, 2 - Read in data and compare
    HM_size problem_size = small;             // s small has 34 fuel nuclides, large has ~300 fuel nuclides                                
    int numIsotopes   = 68;                   // Number of isotopes
    int numOMPThreads = 1;                    // j default number of OMP threads
    energyMode mode = ce;                     // s small has 34 fuel nuclides, large has ~300 fuel nuclides                                
    int numHashBins = 4000;                   // number of hash bins when running in CE mode
    bool sampleProduct = false;               // k flag to sample products after XS lookup
    bool printData     = false;               // p Print reaction data
    const char *isotopeNames[500] = {
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

};

template <class myType>
void unwrapFrom2Dto1D(std::vector<std::vector<myType>> a, myType *b, int rows, int cols) 
{

  for (int iRow = 0; iRow < rows; iRow++)
  {
    for (int iCol = 0; iCol < cols; iCol++)
    {
      b[iRow * cols + iCol] = a[iRow][iCol];
    }
  }

}



// ProtareInit.cu functions
class TallyProductHandler : public MCGIDI::Sampling::ProductHandler {

    public:
        int dummyCounter;

        MCGIDI_HOST_DEVICE TallyProductHandler( ) {}
        MCGIDI_HOST_DEVICE ~TallyProductHandler( ) {}

        MCGIDI_HOST_DEVICE std::size_t size( ) { return 0; }
        MCGIDI_HOST_DEVICE void clear( ) {}

        MCGIDI_HOST_DEVICE void push_back( MCGIDI::Sampling::Product &a_product ) 
        {
          dummyCounter += 1;
        }

};
std::vector<MCGIDI::Protare *> initMCProtares(
    int        numIsotopes, 
    const char *isotopeNames[],
    energyMode mode,
    int numHashIns);
void printReactionData(std::vector<MCGIDI::Protare *> protares);
std::vector<char *> copyProtaresFromHostToDevice(
    std::vector<MCGIDI::Protare *> protares);
__global__ void setUp( int a_numIsotopes, MCGIDI::DataBuffer **a_buf );
MCGIDI::MultiGroupHash * getMGHash(const char *isotopeNames[]);
MCGIDI::DomainHash * getCEHash(const int nBins = 4000);

// XSCalc.cu functions
template <typename T>
__global__ void calcScatterMacroXSs(
    char   **deviceProtares,
    T      *domainHash,
    int    *materialComposition,
    double *numberDensities,
    double *verification,
    int     maxNumberIsotopes,
    int     numCollisions,
    bool    sampleProduct = false);
template <typename T>
__global__ void calcTotalMacroXSs(
    char   **deviceProtares,
    T      *domainHash,
    int    *materialComposition,
    double *numberDensities,
    double *verification,
    int     maxNumberIsotopes,
    int     numCollisions,
    bool    sampleProduct = false);
template <typename T>
void calcScatterMacroXSs(
    std::vector<MCGIDI::Protare *> protares,
    T      *domainHash,
    int    *materialComposition,
    double *numberDensities,
    double *verification,
    int     maxNumberIsotopes,
    int     numLookups,
    bool    sampleProduct = false);
template <typename T>
void calcTotalMacroXSs(
    std::vector<MCGIDI::Protare *> protares,
    T      *domainHash,
    int    *materialComposition,
    double *numberDensities,
    double *verification,
    int     maxNumberIsotopes,
    int     numLookups,
    bool    sampleProduct = false);
MCGIDI_HOST_DEVICE TallyProductHandler sampleProducts(
    MCGIDI::ProtareSingle *MCProtarei,
    int hashIndex,
    double temperature,
    double energy, 
    double crossSection,
    uint64_t * seed);
MCGIDI_HOST_DEVICE TallyProductHandler sampleProducts(
    MCGIDI::Protare *MCProtare,
    int hashIndex,
    double temperature,
    double energy, 
    double crossSection,
    uint64_t * seed);

// Materials.cu functions
MCGIDI_HOST_DEVICE int pick_mat(uint64_t * seed);
std::vector< std::vector<int> > initMaterialCompositions(HM_size size);
std::vector< std::vector<double> > initNumberDensities(
    std::vector< std::vector<int> > materialCompositions);

// Utils.cu functions
MCGIDI_HOST_DEVICE double myRNG( uint64_t *seed );
MCGIDI_HOST_DEVICE double LCG_random_double(uint64_t * seed);
MCGIDI_HOST_DEVICE uint64_t fast_forward_LCG(uint64_t seed, uint64_t n);
bool approximatelyEqual(double *a,  double *b, int size, double epsilon);
bool approximatelyEqual(double a, double b, double epsilon);
double get_time();
LookupRate_t calcLookupRate(std::vector<double> edge_times, 
    int numLookups,
    int numBatches,
    std::string tag);
void setCudaOptions();

#endif
