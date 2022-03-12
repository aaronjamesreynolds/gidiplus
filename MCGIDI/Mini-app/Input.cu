#include "Mini-app.cuh"

/*
===============================================================================
Input class that parses command line arguments
*/
Input::Input(int argc, char **argv)
{
  int c, prev_ind, prev_opt;
  opterr = 0;
  const char * options_string = "n:c:b:t:s:phmk";

  while (prev_ind = optind, (c = getopt (argc, argv, options_string)) != -1)
  {
    // Catch an option being read as an argument
    if ( optind == prev_ind + 2 && *optarg == '-' ) 
    {
      prev_opt = c;
      c = '!';
    }
    switch (c)
    {
      // Number of lookups
      case 'N':
        {
          numDeviceLookups = atoi(optarg);
          break;
        }
      // Number of lookups
      case 'n':
        {
          numHostLookups = atoi(optarg);
          break;
        }

      // Number of batches to sample
      case 'b':
        {
          numBatches = atoi(optarg);
          break;
        }
      // Number of device threads
      case 't':
        {
          numThreads = atoi(optarg);
          break;
        }
      // Set continuous energy mode and number of hash bins
      case 'c':
        {
          mode = ce;
          numHashBins = atoi(optarg);
          break;
        }
      // Size of problem, which sets number of isotopes
      case 's':
        {
          char problem_size_arg = *optarg;
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
          break;
        }
      // Flag to print reaction data
      case 'm':
        {
          mode = mg;
          break;
        }

      // Flag to print reaction data
      case 'p':
        {
          printData = true;
          break;
        }
      // Flag sample products
      case 'k':
        {
          sampleProduct = true;
          break;
        }
      // Flag to print reaction data
      case 'h':
        {
          printUsage();
          exit(0);
          break;
        }
      // Catch missing arguments
      case '?':
        {
          if (optopt == 'N' || optopt == 'n' || optopt == 'b' || optopt == 't' 
              || optopt == 's' || optopt == 'c' )
              fprintf (stderr, "Option -%c requires an argument.\n", optopt);
            else if (isprint (optopt))
              fprintf (stderr, "Unknown option `-%c'.\n", optopt);
            else
              fprintf (stderr,
                  "Unknown option character `\\x%x'.\n",
                  optopt);
          abort();
        }
      case '!':          
        {
          fprintf (stderr, "Option -%c requires an argument.\n", prev_opt);
          abort();
        }
      default:
        abort ();
    }
  }
}

/*
===============================================================================
Print input options
*/
void Input::printInputOptions()
{

  printf("mode                                            = ");
  switch(mode)
  {
    case ce:
      printf("continuous energy (%d hash bins)\n", numHashBins);
      break;
    case mg:
      printf("multigroup (230-group data)\n");
      break;
  }
  printf("print protare data                              = %d\n",     printData);
  printf("sample reaction products                        = %d\n",     sampleProduct);
  printf("number of device XS lookups                     = %g\n",     static_cast<double>(numDeviceLookups));
  printf("number of host XS lookups                       = %g\n",     static_cast<double>(numHostLookups));
  printf("number of batches to sample                     = %d\n",     numBatches);
  printf("number of isotopes                              = %d\n",     numIsotopes);
  printf("number of threads/block in each kernel launch   = %d\n\n",   numThreads);

}

/*
===============================================================================
Print usage message
*/
void Input::printUsage()
{

  printf("\n=== USAGE ===\n\n");
  printf("-c (#)    hash bins in continuous energy mode   (default: %d)\n", 4000);
  printf("-m (flag) use multi-group mode                  (default: %d)\n", 1);
  printf("-n (#)    device XS lookups in a batch          (default: %g)\n", (double) 1000000000);
  printf("-n (#)    host XS lookups in a batch            (default: %g)\n", (double) 10000000);
  printf("-b (#)    batches to sample                     (default: %d)\n", 1);
  printf("-t (#)    threads/block in each kernel launch   (default: %d)\n", 512);
  printf("-s (s/l)  small/large problem size              (default: %c)\n", 's');
  printf("-p (flag) print protare data                    (default: %d)\n", 0);
  printf("-k (flag) sample reaction products              (default: %d)\n", 0);
  printf("-h (flag) print usage                           (default: %d)\n\n", 0);

}

/*
===============================================================================
Print logo
*/
void Input::printLogo()
{
  printf("\n");
  printf("+-+-+-+-+-+-+ +-+-+-+-+-+-+-+-+\n");
  printf("|M|C|G|I|D|I| |M|i|n|i|-|A|p|p|\n");
  printf("+-+-+-+-+-+-+ +-+-+-+-+-+-+-+-+\n");
  printf("\n");
  printf("Use option -h for usage\n\n");
}
