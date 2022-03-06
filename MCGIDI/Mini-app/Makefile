#===============================================================================
# User Options
#===============================================================================

COMPILER    = nvidia
OPTIMIZE    = yes
DEBUG       = no
PROFILE     = no
SM_VERSION  = 70

#===============================================================================
# Program name & source code list
#===============================================================================

program = mcgidi-mini-app

source = \
Mini-app.cu \
Input.cu \
Materials.cu \
XSCalc.cu \
ProtaresInit.cu \
Utils.cu 

#===============================================================================
# Sets Flags
#===============================================================================

# Get paths to GIDI dependencies 
GIDI_PLUS_PATH ?= $(abspath ../..)
include $(GIDI_PLUS_PATH)/Makefile.paths
local_CXXFLAGS += -I$(GIDI_PATH)/Test/Utilities -I$(MCGIDI_PATH)/Test/Utilities
local_CXXFLAGS += $(LIB_PATHS) -L$(GIDI_PATH)/Test/Utilities -L$(MCGIDI_PATH)/Test/Utilities

# Standard Flags
CFLAGS := $(local_CXXFLAGS)
LIBS =  -lGIDI_testUtilities -lMCGIDI_testUtilities -lMCGIDI -lGIDI -lPoPI -lHAPI -lLUPI \
        -lnumericalfunctions -lstatusMessageReporting -lpugixml -lexpat -lm $(HDF5_LIBS)

# Linker Flags
LDFLAGS = -lm

# NVIDIA Compiler
ifeq ($(COMPILER),nvidia)
  CC = nvcc
  CFLAGS += -x cu --relocatable-device-code=true -std=c++14
  CFLAGS += -gencode=arch=compute_70,code=sm_70  
  CFLAGS += -Xcompiler -fopenmp
endif

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -G -lineinfo
  LDFLAGS += -g -G -lineinfo
endif

# Profiling Flags
ifeq ($(PROFILE),yes)
  CFLAGS += -pg
  LDFLAGS += -pg
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(source) 
	$(CC) $(CFLAGS) $(source) -o $@ $(LIBS)

clean:
	rm -rf $(program) 

