To build the mini-app, first build gidi-plus with nvcc using the following command: 

  make CXX=nvcc CXXFLAGS="-x cu --relocatable-device-code=true -O3 -std=c++14 -gencode=arch=compute\_70,code=sm\_70 -v"

Then navigate to this directory and do: 
  
  make
