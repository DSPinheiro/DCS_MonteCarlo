# DCSsimu
Simulation program for a Double Crystal Spectrometer.
The simulation is performed using a Monte Carlo method with full geometric considerations.
This is a close to direct translation in C++ of the original program written in FORTRAN so some things might seem wierd.

You can now set the output file directories, the workspace location and set the name of the input configuration file you want to use.
A new file save system is present which serializes the GUI settings fields out on to a `.dsp` file, which can be loaded using the *Load* and *Save* buttons on the new GUI.

The new OpenGL view will show you a 3D representation of the positions of all simulated elements and their motions during the simulation. You can zoom, rotate and pan around the OpenGL view.

## How to build/install
```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release --target install
```

> For linux replace the last command by: `make install`

## Changelog

**20/07/2023**\
Fixed some issues with the crystal reflection profiles.\
Now we can use the results to compare more directly with experiment and across simulations.\

**11/07/2023**\
GPU Accelerated Simulation has been implemented.\
Each bin can now be executed in the GPU (Nvidia) using CUDA libraries.\
The device code is compiled as a seperate static library but that might change.\

**03/07/2023**\
Major main loop refactoring to simulate the bin in a seperate function.\
Input parameters explanation.\

**26/06/2023**\
Added OpenMP support for event generation.\
The user can now specify a number of threads to use for the simulation.\

**23/06/2023**\
Merge standalone and library branches.\
Small fixes.\

**22/05/2023**\
Add new GUI to the simulation.\
Add support to C++17. Add as a module to the DCS control and software acquisition.\
Support to the `DCrystal_input.path` was removed.\

**18/05/2023**\
Add new save file format (Easier to manage).\
Add standalone run arguments.\
Add a library target.\
Add Cmake compatibility.
