# DCSsimu
Simulation program for a Double Crystal Spectrometer.
The simulation is performed using a Monte Carlo method with full geometric considerations.
This is a close to direct translation in C++ of the original program written in FORTRAN, we are still cleaning up, so some things might seem wierd.

Programed using Qt5.15.1.

After compiling the project add the file "DCrystal_input.path" next to the compiled executable and configure the absolute path to the folder simulacoesDCS on your system. The location of the .path file is the only hardcoded path in the program, everything else should be relative to the simulacoesDCS folder.

The new OpenGL view will show you a 3D representation of the positions of all simulated elements and their motions during the simulation. You can zoom, rotate and pan around the OpenGL view.
