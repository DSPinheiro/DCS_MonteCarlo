# DCSsimu
Simulation program for a Double Crystal Spectrometer.
The simulation is performed using a Monte Carlo method with full geometric considerations.
This is a close to direct translation in C++ of the original program written in FORTRAN so some things might seem wierd.

Programed using the Qt5 Creator, so the files are not split in src and header folders.

To define your workspace folder (which might contain DCrystal_input.input) specify the `-w` option.
Type `DCSsimu -h` for a list of commands.
You can now set the output file directories, the workspace location and set the name of the input configuration file you want to use.
A new file save system is present which serializes the GUI settings fields out on to a `.dsp` file, which can be loaded using the *Load* and *Save* buttons on the new GUI.

The new OpenGL view will show you a 3D representation of the positions of all simulated elements and their motions during the simulation. You can zoom, rotate and pan around the OpenGL view.

## Changelog

**22/05/2023**\
Add new GUI to the simulation.\
Add support to C++17. Add as a module to the DCS control and software acquisition.\
Support to the `DCrystal_input.path` was removed.\

**18/05/2023**\
Add new save file format (Easier to manage).\
Add standalone run arguments.\
Add a library target.\
Add Cmake compatibility.