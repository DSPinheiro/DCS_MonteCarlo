//============================================================================
// Name        : DCS_MonteCarlo.cpp
// Author      : Daniel Pinheiro, Pedro Amaro, César Godinho
// Version     : 2.0
// Copyright   : Your copyright notice
// Description : Entry point for the DCS simulation
//============================================================================

#include "../include/dcs_montecarlo.hh"
#include "../include/Util.h"

#include "../include/input_settings.h" 

#include "configure.inl"

#include <filesystem>


using namespace std;

int ENTRYPOINT(int argc, char *argv[]){
#ifndef QT_EXISTS
    CommandLineParams input_params = command_line_parser(argc, argv);
    if(!input_params.valid) return 0;

    Unit_energy = PhysicalParametersInput.Unit_energy;

    // Setup workspace dir
    if(!input_params.input_dir_path.empty())
    {
        strcpy(File_simu, input_params.input_dir_path.c_str());
    }
    else // Else try and use the default workspace name
    {
        input_params.input_dir_path = "simuWorkspace";
        strcpy(File_simu, "simuWorkspace");
    }
    //std::cout << "Path to simulation workspace: " << File_simu << "\n\n" << std::endl;
    
    char inFile[1024];
    // Setup input config file name
    strcpy(inFile, File_simu);
    if(!input_params.input_config_file_path.empty())
    {
        strcat(inFile, "/");
        strcat(inFile, input_params.input_config_file_path.c_str());
    }
    else // Else try and use the default filename
    {
        strcat(inFile, "/DCrystal_input.input");
    }

    // Configure the output directory
    if(!input_params.output_dir_path.empty())
    {
        
        if(std::filesystem::path(input_params.output_dir_path).is_relative())
        {
            strcat(Output_dir, File_simu);
            strcat(Output_dir, "/");
            strcat(Output_dir, input_params.output_dir_path.c_str());
        }
        else // Just use a path relative to the executable's location
        {
            strcat(Output_dir, input_params.output_dir_path.c_str());
        }
    }
    else
    {
        strcat(Output_dir, "output/");
    }

    if(!filesystem::is_directory(Output_dir) || !filesystem::exists(Output_dir))
    {
        filesystem::create_directories(Output_dir);
    }


    if(InputSettingsPrompt::configure(inFile) != 0) return 0;

    // Make aditional configurations
    Unit_energy = PhysicalParametersInput.Unit_energy;
    if(GeometryInput.imh == 2 and GeometryInput.imk == 2 and GeometryInput.iml == 2)
    {
        refra_corr = refra_corrNIST;
    }
    else
    {
        refra_corr = refra_corrPARIS;
    }

    FullEnergySpectrumInput.energy_spectrum_file = input_params.input_dir_path + "/Energy_spectrum.txt";

    SimulationInterface::headlessSimu();
#else
    QApplication a(argc, argv);

    Unit_energy = PhysicalParametersInput.Unit_energy;

    // DCS_GUI w;
    GUISettingsWindow w;
    InputSettingsPrompt input_window(&w);
    input_window.show();
    // w.show();
    return a.exec();
#endif
}
