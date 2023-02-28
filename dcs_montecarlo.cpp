//============================================================================
// Name        : DCS_MonteCarlo.cpp
// Author      : Daniel Pinheiro, Pedro Amaro, César Godinho
// Version     : 2.0
// Copyright   : Your copyright notice
// Description : Entry point for the DCS simulation
//============================================================================

#include "dcs_montecarlo.hh"
<<<<<<< HEAD
#include "Util.h"
=======
#include <Util.h>
>>>>>>> d08f139 (Consolidation into Util file)

#include "input_settings.h" 

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
<<<<<<< HEAD
        else // Just use a path relative to the executable's location
        {
            strcat(Output_dir, input_params.output_dir_path.c_str());
=======
    }else{
        GraphOptionsInput.MakeDislin = false;
        //old unimplemented code?
        //Obtain_input_root();
    }


    if(UserSettingsInput.mask_C1 < 0 || UserSettingsInput.mask_C1 > 2){
        cout << "bad input for first crystal mask: " << UserSettingsInput.mask_C1 << endl;
        cout << "value of 0 for no mask, 1 for mask on the bottom and 2 for mask on the top" << endl;

        return 0;
    }

    if(UserSettingsInput.mask_C2 < 0 || UserSettingsInput.mask_C2 > 2){
            cout << "bad input for second crystal mask: " << UserSettingsInput.mask_C2 << endl;
            cout << "value of 0 for no mask, 1 for mask on the bottom and 2 for mask on the top" << endl;

            return 0;
    }


    if(!GraphOptionsInput.MakeDislin){
        GraphOptionsInput.make_graph_profile = false;
        GraphOptionsInput.make_image_plates = false;
        AnalysiesCrystaltiltsInput.make_graph_widths = false;
    }

    if(FullEnergySpectrumInput.make_more_lines == 1){
        if(!FullEnergySpectrumInput.Do_background){
            if(FullEnergySpectrumInput.p3_ener + FullEnergySpectrumInput.p2_ener + FullEnergySpectrumInput.p1_ener > 1.0){
                cout << "bad input for lines proportion: " << FullEnergySpectrumInput.p1_ener << " + " << FullEnergySpectrumInput.p2_ener << " + " << FullEnergySpectrumInput.p3_ener << " is greater than 1" << endl;

                return 0;
            }
        }

        reques_energ[0] = FullEnergySpectrumInput.linelamda1;
        reques_energ[1] = FullEnergySpectrumInput.linelamda2;
        reques_energ[2] = FullEnergySpectrumInput.linelamda3;
        reques_energ[3] = FullEnergySpectrumInput.linelamda4;

        reques_width[0] = FullEnergySpectrumInput.naturalwidth1;
        reques_width[1] = FullEnergySpectrumInput.naturalwidth2;
        reques_width[2] = FullEnergySpectrumInput.naturalwidth3;
        reques_width[3] = FullEnergySpectrumInput.naturalwidth4;

    }else if(FullEnergySpectrumInput.make_more_lines == 0){
        reques_energ[0] = linelamda;
        reques_energ[1] = FullEnergySpectrumInput.linelamda2;
        reques_energ[2] = FullEnergySpectrumInput.linelamda3;
        reques_energ[3] = FullEnergySpectrumInput.linelamda4;

        reques_width[0] = naturalwidth;
        reques_width[1] = FullEnergySpectrumInput.naturalwidth2;
        reques_width[2] = FullEnergySpectrumInput.naturalwidth3;
        reques_width[3] = FullEnergySpectrumInput.naturalwidth4;
    }else{
        cout << "Reading input energy spectrum..." << endl;

        Util::Read_EnergySpectrum();

        cout << "Input energy spectrum read." << endl;
    }

    if(GeometryInput.crystal_Si){
        d_lat = a_si_para / sqrt(pow(GeometryInput.imh, 2) + pow(GeometryInput.imk, 2) + pow(GeometryInput.iml, 2));
    }else{
        d_lat = a_Ge_para / sqrt(pow(GeometryInput.imh, 2) + pow(GeometryInput.imk, 2) + pow(GeometryInput.iml, 2));
    }

    if(FullEnergySpectrumInput.make_more_lines == 0 || FullEnergySpectrumInput.make_more_lines == 1){
        if(PhysicalParametersInput.Unit_energy == evv[0]){
            for(int i = 0; i < 4; i++){
                if(reques_energ[i] < 10.0){
                    cout << "bad input on the energies. requested energy less than 10 eV" << endl;

                    return 0;
                }
            }
        }else if(PhysicalParametersInput.Unit_energy == "A"){
            for(int i = 0; i < 4; i++){
                if(reques_energ[i] > 10.0){
                    cout << "bad input on the energies. requested energy more than 10 A" << endl;

                    return 0;
                }
            }
        }else{
            cout << "bad input on the energy unit: " << PhysicalParametersInput.Unit_energy << endl;

            return 0;
        }
    }else{
        bool usable;

        if(PhysicalParametersInput.Unit_energy == "keV"){
            usable = Util::CheckSpectrum("eV");

            if(! usable){
                cout << "bad input on the energies. requested energy spectrum will not be visible in output" << endl;

                return 0;
            }
        }else if(PhysicalParametersInput.Unit_energy == "eV"){
            usable = Util::CheckSpectrum("eV");

            if(! usable){
                cout << "bad input on the energies. requested energy spectrum will not be visible in output" << endl;

                return 0;
            }
        }else if(PhysicalParametersInput.Unit_energy == "A"){
            usable = Util::CheckSpectrum("A");

            if(! usable){
                cout << "bad input on the energies. requested energy spectrum will not be visible in output" << endl;

                return 0;
            }
        }else{
            cout << "bad input on the energy unit: " << PhysicalParametersInput.Unit_energy << endl;

            return 0;
>>>>>>> d08f139 (Consolidation into Util file)
        }
    }
    else
    {
        strcat(Output_dir, "output");
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
