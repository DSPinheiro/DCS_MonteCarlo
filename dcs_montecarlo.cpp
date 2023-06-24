//#ifndef LIB_DEF
//#pragma comment(linker, "/SUBSYSTEM:windows /ENTRY:mainCRTStartup")
//#endif

//============================================================================
// Name        : DCS_MonteCarlo.cpp
// Author      : Daniel Pinheiro, Pedro Amaro, César Godinho
// Version     : 2.0
// Copyright   : Your copyright notice
// Description : Entry point for the DCS simulation
//============================================================================

#include "dcs_montecarlo.hh"
#include "Util.h"
#include <QtWidgets/qmessagebox.h>

#include "input_settings.h" 

#include <filesystem>


using namespace std;

int ENTRYPOINT(int argc, char *argv[]){

    QApplication a(argc, argv);

    Unit_energy = PhysicalParametersInput.Unit_energy;

    // This is my proposal (César) Just use an argument on the standalone executable instead of a .path file
    // The new version does not require the input file
    // However, there are a few option that need to be added to the GUI before we can remove this feature
    // Also just leaving for compatibility is ok I believe
    // if(!configure(argc, argv)) return 0;

    // DCS_GUI w;
    GUISettingsWindow w;
    InputSettingsPrompt input_window(&w);
    input_window.show();
    // w.show();
    return a.exec();
}
