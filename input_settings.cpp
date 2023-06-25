#include "input_settings.h"
#include <fstream>
#include <filesystem>
#include <iostream>

#ifdef QT_EXISTS
#include "ui_input_settings_prompt.h"
#include <QPushButton>
#include <QFileDialog>
#include <QMessageBox>
#endif

using namespace std;

int InputSettingsPrompt::configure(const std::string& inFile)
{
    // Parse the input file
    ifstream inputFile(inFile);
    string line;
    if(inputFile.is_open()){
        char str[1];
        char* firstChar = str;
        inputFile.read(firstChar, 1);
        inputFile.seekg(0);

        string currStruct = "";

        #ifdef QT_EXISTS

            QString message = "Path to simulation workspace: ";
            message.append(File_simu);
            QMessageBox::information(nullptr, message, tr("An input file with the experimental setup parameters has been found!"), QMessageBox::Ok);
        #else
            cout << "An input file with the C++ configuration model has been found." << endl;
        #endif

        while (getline(inputFile, line)){
            if (line.size() < 1) continue;
            if(line[0] != '/' && line[1] != '/'){

                if(line.find("Geometry") != string::npos)
                    currStruct = "Geometry";
                else if(line.find("UserSettings") != string::npos)
                    currStruct = "UserSettings";
                else if(line.find("GeoParapathlenghts") != string::npos)
                    currStruct = "GeoParapathlenghts";
                else if(line.find("Geolenghtelemets") != string::npos)
                    currStruct = "Geolenghtelemets";
                else if(line.find("GeoParameters") != string::npos)
                    currStruct = "GeoParameters";
                else if(line.find("CurveVerticalTilt") != string::npos)
                    currStruct = "CurveVerticalTilt";
                else if(line.find("Graph_options") != string::npos)
                    currStruct = "Graph_options";
                else if(line.find("plotparameters") != string::npos)
                    currStruct = "plotparameters";
                else if(line.find("numberrays") != string::npos)
                    currStruct = "numberrays";
                else if(line.find("physical_parameters") != string::npos)
                    currStruct = "physical_parameters";
                else if(line.find("polarization_parameters") != string::npos)
                    currStruct = "polarization_parameters";
                else if(line.find("temperature_parameters") != string::npos)
                    currStruct = "temperature_parameters";
                else if(line.find("fullenergyspectrum") != string::npos)
                    currStruct = "fullenergyspectrum";
                else if(line.find("Curved_Crystal") != string::npos)
                    currStruct = "Curved_Crystal";
                else if(line.find("ParallelSettings") != string::npos)
                    currStruct = "ParallelSettings";
                else if(line.find("AnalysiesCrystaltilts") != string::npos)
                    currStruct = "AnalysiesCrystaltilts";



                if(currStruct == "Geometry"){

                    string elem = split(line, "//")[0];
                    vector<string> items = split(elem, "=");

                    trim(items[0]);

                    if(items[0] == "mode_bragg_geo"){
                        trim(items[1]);
                        GeometryInput.mode_bragg_geo = (items[1] == ".true.");
                    }else if(items[0] == "imh"){
                        trim(items[1]);
                        GeometryInput.imh = stoi(items[1]);
                    }else if(items[0] == "imk"){
                        trim(items[1]);
                        GeometryInput.imk = stoi(items[1]);
                    }else if(items[0] == "iml"){
                        trim(items[1]);
                        GeometryInput.iml = stoi(items[1]);
                    }else if(items[0] == "crystal_Si"){
                        trim(items[1]);
                        GeometryInput.crystal_Si = (items[1] == ".true.");
                    }

                }else if(currStruct == "UserSettings"){

                    string elem = split(line, "//")[0];
                    vector<string> items = split(elem, "=");

                    trim(items[0]);

                    if(items[0] == "see_para"){
                        trim(items[1]);
                        UserSettingsInput.see_para = (items[1] == ".true.");
                    }else if(items[0] == "see_anti"){
                        trim(items[1]);
                        UserSettingsInput.see_anti = (items[1] == ".true.");
                    }else if(items[0] == "Make_Vertical"){
                        trim(items[1]);
                        UserSettingsInput.Make_Vertical = (items[1] == ".true.");
                    }else if(items[0] == "Make_Horizontal"){
                        trim(items[1]);
                        UserSettingsInput.Make_Horizontal = (items[1] == ".true.");
                    }else if(items[0] == "angle_aprox"){
                        trim(items[1]);
                        UserSettingsInput.angle_aprox = stoi(items[1]);
                    }else if(items[0] == "fitting"){
                        trim(items[1]);
                        UserSettingsInput.fitting = (items[1] == ".true.");
                    }else if(items[0] == "TrueVoigt"){
                        trim(items[1]);
                        UserSettingsInput.TrueVoigt = (items[1] == ".true.");
                    }else if(items[0] == "Simple_simu"){
                        trim(items[1]);
                        UserSettingsInput.Simple_simu = (items[1] == ".true.");
                    }else if(items[0] == "center_1crys"){
                        trim(items[1]);
                        UserSettingsInput.center_1crys = (items[1] == ".true.");
                    }else if(items[0] == "center_2crys"){
                        trim(items[1]);
                        UserSettingsInput.center_2crys = (items[1] == ".true.");
                    }else if(items[0] == "mask_C1"){
                        trim(items[1]);
                        UserSettingsInput.mask_C1 = stoi(items[1]);
                    }else if(items[0] == "mask_C2"){
                        trim(items[1]);
                        UserSettingsInput.mask_C2 = stoi(items[1]);
                    }else if(items[0] == "print_scan"){
                        trim(items[1]);
                        UserSettingsInput.print_scan = (items[1] == ".true.");
                    }else if(items[0] == "center_Mask"){
                        trim(items[1]);
                        UserSettingsInput.center_Mask = (items[1] == ".true.");
                    }else if(items[0] == "make_mask_test"){
                        trim(items[1]);
                        UserSettingsInput.make_mask_test = (items[1] == ".true.");
                    }

                }else if(currStruct == "GeoParapathlenghts"){

                    string elem = split(line, "//")[0];
                    vector<string> items = split(elem, "=");

                    trim(items[0]);

                    if(items[0] == "type_source"){
                        trim(items[1]);
                        GeoParapathlengthsInput.type_source = split(items[1], "\"")[1];
                    }else if(items[0] == "LT_aper"){
                        trim(items[1]);
                        GeoParapathlengthsInput.LT_aper = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "dist_T_Cr1"){
                        trim(items[1]);
                        GeoParapathlengthsInput.dist_T_Cr1 = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "dist_Cr1_Cr2"){
                        trim(items[1]);
                        GeoParapathlengthsInput.dist_Cr1_Cr2 = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "dist_Cr2_Det"){
                        trim(items[1]);
                        GeoParapathlengthsInput.dist_Cr2_Det = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "dist_aper_det"){
                        trim(items[1]);
                        GeoParapathlengthsInput.dist_aper_det = stod(split(items[1], "d0")[0]);
                    }

                }else if(currStruct == "Geolenghtelemets"){

                    string elem = split(line, "//")[0];
                    vector<string> items = split(elem, "=");

                    trim(items[0]);

                    if(items[0] == "S_aper"){
                        trim(items[1]);
                        GeolengthelementsInput.S_aper = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "S_aper_var"){
                        trim(items[1]);
                        GeolengthelementsInput.S_aper_var = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "S_sour"){
                        trim(items[1]);
                        GeolengthelementsInput.S_sour = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "y_sour"){
                        trim(items[1]);
                        GeolengthelementsInput.y_sour = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "z_sour"){
                        trim(items[1]);
                        GeolengthelementsInput.z_sour = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "y_aper"){
                        trim(items[1]);
                        GeolengthelementsInput.y_aper = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "z_aper"){
                        trim(items[1]);
                        GeolengthelementsInput.z_aper = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "S_shi_hor_B"){
                        trim(items[1]);
                        GeolengthelementsInput.S_shi_hor_B = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "S_shi_hor_A"){
                        trim(items[1]);
                        GeolengthelementsInput.S_shi_hor_A = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "S_shi_ver_B"){
                        trim(items[1]);
                        GeolengthelementsInput.S_shi_ver_B = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "S_shi_ver_A"){
                        trim(items[1]);
                        GeolengthelementsInput.S_shi_ver_A = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "y_first_crys"){
                        trim(items[1]);
                        GeolengthelementsInput.y_first_crys = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "z_first_crys"){
                        trim(items[1]);
                        GeolengthelementsInput.z_first_crys = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "ydetc"){
                        trim(items[1]);
                        GeolengthelementsInput.ydetc = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "zdetc"){
                        trim(items[1]);
                        GeolengthelementsInput.zdetc = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "shift_det_ver"){
                        trim(items[1]);
                        GeolengthelementsInput.shift_det_ver = stod(split(items[1], "d0")[0]);
                    }

                }else if(currStruct == "GeoParameters"){

                    string elem = split(line, "//")[0];
                    vector<string> items = split(elem, "=");

                    trim(items[0]);

                    if(items[0] == "Exp_crys1"){
                        trim(items[1]);
                        GeoParametersInput.Exp_crys1 = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "teta_table"){
                        trim(items[1]);
                        GeoParametersInput.teta_table = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "OffsetRotCry1"){
                        trim(items[1]);
                        GeoParametersInput.OffsetRotCry1 = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "teta_detec_para"){
                        trim(items[1]);
                        GeoParametersInput.teta_detec_para = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "teta_detec_anti"){
                        trim(items[1]);
                        GeoParametersInput.teta_detec_anti = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "tilt_C1"){
                        trim(items[1]);
                        GeoParametersInput.tilt_C1 = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "tilt_C2"){
                        trim(items[1]);
                        GeoParametersInput.tilt_C2 = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "xsi"){
                        trim(items[1]);
                        GeoParametersInput.xsi = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "center_1cry_at"){
                        trim(items[1]);
                        GeoParametersInput.center_1cry_at = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "S_shi_ver_B"){
                        trim(items[1]);
                        GeoParametersInput.center_2cry_at = stod(split(items[1], "d0")[0]);
                    }

                }else if(currStruct == "CurveVerticalTilt"){

                    string elem = split(line, "//")[0];
                    vector<string> items = split(elem, "=");

                    trim(items[0]);

                    if(items[0] == "make_CurveTilt"){
                        trim(items[1]);
                        CurveVerticalTiltInput.make_CurveTilt = (items[1] == ".true.");
                    }else if(items[0] == "phas_tilt1"){
                        trim(items[1]);
                        CurveVerticalTiltInput.phas_tilt1 = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "phas_tilt2"){
                        trim(items[1]);
                        CurveVerticalTiltInput.phas_tilt2 = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "offsettilt1"){
                        trim(items[1]);
                        CurveVerticalTiltInput.offsettilt1 = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "offsettilt2"){
                        trim(items[1]);
                        CurveVerticalTiltInput.offsettilt2 = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "consttilt1"){
                        trim(items[1]);
                        CurveVerticalTiltInput.consttilt1 = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "consttilt2"){
                        trim(items[1]);
                        CurveVerticalTiltInput.consttilt2 = stod(split(items[1], "d0")[0]);
                    }

                }else if(currStruct == "Graph_options"){

                    string elem = split(line, "//")[0];
                    vector<string> items = split(elem, "=");

                    trim(items[0]);

                    if(items[0] == "make_graph_profile"){
                        trim(items[1]);
                        GraphOptionsInput.make_graph_profile = (items[1] == ".true.");
                    }else if(items[0] == "make_image_plates"){
                        trim(items[1]);
                        GraphOptionsInput.make_image_plates = (items[1] == ".true.");
                    }else if(items[0] == "make_imageC1_After_refle"){
                        trim(items[1]);
                        GraphOptionsInput.make_imageC1_After_refle = (items[1] == ".true.");
                    }else if(items[0] == "make_imageC2_After_refle"){
                        trim(items[1]);
                        GraphOptionsInput.make_imageC2_After_refle = (items[1] == ".true.");
                    }

                }else if(currStruct == "plotparameters"){

                    string elem = split(line, "//")[0];
                    vector<string> items = split(elem, "=");

                    trim(items[0]);

                    if(items[0] == "delta_angl"){
                        trim(items[1]);
                        PlotParametersInput.delta_angl = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "shift_disp_window"){
                        trim(items[1]);
                        PlotParametersInput.shift_disp_window = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "nubins"){
                        trim(items[1]);
                        PlotParametersInput.nubins = stoi(items[1]);
                    }

                }else if(currStruct == "numberrays"){

                    string elem = split(line, "//")[0];
                    vector<string> items = split(elem, "=");

                    trim(items[0]);

                    if(items[0] == "nbeams"){
                        trim(items[1]);
                        NumberRaysInput.nbeams = stoi(items[1]);
                    }else if(items[0] == "number_rotati"){
                        trim(items[1]);
                        NumberRaysInput.number_rotati = stoi(items[1]);
                    }else if (items[0] == "number_graph_events") {
                        trim(items[1]);
                        NumberRaysInput.number_events = stoi(items[1]);
                    }

                }else if(currStruct == "physical_parameters"){

                    string elem = split(line, "//")[0];
                    vector<string> items = split(elem, "=");

                    trim(items[0]);

                    if(items[0] == "Unit_energy"){
                        trim(items[1]);
                        PhysicalParametersInput.Unit_energy = split(items[1], "\"")[1];
                    }else if(items[0] == "linelamda"){
                        trim(items[1]);
                        PhysicalParametersInput.linelamda = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "naturalwidth"){
                        trim(items[1]);
                        PhysicalParametersInput.naturalwidth = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "gauss_Doop"){
                        trim(items[1]);
                        PhysicalParametersInput.gauss_Doop = stod(split(items[1], "d0")[0]);
                    }

                }else if(currStruct == "polarization_parameters"){

                    string elem = split(line, "//")[0];
                    vector<string> items = split(elem, "=");

                    trim(items[0]);

                    if(items[0] == "mka_poli"){
                        trim(items[1]);
                        PolarizationParametersInput.mka_poli = (items[1] == ".true.");
                    }else if(items[0] == "relationP_S"){
                        trim(items[1]);
                        PolarizationParametersInput.relationP_S = stod(split(items[1], "d0")[0]);
                    }

                }else if(currStruct == "temperature_parameters"){

                    string elem = split(line, "//")[0];
                    vector<string> items = split(elem, "=");

                    trim(items[0]);

                    if(items[0] == "T_crystal_1_para"){
                        trim(items[1]);
                        TemperatureParametersInput.T_crystal_1_para = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "T_crystal_1_anti"){
                        trim(items[1]);
                        TemperatureParametersInput.T_crystal_1_anti = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "T_crystal_2_para"){
                        trim(items[1]);
                        TemperatureParametersInput.T_crystal_2_para = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "T_crystal_2_anti"){
                        trim(items[1]);
                        TemperatureParametersInput.T_crystal_2_anti = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "mk_temp_bin"){
                        trim(items[1]);
                        TemperatureParametersInput.mk_temp_bin = (items[1] == ".true.");
                    }else if(items[0] == "AA_tempera"){
                        trim(items[1]);
                        TemperatureParametersInput.AA_tempera = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "TT_tempera"){
                        trim(items[1]);
                        TemperatureParametersInput.TT_tempera = stod(split(items[1], "d0")[0]);
                    }

                }else if(currStruct == "fullenergyspectrum"){

                    string elem = split(line, "//")[0];
                    vector<string> items = split(elem, "=");

                    trim(items[0]);

                    if(items[0] == "make_more_lines"){
                        trim(items[1]);
                        FullEnergySpectrumInput.make_more_lines = stoi(items[1]);
                    }else if(items[0] == "linelamda1"){
                        trim(items[1]);
                        FullEnergySpectrumInput.linelamda1 = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "naturalwidth1"){
                        trim(items[1]);
                        FullEnergySpectrumInput.naturalwidth1 = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "p1_ener"){
                        trim(items[1]);
                        FullEnergySpectrumInput.p1_ener = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "linelamda2"){
                        trim(items[1]);
                        FullEnergySpectrumInput.linelamda2 = (items[1] == ".true.");
                    }else if(items[0] == "naturalwidth2"){
                        trim(items[1]);
                        FullEnergySpectrumInput.naturalwidth2 = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "p2_ener"){
                        trim(items[1]);
                        FullEnergySpectrumInput.p2_ener = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "linelamda3"){
                        trim(items[1]);
                        FullEnergySpectrumInput.linelamda3 = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "naturalwidth3"){
                        trim(items[1]);
                        FullEnergySpectrumInput.naturalwidth3 = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "p3_ener"){
                        trim(items[1]);
                        FullEnergySpectrumInput.p3_ener = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "linelamda4"){
                        trim(items[1]);
                        FullEnergySpectrumInput.linelamda4 = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "naturalwidth4"){
                        trim(items[1]);
                        FullEnergySpectrumInput.naturalwidth4 = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "Do_background"){
                        trim(items[1]);
                        FullEnergySpectrumInput.Do_background = (items[1] == ".true.");
                    }

                }else if(currStruct == "Curved_Crystal"){

                    string elem = split(line, "//")[0];
                    vector<string> items = split(elem, "=");

                    trim(items[0]);

                    if(items[0] == "Curve_crystall"){
                        trim(items[1]);
                        CurvedCrystalInput.Curve_crystall = (items[1] == ".true.");
                    }else if(items[0] == "R_cur_crys_1"){
                        trim(items[1]);
                        CurvedCrystalInput.R_cur_crys_1 = stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "R_cur_crys_2"){
                        trim(items[1]);
                        CurvedCrystalInput.R_cur_crys_2 = stod(split(items[1], "d0")[0]);
                    }

                }else if(currStruct == "ParallelSettings"){
                    
                    
                    string elem = split(line, "//")[0];
                    vector<string> items = split(elem, "=");

                    trim(items[0]);

                    if(items[0] == "Make_GPU"){
                        trim(items[1]);
                        ParallelSettingsInput.Make_GPU = (items[1] == ".true.");
                    }else if(items[0] == "OMP_threads"){
                        trim(items[1]);
                        ParallelSettingsInput.OMP_threads = stoi(items[1]);
                    }

                }else if(currStruct == "AnalysiesCrystaltilts"){
                    
                    
                    string elem = split(line, "//")[0];
                    vector<string> items = split(elem, "=");

                    trim(items[0]);

                    //TODO: UNIMPLEMENTED ANALYSIS FEATURE

                }
            }
        }
    }else{
    #ifdef QT_EXISTS
        QString message = "Could not open input file: ";
        message.append(QString::fromUtf8(inFile.c_str()));

        QMessageBox msgBox;
        msgBox.setText(message);
        msgBox.setInformativeText("Please check if the file still exists or has been corruped.");
        msgBox.setStandardButtons(QMessageBox::Close);
        msgBox.setDefaultButton(QMessageBox::Close);
        int ret = msgBox.exec();

        return ret;
    #else
        cout << "Could not open input file: " << inFile << endl;
    #endif
    }
    inputFile.close();
    //cout << "Input file read." << endl;

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

    return 0; // Dirty workaround
}

#ifdef QT_EXISTS
InputSettingsPrompt::InputSettingsPrompt(GUISettingsWindow* simWindow, QWidget *parent) : ui(new Ui::InputSettingsPrompt), simWindow(simWindow)
{
    ui->setupUi(this);
    (void)connect(ui->pushButton, &QPushButton::clicked, this, &InputSettingsPrompt::loadInputConfig);

    connect(ui->pushButton_2, &QPushButton::clicked, this, [this]() {
        QString dirname = QFileDialog::getExistingDirectory(this,
            tr("Open Workspace Directory"),
            QString(),
            QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks
        );

        if(dirname.isEmpty())
        {
            return;
        }

        ui->lineEdit->setText(dirname + "/");
    });

    connect(ui->pushButton_3, &QPushButton::clicked, this, [this]() {
        QString dirname = QFileDialog::getExistingDirectory(this,
            tr("Open Output Directory"),
            QString(),
            QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks
        );

        if(dirname.isEmpty())
        {
            return;
        }

        ui->lineEdit_3->setText(dirname);
    });
}

void InputSettingsPrompt::loadInputConfig()
{
    char inFile[200];
    std::string input_dir_path = ui->lineEdit->text().toStdString();
    std::string input_config_file_path = ui->lineEdit_2->text().toStdString();
    std::string output_dir_path = ui->lineEdit_3->text().toStdString();

    // Setup workspace dir
    if(!input_dir_path.empty())
    {
        strcpy(File_simu, input_dir_path.c_str());
    }
    else // Else try and use the default workspace name
    {
        strcpy(File_simu, ui->lineEdit->placeholderText().toStdString().c_str());
    }
    //std::cout << "Path to simulation workspace: " << File_simu << "\n\n" << std::endl;

    // Setup input config file name
    strcpy(inFile, File_simu);
    if(!input_config_file_path.empty())
    {
        strcat(inFile, "/");
        strcat(inFile, input_config_file_path.c_str());
    }
    else // Else try and use the default filename
    {
        strcat(inFile, ui->lineEdit_2->placeholderText().toStdString().c_str());
    }

    // Configure the output directory
    if(!output_dir_path.empty())
    {
        
        if(std::filesystem::path(output_dir_path).is_relative())
        {
            strcat(Output_dir, File_simu);
            strcat(Output_dir, "/");
            strcat(Output_dir, output_dir_path.c_str());
        }
        else // Just use a path relative to the executable's location
        {
            strcat(Output_dir, output_dir_path.c_str());
        }
    }
    else
    {
        strcat(Output_dir, ui->lineEdit_3->placeholderText().toStdString().c_str());
    }

    if(!filesystem::is_directory(Output_dir) || !filesystem::exists(Output_dir))
    {
        filesystem::create_directories(Output_dir);
    }

    if(configure(inFile) == 0)
    {
        simWindow->setup();
        simWindow->updateElements();
        simWindow->show();
        close();
    }
    else
    {
        QCoreApplication::exit(1);
    }
}
#endif