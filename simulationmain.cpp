#include <iostream>
#include <float.h>
#include <thread>

#include "simulationmain.h"
#include "ui_simulationmain.h"

#include "simuGlobals.hh"
#include "obtain_energyspectrum.hh"
#include "CheckInputSpectrum.hh"
#include "double_crystal_diffraction.hh"

using namespace std;


extern Geometry Geometry;
extern UserSettings UserSettings;
extern GeoParapathlengths GeoParapathlengths;
extern Geolengthelements Geolengthelements;
extern GeoParameters GeoParameters;
extern CurveVerticalTilt CurveVerticalTilt;
extern Graph_options Graph_options;
extern plotparameters plotparameters;
extern numberrays numberrays;
extern physical_parameters physical_parameters;
extern polarization_parameters polarization_parameters;
extern temperature_parameters temperature_parameters;
extern fullenergyspectrum fullenergyspectrum;
extern Curved_Crystal Curved_Crystal;
extern AnalysiesCrystaltilts AnalysiesCrystaltilts;

extern double reques_energ[4];
extern double reques_width[4];

extern double linelamda, naturalwidth;

extern double d_lat;

extern bool exist_file, usable;

extern pick picks[5];

extern double gauss_Doop_ev;

extern char File_simu[200];

bool first = true;

thread t1;

std::atomic<bool> done(false);

void SimulationMain::guiSimu(){
    if(UserSettings.mask_C1 < 0 or UserSettings.mask_C1 > 2){
        cout << "bad input for first crystal mask: " << UserSettings.mask_C1 << endl;
        throw runtime_error("value of 0 for no mask, 1 for mask on the bottom and 2 for mask on the top");
    }

    if(UserSettings.mask_C2 < 0 or UserSettings.mask_C2 > 2){
            cout << "bad input for second crystal mask: " << UserSettings.mask_C2 << endl;
            throw runtime_error("value of 0 for no mask, 1 for mask on the bottom and 2 for mask on the top");
    }


    if(not Graph_options.MakeDislin){
        Graph_options.make_graph_profile = false;
        Graph_options.make_image_plates = false;
        AnalysiesCrystaltilts.make_graph_widths = false;
    }

    if(fullenergyspectrum.make_more_lines == 1){
        if(not fullenergyspectrum.Do_background){
            if(fullenergyspectrum.p3_ener + fullenergyspectrum.p2_ener + fullenergyspectrum.p1_ener > 1.0){
                throw runtime_error("bad input for lines proportion: " + to_string(fullenergyspectrum.p1_ener) + " + " + to_string(fullenergyspectrum.p2_ener) + " + " + to_string(fullenergyspectrum.p3_ener) + " is greater than 1");
            }
        }

        reques_energ[0] = fullenergyspectrum.linelamda1;
        reques_energ[1] = fullenergyspectrum.linelamda2;
        reques_energ[2] = fullenergyspectrum.linelamda3;
        reques_energ[3] = fullenergyspectrum.linelamda4;

        reques_width[0] = fullenergyspectrum.naturalwidth1;
        reques_width[1] = fullenergyspectrum.naturalwidth2;
        reques_width[2] = fullenergyspectrum.naturalwidth3;
        reques_width[3] = fullenergyspectrum.naturalwidth4;

    }else if(fullenergyspectrum.make_more_lines == 0){
        reques_energ[0] = linelamda;
        reques_energ[1] = fullenergyspectrum.linelamda2;
        reques_energ[2] = fullenergyspectrum.linelamda3;
        reques_energ[3] = fullenergyspectrum.linelamda4;

        reques_width[0] = naturalwidth;
        reques_width[1] = fullenergyspectrum.naturalwidth2;
        reques_width[2] = fullenergyspectrum.naturalwidth3;
        reques_width[3] = fullenergyspectrum.naturalwidth4;
    }else{
        cout << "Reading input energy spectrum..." << endl;

        Obtain_EnergySpectrum::Read_EnergySpectrum();

        cout << "Input energy spectrum read." << endl;
    }

    if(Geometry.crystal_Si){
        d_lat = a_si_para / sqrt(pow(Geometry.imh, 2) + pow(Geometry.imk, 2) + pow(Geometry.iml, 2));
    }else{
        d_lat = a_Ge_para / sqrt(pow(Geometry.imh, 2) + pow(Geometry.imk, 2) + pow(Geometry.iml, 2));
    }

    if(fullenergyspectrum.make_more_lines == 0 or fullenergyspectrum.make_more_lines == 1){
        if(physical_parameters.Unit_energy == evv[0]){
            for(int i = 0; i < 4; i++){
                if(reques_energ[i] < 10.0){
                    throw runtime_error("bad input on the energies. requested energy less than 10 eV");
                }
            }
        }else if(physical_parameters.Unit_energy == "A"){
            for(int i = 0; i < 4; i++){
                if(reques_energ[i] > 10.0){
                    throw runtime_error("bad input on the energies. requested energy more than 10 A");
                }
            }
        }else{
            throw runtime_error("bad input on the energy unit: " + physical_parameters.Unit_energy);
        }
    }else{
        if(physical_parameters.Unit_energy == "keV"){
            usable = CheckInputSpectrum::CheckSpectrum("eV");

            if(not usable){
                throw runtime_error("bad input on the energies. requested energy spectrum will not be visible in output");
            }
        }else if(physical_parameters.Unit_energy == "eV"){
            usable = CheckInputSpectrum::CheckSpectrum("eV");

            if(not usable){
                throw runtime_error("bad input on the energies. requested energy spectrum will not be visible in output");
            }
        }else if(physical_parameters.Unit_energy == "A"){
            usable = CheckInputSpectrum::CheckSpectrum("A");

            if(not usable){
                throw runtime_error("bad input on the energies. requested energy spectrum will not be visible in output");
            }
        }else{
            throw runtime_error("bad input on the energy unit: " + physical_parameters.Unit_energy);
        }
    }


    if(fullenergyspectrum.make_more_lines == 1){
        for(int i = 0; i < 4; i++){
            reques_width[i] = reques_width[i] / 2.0;

            if(physical_parameters.Unit_energy == evv[0]){
                picks[i].lamda = Convert_Ag_minusone_eV / reques_energ[i];
                picks[i].natural_varia = Convert_Ag_minusone_eV * reques_width[i] / (pow(reques_energ[i], 2) - pow(reques_width[i], 2));
            }else{
                picks[i].lamda = reques_energ[i];
                picks[i].natural_varia = reques_width[i];
            }
        }
    }else if(fullenergyspectrum.make_more_lines == 0){
        reques_width[1] = reques_width[1] / 2.0;

        if(physical_parameters.Unit_energy == evv[0]){
            picks[1].lamda = Convert_Ag_minusone_eV / reques_energ[1];
            picks[1].natural_varia = Convert_Ag_minusone_eV * reques_width[1] / (pow(reques_energ[1], 2) - pow(reques_width[1], 2));
        }else{
            picks[1].lamda = reques_energ[1];
            picks[1].natural_varia = reques_width[1];
        }
    }


    gauss_Doop_ev = physical_parameters.gauss_Doop;
    physical_parameters.gauss_Doop = Convert_Ag_minusone_eV * physical_parameters.gauss_Doop / (pow(reques_energ[1], 2) - pow(physical_parameters.gauss_Doop, 2));


    if(Geometry.mode_bragg_geo){
        Double_Crystal_diffraction::Make_Simu(this);
    }else{
        cout << "unimplemented transmission mode" << endl;
    }
}

QCPErrorBars *errorBars_1, *errorBars;

QImage *img, *scale;

SimulationMain::SimulationMain(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::SimulationMain)
{
    ui->setupUi(this);

    ui->AP_histogram->addGraph();
    ui->P_histogram->addGraph();

    ui->AP_histogram->graph(0)->setLineStyle(QCPGraph::lsStepCenter);
    ui->P_histogram->graph(0)->setLineStyle(QCPGraph::lsStepCenter);

    QPen pen;
    pen.setColor(QColor(0, 0, 255, 255));

    ui->AP_histogram->plotLayout()->insertRow(0);
    ui->AP_histogram->plotLayout()->addElement(0, 0, new QCPTextElement(ui->AP_histogram, "Antiparallel Histogram", QFont("sans", 8, QFont::Bold)));
    ui->P_histogram->plotLayout()->insertRow(0);
    ui->P_histogram->plotLayout()->addElement(0, 0, new QCPTextElement(ui->P_histogram, "Parallel Histogram", QFont("sans", 8, QFont::Bold)));

    ui->AP_histogram->yAxis->setLabel("Counts");
    ui->AP_histogram->xAxis->setLabel("Crystal 2 antiparallel angle");

    ui->P_histogram->yAxis->setLabel("Counts");
    ui->P_histogram->xAxis->setLabel("Crystal 2 parallel angle");

    ui->AP_histogram->xAxis2->setVisible(true);
    ui->AP_histogram->yAxis2->setVisible(true);
    ui->AP_histogram->xAxis2->setTickLabels(false);
    ui->AP_histogram->yAxis2->setTickLabels(false);
    ui->AP_histogram->xAxis2->setTicks(false);
    ui->AP_histogram->yAxis2->setTicks(false);
    ui->AP_histogram->xAxis2->setSubTicks(false);
    ui->AP_histogram->yAxis2->setSubTicks(false);
    ui->AP_histogram->xAxis->setTickLabelRotation(28);

    ui->P_histogram->xAxis2->setVisible(true);
    ui->P_histogram->yAxis2->setVisible(true);
    ui->P_histogram->xAxis2->setTickLabels(false);
    ui->P_histogram->yAxis2->setTickLabels(false);
    ui->P_histogram->xAxis2->setTicks(false);
    ui->P_histogram->yAxis2->setTicks(false);
    ui->P_histogram->xAxis2->setSubTicks(false);
    ui->P_histogram->yAxis2->setSubTicks(false);
    ui->P_histogram->xAxis->setTickLabelRotation(28);

    lastHistogramUpdate = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();

    errorBars = new QCPErrorBars(ui->AP_histogram->xAxis, ui->AP_histogram->yAxis);
    errorBars->removeFromLegend();
    errorBars->setAntialiased(false);
    errorBars->setDataPlottable(ui->AP_histogram->graph(0));
    errorBars->setPen(QPen(QColor(180,180,180)));
    errorBars->setErrorType(QCPErrorBars::etValueError);

    errorBars_1 = new QCPErrorBars(ui->P_histogram->xAxis, ui->P_histogram->yAxis);
    errorBars_1->removeFromLegend();
    errorBars_1->setAntialiased(false);
    errorBars_1->setDataPlottable(ui->P_histogram->graph(0));
    errorBars_1->setPen(QPen(QColor(180,180,180)));
    errorBars_1->setErrorType(QCPErrorBars::etValueError);

    img = new QImage(n_his_ima, n_his_ima, QImage::Format::Format_RGB16);
    scale = new QImage(n_his_ima / 10, n_his_ima, QImage::Format::Format_RGB16);

    for(int i = 0; i < n_his_ima / 10; i++){
        for(int j = 0; j < n_his_ima; j++){
            int r = (int)(255 * red(2 * ((double)j / (double)n_his_ima) - 1.0)), g = (int)(255 * green(2 * ((double)j / (double)n_his_ima) - 1.0)), b = (int)(255 * blue(2 * ((double)j / (double)n_his_ima) - 1.0));
            scale->setPixelColor(i, n_his_ima - 1 - j, QColor(r, g, b));
        }
    }
}

SimulationMain::~SimulationMain()
{
    delete ui;
    exit(0);
}

void SimulationMain::showEvent(QShowEvent *)
{
    if(first){
        t1 = std::thread([&]{SimulationMain::guiSimu(); done = true;});
        first = false;
    }
}

void SimulationMain::closeEvent(QCloseEvent *){
    if(done){
        t1.join();
        exit(0);
    }else
        throw runtime_error("Simulation ended by user input before finishing. Results are incomplete.");
}

void SimulationMain::changeStats(double c_sour, double c_cr1, double c_cr2_para, double c_cr2_anti, double c_detc_para, double c_detc_anti, double delrot){
    ui->Ecnts_val->setText(QString(split(to_string(c_sour), ".")[0].c_str()));
    ui->C1cnts_val->setText(QString(split(to_string(c_cr1), ".")[0].c_str()));
    ui->C2Pcnts_val->setText(QString(split(to_string(c_cr2_para), ".")[0].c_str()));
    ui->DPcnts_val->setText(QString(split(to_string(c_detc_para), ".")[0].c_str()));
    ui->C2Acnts_val->setText(QString(split(to_string(c_cr2_anti), ".")[0].c_str()));
    ui->DAcnts_val->setText(QString(split(to_string(c_detc_anti), ".")[0].c_str()));
    ui->currRot_val->setText(QString(to_string(delrot).c_str()));
}

void SimulationMain::changePlots(vector<plot> para, vector<plot> anti){
    long int currTime = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();

    if (currTime - lastHistogramUpdate >= 100)
    {
        lastHistogramUpdate = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
        QVector<double> qx_para, qy_para, qx_anti, qy_anti, e_para, e_anti;

        for(unsigned int i = 0; i < para.size(); i++){
            qx_para << para[i].x;
            qy_para << para[i].y;
            e_para << para[i].error;
        }

        for(unsigned int i = 0; i < anti.size(); i++){
            qx_anti << anti[i].x;
            qy_anti << anti[i].y;
            e_anti << anti[i].error;
        }

        errorBars->addData(e_anti);
        errorBars_1->addData(e_para);

        ui->AP_histogram->graph()->setData(qx_anti, qy_anti);
        ui->P_histogram->graph()->setData(qx_para, qy_para);

        ui->AP_histogram->graph()->rescaleAxes();
        ui->P_histogram->graph()->rescaleAxes();

        if(!ui->AP_histogram->paintingActive())
            ui->AP_histogram->replot(QCustomPlot::rpQueuedReplot);
        if(!ui->P_histogram->paintingActive())
            ui->P_histogram->replot(QCustomPlot::rpQueuedReplot);
    }
}

void SimulationMain::changePlates(double hist_image[n_his_ima][n_his_ima], double max_z, int crystal){
    for(int i = 0; i < n_his_ima; i++){
        for(int j = 0; j < n_his_ima; j++){
            int r = (int)(255 * red(2 * ((hist_image[i][j] / max_z) - 0.5))), g = (int)(255 * green(2 * ((hist_image[i][j] / max_z) - 0.5))), b = (int)(255 * blue(2 * ((hist_image[i][j] / max_z) - 0.5)));
            img->setPixelColor(i, n_his_ima - 1 - j, QColor(r, g, b));
        }
    }

    int w, h, ws, hs;

    if(crystal == 0){
        w = ui->E_ImagePlate->width();
        h = ui->E_ImagePlate->height();

        ws = ui->E_cScale->width();
        hs = ui->E_cScale->height();

        ui->E_ImagePlate->setPixmap(QPixmap::fromImage(*img).scaled(w,h,Qt::KeepAspectRatio));
        ui->E_cScale->setPixmap(QPixmap::fromImage(*scale).scaled(ws, hs, Qt::KeepAspectRatio));
        ui->E_vScale->setText(QString(split(to_string(max_z), ".")[0].c_str()));
    }else if(crystal == 1){
        w = ui->C1_ImagePlate->width();
        h = ui->C1_ImagePlate->height();

        ws = ui->C1_cScale->width();
        hs = ui->C1_cScale->height();

        ui->C1_ImagePlate->setPixmap(QPixmap::fromImage(*img).scaled(w,h,Qt::KeepAspectRatio));
        ui->C1_cScale->setPixmap(QPixmap::fromImage(*scale).scaled(ws, hs, Qt::KeepAspectRatio));
        ui->C1_vScale->setText(QString(split(to_string(max_z), ".")[0].c_str()));
    }else if(crystal == 4){
        w = ui->C2A_ImagePlate->width();
        h = ui->C2A_ImagePlate->height();

        ws = ui->C2A_cScale->width();
        hs = ui->C2A_cScale->height();

        ui->C2A_ImagePlate->setPixmap(QPixmap::fromImage(*img).scaled(w,h,Qt::KeepAspectRatio));
        ui->C2A_cScale->setPixmap(QPixmap::fromImage(*scale).scaled(ws, hs, Qt::KeepAspectRatio));
        ui->C2A_vScale->setText(QString(split(to_string(max_z), ".")[0].c_str()));
    }else if(crystal == 5){
        w = ui->DA_ImagePlate->width();
        h = ui->DA_ImagePlate->height();

        ws = ui->DA_cScale->width();
        hs = ui->DA_cScale->height();

        ui->DA_ImagePlate->setPixmap(QPixmap::fromImage(*img).scaled(w,h,Qt::KeepAspectRatio));
        ui->DA_cScale->setPixmap(QPixmap::fromImage(*scale).scaled(ws, hs, Qt::KeepAspectRatio));
        ui->DA_vScale->setText(QString(split(to_string(max_z), ".")[0].c_str()));
    }else if(crystal == 2){
        w = ui->C2P_ImagePlate->width();
        h = ui->C2P_ImagePlate->height();

        ws = ui->C2P_cScale->width();
        hs = ui->C2P_cScale->height();

        ui->C2P_ImagePlate->setPixmap(QPixmap::fromImage(*img).scaled(w,h,Qt::KeepAspectRatio));
        ui->C2P_cScale->setPixmap(QPixmap::fromImage(*scale).scaled(ws, hs, Qt::KeepAspectRatio));
        ui->C2P_vScale->setText(QString(split(to_string(max_z), ".")[0].c_str()));
    }else if(crystal == 3){
        w = ui->DP_ImagePlate->width();
        h = ui->DP_ImagePlate->height();

        ws = ui->DP_cScale->width();
        hs = ui->DP_cScale->height();

        ui->DP_ImagePlate->setPixmap(QPixmap::fromImage(*img).scaled(w,h,Qt::KeepAspectRatio));
        ui->DP_cScale->setPixmap(QPixmap::fromImage(*scale).scaled(ws, hs, Qt::KeepAspectRatio));
        ui->DP_vScale->setText(QString(split(to_string(max_z), ".")[0].c_str()));
    }
}

void SimulationMain::changeTimes(int timeSlot, int h, int m, int s){

    if(timeSlot == 0)
        ui->simstart->setText(QString(("Simulation start at:      " + to_string(h) + "h    " + to_string(m) + "m     " + to_string(s) + "s").c_str()));
    else if(timeSlot == 1)
        ui->simremain->setText(QString(("Remaining time estimate:      " + to_string(h) + "h    " + to_string(m) + "m     " + to_string(s) + "s").c_str()));
}
