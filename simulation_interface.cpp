#include <iostream>
#include <float.h>
#include <thread>

#include "simulation_interface.h"

#include "simu_starter.hh"
#include "Util.h"

#ifdef QT_EXISTS
#include "ui_simulation_window.h"
#endif

using namespace std;


bool first = true;

thread t1;

std::atomic<bool> done(false);

#ifndef QT_EXISTS
void SimulationInterface::headlessSimu(){
    if(UserSettingsInput.mask_C1 < 0 || UserSettingsInput.mask_C1 > 2){
        cout << "bad input for first crystal mask: " << UserSettingsInput.mask_C1 << endl;
        throw runtime_error("value of 0 for no mask, 1 for mask on the bottom and 2 for mask on the top");
    }

    if(UserSettingsInput.mask_C2 < 0 || UserSettingsInput.mask_C2 > 2){
        cout << "bad input for second crystal mask: " << UserSettingsInput.mask_C2 << endl;
        throw runtime_error("value of 0 for no mask, 1 for mask on the bottom and 2 for mask on the top");
    }

    if(FullEnergySpectrumInput.make_more_lines == 1){
        if(!FullEnergySpectrumInput.Do_background){
            if(FullEnergySpectrumInput.p3_ener + FullEnergySpectrumInput.p2_ener + FullEnergySpectrumInput.p1_ener > 1.0){
                throw runtime_error("bad input for lines proportion: " + to_string(FullEnergySpectrumInput.p1_ener) + " + " + to_string(FullEnergySpectrumInput.p2_ener) + " + " + to_string(FullEnergySpectrumInput.p3_ener) + " is greater than 1");
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
        
        Util::Read_EnergySpectrum(FullEnergySpectrumInput.energy_spectrum_file);
        
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
                    throw runtime_error("bad input on the energies. requested energy less than 10 eV");
                }
            }
        }else if(PhysicalParametersInput.Unit_energy == "A"){
            for(int i = 0; i < 4; i++){
                if(reques_energ[i] > 10.0){
                    throw runtime_error("bad input on the energies. requested energy more than 10 A");
                }
            }
        }else{
            throw runtime_error("bad input on the energy unit: " + PhysicalParametersInput.Unit_energy);
        }
    }else{
        bool usable;
        if(PhysicalParametersInput.Unit_energy == "keV"){
            usable = Util::CheckSpectrum("eV");

            if(! usable){
                throw runtime_error("bad input on the energies. requested energy spectrum will not be visible in output");
            }
        }else if(PhysicalParametersInput.Unit_energy == "eV"){
            usable = Util::CheckSpectrum("eV");

            if(! usable){
                throw runtime_error("bad input on the energies. requested energy spectrum will not be visible in output");
            }
        }else if(PhysicalParametersInput.Unit_energy == "A"){
            usable = Util::CheckSpectrum("A");

            if(! usable){
                throw runtime_error("bad input on the energies. requested energy spectrum will not be visible in output");
            }
        }else{
            throw runtime_error("bad input on the energy unit: " + PhysicalParametersInput.Unit_energy);
        }
    }


    if(FullEnergySpectrumInput.make_more_lines == 1){
        for(int i = 0; i < 4; i++){
            reques_width[i] /= 2.0;

            if(PhysicalParametersInput.Unit_energy == evv[0]){
                picks[i].lamda = Convert_Ag_minusone_eV / reques_energ[i];
                picks[i].natural_varia = Convert_Ag_minusone_eV * reques_width[i] / (pow(reques_energ[i], 2) - pow(reques_width[i], 2));
            }else{
                picks[i].lamda = reques_energ[i];
                picks[i].natural_varia = reques_width[i];
            }
        }
    }else if(FullEnergySpectrumInput.make_more_lines == 0){
        reques_width[1] = reques_width[1] / 2.0;

        if(PhysicalParametersInput.Unit_energy == evv[0]){
            picks[1].lamda = Convert_Ag_minusone_eV / reques_energ[1];
            picks[1].natural_varia = Convert_Ag_minusone_eV * reques_width[1] / (pow(reques_energ[1], 2) - pow(reques_width[1], 2));
        }else{
            picks[1].lamda = reques_energ[1];
            picks[1].natural_varia = reques_width[1];
        }
    }


    gauss_Doop_ev = PhysicalParametersInput.gauss_Doop;
    PhysicalParametersInput.gauss_Doop = Convert_Ag_minusone_eV * PhysicalParametersInput.gauss_Doop / (pow(reques_energ[1], 2) - pow(PhysicalParametersInput.gauss_Doop, 2));


    if(GeometryInput.mode_bragg_geo){
        Simu_Starter::Make_Simu(nullptr);
    }else{
        cout << "unimplemented transmission mode" << endl;
    }
}

#else
void SimulationInterface::guiSimu(){
    
    stringstream logString;

    if(UserSettingsInput.mask_C1 < 0 || UserSettingsInput.mask_C1 > 2){
        
        logString << "bad input for first crystal mask: " << UserSettingsInput.mask_C1 << endl;
        LogLine(logString.str());
        throw runtime_error("value of 0 for no mask, 1 for mask on the bottom and 2 for mask on the top");
    }

    if(UserSettingsInput.mask_C2 < 0 || UserSettingsInput.mask_C2 > 2){
        logString.clear();
        logString << "bad input for second crystal mask: " << UserSettingsInput.mask_C2 << endl;
        LogLine(logString.str());
        throw runtime_error("value of 0 for no mask, 1 for mask on the bottom and 2 for mask on the top");
    }


    if(FullEnergySpectrumInput.make_more_lines == 1){
        if(!FullEnergySpectrumInput.Do_background){
            if(FullEnergySpectrumInput.p3_ener + FullEnergySpectrumInput.p2_ener + FullEnergySpectrumInput.p1_ener > 1.0){
                throw runtime_error("bad input for lines proportion: " + to_string(FullEnergySpectrumInput.p1_ener) + " + " + to_string(FullEnergySpectrumInput.p2_ener) + " + " + to_string(FullEnergySpectrumInput.p3_ener) + " is greater than 1");
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
        logString.clear();
        logString << "Reading input energy spectrum..." << endl;
        LogLine(logString.str());
        
        Util::Read_EnergySpectrum(FullEnergySpectrumInput.energy_spectrum_file);
        
        logString.clear();
        logString << "Input energy spectrum read." << endl;
        LogLine(logString.str());

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
                    throw runtime_error("bad input on the energies. requested energy less than 10 eV");
                }
            }
        }else if(PhysicalParametersInput.Unit_energy == "A"){
            for(int i = 0; i < 4; i++){
                if(reques_energ[i] > 10.0){
                    throw runtime_error("bad input on the energies. requested energy more than 10 A");
                }
            }
        }else{
            throw runtime_error("bad input on the energy unit: " + PhysicalParametersInput.Unit_energy);
        }
    }else{
        bool usable;
        if(PhysicalParametersInput.Unit_energy == "keV"){
            usable = Util::CheckSpectrum("eV");

            if(! usable){
                throw runtime_error("bad input on the energies. requested energy spectrum will not be visible in output");
            }
        }else if(PhysicalParametersInput.Unit_energy == "eV"){
            usable = Util::CheckSpectrum("eV");

            if(! usable){
                throw runtime_error("bad input on the energies. requested energy spectrum will not be visible in output");
            }
        }else if(PhysicalParametersInput.Unit_energy == "A"){
            usable = Util::CheckSpectrum("A");

            if(! usable){
                throw runtime_error("bad input on the energies. requested energy spectrum will not be visible in output");
            }
        }else{
            throw runtime_error("bad input on the energy unit: " + PhysicalParametersInput.Unit_energy);
        }
    }


    if(FullEnergySpectrumInput.make_more_lines == 1){
        for(int i = 0; i < 4; i++){
            reques_width[i] /= 2.0;

            if(PhysicalParametersInput.Unit_energy == evv[0]){
                picks[i].lamda = Convert_Ag_minusone_eV / reques_energ[i];
                picks[i].natural_varia = Convert_Ag_minusone_eV * reques_width[i] / (pow(reques_energ[i], 2) - pow(reques_width[i], 2));
            }else{
                picks[i].lamda = reques_energ[i];
                picks[i].natural_varia = reques_width[i];
            }
        }
    }else if(FullEnergySpectrumInput.make_more_lines == 0){
        reques_width[1] = reques_width[1] / 2.0;

        if(PhysicalParametersInput.Unit_energy == evv[0]){
            picks[1].lamda = Convert_Ag_minusone_eV / reques_energ[1];
            picks[1].natural_varia = Convert_Ag_minusone_eV * reques_width[1] / (pow(reques_energ[1], 2) - pow(reques_width[1], 2));
        }else{
            picks[1].lamda = reques_energ[1];
            picks[1].natural_varia = reques_width[1];
        }
    }


    gauss_Doop_ev = PhysicalParametersInput.gauss_Doop;
    PhysicalParametersInput.gauss_Doop = Convert_Ag_minusone_eV * PhysicalParametersInput.gauss_Doop / (pow(reques_energ[1], 2) - pow(PhysicalParametersInput.gauss_Doop, 2));


    if(GeometryInput.mode_bragg_geo){
        Simu_Starter::Make_Simu(this);
    }else{
        logString.clear();
        logString << "unimplemented transmission mode" << endl;
        LogLine(logString.str());

    }
}

//QCPErrorBars *errorBars_1, *errorBars;

QImage *img, *scale;


SimulationInterface::SimulationInterface(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::SimulationWindow),
    running(false),
    open(true)
{
    ui->setupUi(this);

    ui->AP_histogram->addGraph();
    ui->P_histogram->addGraph();

    ui->AP_histogram->graph(0)->setLineStyle(QCPGraph::lsStepCenter);
    ui->P_histogram->graph(0)->setLineStyle(QCPGraph::lsStepCenter);

    ui->AP_histogram->addGraph(ui->AP_histogram->xAxis2, ui->AP_histogram->yAxis);
    //ui->P_histogram->addGraph(ui->P_histogram->xAxis2, ui->P_histogram->yAxis);

    ui->AP_histogram->graph(1)->setLineStyle(QCPGraph::lsStepCenter);
    //ui->P_histogram->graph(1)->setLineStyle(QCPGraph::lsStepCenter);

    ui->AP_histogram->graph(1)->setVisible(false);
    //ui->P_histogram->setVisible(false);

    QPen pen;
    pen.setColor(QColor(0, 0, 255, 255));

    ui->AP_histogram->plotLayout()->insertRow(0);
    ui->AP_histogram->plotLayout()->addElement(0, 0, new QCPTextElement(ui->AP_histogram, "Antiparallel Histogram", QFont("sans", 8, QFont::Bold)));
    ui->P_histogram->plotLayout()->insertRow(0);
    ui->P_histogram->plotLayout()->addElement(0, 0, new QCPTextElement(ui->P_histogram, "Parallel Histogram", QFont("sans", 8, QFont::Bold)));

    ui->AP_histogram->yAxis->setLabel("Counts");
    ui->AP_histogram->xAxis->setLabel("Crystal 2 antiparallel angle");
    ui->AP_histogram->xAxis2->setLabel("Energy (eV)");

    ui->P_histogram->yAxis->setLabel("Counts");
    ui->P_histogram->xAxis->setLabel("Crystal 2 parallel angle");
    //ui->P_histogram->xAxis2->setLabel("Energy (eV)");

    ui->AP_histogram->xAxis2->setVisible(true);
    ui->AP_histogram->yAxis2->setVisible(true);
    ui->AP_histogram->xAxis2->setTickLabels(true);
    ui->AP_histogram->yAxis2->setTickLabels(false);
    ui->AP_histogram->xAxis2->setTicks(true);
    ui->AP_histogram->yAxis2->setTicks(false);
    ui->AP_histogram->xAxis2->setSubTicks(true);
    ui->AP_histogram->yAxis2->setSubTicks(false);
    ui->AP_histogram->xAxis2->setRangeReversed(true);
    
    ui->AP_histogram->xAxis->setTickLabelRotation(28);

    ui->P_histogram->xAxis2->setVisible(true);
    ui->P_histogram->yAxis2->setVisible(true);
    ui->P_histogram->xAxis2->setTickLabels(false);
    ui->P_histogram->yAxis2->setTickLabels(false);
    ui->P_histogram->xAxis2->setTicks(false);
    ui->P_histogram->yAxis2->setTicks(false);
    ui->P_histogram->xAxis2->setSubTicks(false);
    ui->P_histogram->yAxis2->setSubTicks(false);
    //ui->P_histogram->xAxis2->setRangeReversed(true);
    
    ui->P_histogram->xAxis->setTickLabelRotation(28);

    lastHistogramUpdate = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();

    /*
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
    */

    img = new QImage(n_his_ima, n_his_ima, QImage::Format::Format_RGB16);
    scale = new QImage(n_his_ima / 10, n_his_ima, QImage::Format::Format_RGB16);

    for(int i = 0; i < n_his_ima / 10; i++){
        for(int j = 0; j < n_his_ima; j++){
            int r = (int)(255 * red(2 * ((double)j / (double)n_his_ima) - 1.0)), g = (int)(255 * green(2 * ((double)j / (double)n_his_ima) - 1.0)), b = (int)(255 * blue(2 * ((double)j / (double)n_his_ima) - 1.0));
            scale->setPixelColor(i, n_his_ima - 1 - j, QColor(r, g, b));
        }
    }

    qRegisterMetaType<Stats>("Stats");
    qRegisterMetaType<Plots>("Plots");
    qRegisterMetaType<Plates>("Plates");
    qRegisterMetaType<Times>("Times");
    qRegisterMetaType<std::string>("std::string");
    qRegisterMetaType<QTextCursor>("QTextCursor");

    connect(this, SIGNAL(changeStatsSignal(Stats)), this, SLOT(changeStats(Stats)), Qt::QueuedConnection);
    connect(this, SIGNAL(changePlotsSignal(Plots)), this, SLOT(changePlots(Plots)), Qt::QueuedConnection);
    connect(this, SIGNAL(changePlatesSignal(Plates)), this, SLOT(changePlates(Plates)), Qt::QueuedConnection);
    connect(this, SIGNAL(changeTimesSignal(Times)), this, SLOT(changeTimes(Times)), Qt::QueuedConnection);
    connect(this, SIGNAL(LogLineSignal(std::string)), this, SLOT(LogLine(std::string)), Qt::QueuedConnection);
    connect(this, SIGNAL(showDoneDialogSignal()), this, SLOT(showDoneDialog()), Qt::QueuedConnection);
    
    //make the logbox read only
    ui->logBox->setReadOnly(true);
    logBox = ui->logBox;
}

SimulationInterface::~SimulationInterface()
{
    delete ui;
#ifndef LIB_DEF
    exit(0);
#endif
}

void SimulationInterface::startSimulationThread()
{
#ifdef LIB_DEF
    if(!running)
    {
        // Start the simulation job
        t1 = std::thread([&]{
            SimulationInterface::guiSimu();
            emit SimulationInterface::showDoneDialogSignal();
            done = true;
            running = false;
        });
        running = true;
    }
#endif
}

void SimulationInterface::showEvent(QShowEvent *)
{
#ifndef LIB_DEF    
    open = true;

    if (!running) {
        t1 = std::thread([&] {
            SimulationInterface::guiSimu();
            emit SimulationInterface::showDoneDialogSignal();
            done = true;
            running = false;
        });
        running = true;
    }
#endif
}

void SimulationInterface::closeEvent(QCloseEvent *){
#ifndef LIB_DEF
    open = false;
    
    if(t1.joinable())
    {
        t1.join();
    }
#endif
}

void SimulationInterface::showDoneDialog()
{
    if(open)
    {
        QMessageBox::information(this, tr("DCS Simulation"), tr("Simulation Finished!"), QMessageBox::Ok);
    }
    else
    {
        QMessageBox::information(this, tr("DCS Simulation"), tr("Simulation Finished Early!!!\nDo not use the output results as they are incomplete."), QMessageBox::Ok);
    }
}

void SimulationInterface::changeStats(Stats stats)
{
    ui->Ecnts_val->setText(QString(split(to_string(stats.c_sour), ".")[0].c_str()));
    ui->C1cnts_val->setText(QString(split(to_string(stats.c_cr1), ".")[0].c_str()));
    ui->C2Pcnts_val->setText(QString(split(to_string(stats.c_cr2_para), ".")[0].c_str()));
    ui->DPcnts_val->setText(QString(split(to_string(stats.c_detc_para), ".")[0].c_str()));
    ui->C2Acnts_val->setText(QString(split(to_string(stats.c_cr2_anti), ".")[0].c_str()));
    ui->DAcnts_val->setText(QString(split(to_string(stats.c_detc_anti), ".")[0].c_str()));
    ui->currRot_val->setText(QString(to_string(stats.delrot * convdeg).c_str()));
    ui->GL3Dvis->setDelrot(stats.delrot);
    ui->GL3Dvis->setEventsToTrace(stats.events_para, stats.events_anti);
    ui->GL3Dvis->update();
}

void SimulationInterface::changePlots(Plots plots){
    long int currTime = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();

    if (currTime - lastHistogramUpdate >= 100)
    {
        lastHistogramUpdate = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
        QVector<double> qe_para, qx_para, qy_para, qe_anti, qx_anti, qy_anti, e_para, e_anti;

        for(unsigned int i = 0; i < plots.para.size(); i++){
            qe_para << plots.para[i].energy;
            qx_para << plots.para[i].x;
            qy_para << plots.para[i].y;
            e_para << plots.para[i].error;
        }

        for(unsigned int i = 0; i < plots.anti.size(); i++){
            qe_anti << plots.anti[i].energy;
            qx_anti << plots.anti[i].x;
            qy_anti << plots.anti[i].y;
            e_anti << plots.anti[i].error;
        }

        //errorBars->addData(e_anti);
        //errorBars_1->addData(e_para);

        ui->AP_histogram->graph(0)->setData(qx_anti, qy_anti);
        ui->P_histogram->graph(0)->setData(qx_para, qy_para);

        ui->AP_histogram->graph(0)->rescaleAxes();
        ui->P_histogram->graph(0)->rescaleAxes();

        ui->AP_histogram->graph(1)->setData(qe_anti, qy_anti);
        //ui->P_histogram->graph(1)->setData(qe_para, qy_para);

        ui->AP_histogram->graph(1)->rescaleAxes();
        //ui->P_histogram->graph(1)->rescaleAxes();

        
        if(!ui->AP_histogram->paintingActive())
            ui->AP_histogram->replot(QCustomPlot::rpQueuedReplot);
        if(!ui->P_histogram->paintingActive())
            ui->P_histogram->replot(QCustomPlot::rpQueuedReplot);
    }
}

void SimulationInterface::changePlates(Plates plates){
    for(int i = 0; i < n_his_ima; i++){
        for(int j = 0; j < n_his_ima; j++){
            int r = (int)(255 * red(2 * ((plates.hist_image[i][j] / plates.max_z) - 0.5)));
            int g = (int)(255 * green(2 * ((plates.hist_image[i][j] / plates.max_z) - 0.5)));
            int b = (int)(255 * blue(2 * ((plates.hist_image[i][j] / plates.max_z) - 0.5)));
            img->setPixelColor(i, n_his_ima - 1 - j, QColor(r, g, b));
        }
    }

    int w, h, ws, hs;

    if(plates.crystal == 0){
        w = ui->E_ImagePlate->width();
        h = ui->E_ImagePlate->height();

        ws = ui->E_cScale->width();
        hs = ui->E_cScale->height();

        ui->E_ImagePlate->setPixmap(QPixmap::fromImage(*img).scaled(w,h,Qt::KeepAspectRatio));
        ui->E_cScale->setPixmap(QPixmap::fromImage(*scale).scaled(ws, hs, Qt::KeepAspectRatio));
        ui->E_vScale->setText(QString(split(to_string(plates.max_z), ".")[0].c_str()));
    }else if(plates.crystal == 1){
        w = ui->C1_ImagePlate->width();
        h = ui->C1_ImagePlate->height();

        ws = ui->C1_cScale->width();
        hs = ui->C1_cScale->height();

        ui->C1_ImagePlate->setPixmap(QPixmap::fromImage(*img).scaled(w,h,Qt::KeepAspectRatio));
        ui->C1_cScale->setPixmap(QPixmap::fromImage(*scale).scaled(ws, hs, Qt::KeepAspectRatio));
        ui->C1_vScale->setText(QString(split(to_string(plates.max_z), ".")[0].c_str()));
    }else if(plates.crystal == 4){
        w = ui->C2A_ImagePlate->width();
        h = ui->C2A_ImagePlate->height();

        ws = ui->C2A_cScale->width();
        hs = ui->C2A_cScale->height();

        ui->C2A_ImagePlate->setPixmap(QPixmap::fromImage(*img).scaled(w,h,Qt::KeepAspectRatio));
        ui->C2A_cScale->setPixmap(QPixmap::fromImage(*scale).scaled(ws, hs, Qt::KeepAspectRatio));
        ui->C2A_vScale->setText(QString(split(to_string(plates.max_z), ".")[0].c_str()));
    }else if(plates.crystal == 5){
        w = ui->DA_ImagePlate->width();
        h = ui->DA_ImagePlate->height();

        ws = ui->DA_cScale->width();
        hs = ui->DA_cScale->height();

        ui->DA_ImagePlate->setPixmap(QPixmap::fromImage(*img).scaled(w,h,Qt::KeepAspectRatio));
        ui->DA_cScale->setPixmap(QPixmap::fromImage(*scale).scaled(ws, hs, Qt::KeepAspectRatio));
        ui->DA_vScale->setText(QString(split(to_string(plates.max_z), ".")[0].c_str()));
    }else if(plates.crystal == 2){
        w = ui->C2P_ImagePlate->width();
        h = ui->C2P_ImagePlate->height();

        ws = ui->C2P_cScale->width();
        hs = ui->C2P_cScale->height();

        ui->C2P_ImagePlate->setPixmap(QPixmap::fromImage(*img).scaled(w,h,Qt::KeepAspectRatio));
        ui->C2P_cScale->setPixmap(QPixmap::fromImage(*scale).scaled(ws, hs, Qt::KeepAspectRatio));
        ui->C2P_vScale->setText(QString(split(to_string(plates.max_z), ".")[0].c_str()));
    }else if(plates.crystal == 3){
        w = ui->DP_ImagePlate->width();
        h = ui->DP_ImagePlate->height();

        ws = ui->DP_cScale->width();
        hs = ui->DP_cScale->height();

        ui->DP_ImagePlate->setPixmap(QPixmap::fromImage(*img).scaled(w,h,Qt::KeepAspectRatio));
        ui->DP_cScale->setPixmap(QPixmap::fromImage(*scale).scaled(ws, hs, Qt::KeepAspectRatio));
        ui->DP_vScale->setText(QString(split(to_string(plates.max_z), ".")[0].c_str()));
    }
}

void SimulationInterface::changeTimes(Times times){

    if(times.timeSlot == 0)
        ui->simstart->setText(QString(("Simulation start at:      " + to_string(times.h) + "h    " + to_string(times.m) + "m     " + to_string(times.s) + "s").c_str()));
    else if(times.timeSlot == 1)
        ui->simremain->setText(QString(("Remaining time estimate:      " + to_string(times.h) + "h    " + to_string(times.m) + "m     " + to_string(times.s) + "s").c_str()));
}

void SimulationInterface::LogLine(std::string line) {
    ui->logBox->appendPlainText(QString(line.c_str()));
}
#endif