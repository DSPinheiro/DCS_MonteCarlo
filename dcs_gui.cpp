#include <float.h>

#include <QDoubleValidator>
#include <QFileDialog>
#include <QMessageBox>

#include "dcs_gui.h"
#include "ui_dcs_gui.h"

#include "simuGlobals.hh"
#include "simulationmain.h"
#include "wrapper/sserializer.inl"

#include <fstream>

// using namespace std;

// extern UserSettings UserSettings;
// extern Graph_options Graph_options;
// extern physical_parameters physical_parameters;
// extern temperature_parameters temperature_parameters;
// extern plotparameters plotparameters;
// extern Geolengthelements Geolengthelements;
// extern GeoParameters GeoParameters;
// extern numberrays numberrays;
// extern GeoParapathlengths GeoParapathlengths;

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

static std::vector<uint8_t> serialize_inputs()
{
    std::vector<uint8_t> buffer;

    serialize_data(buffer, Geometry);

    serialize_data(buffer, UserSettings);

    serialize_data(buffer, GeoParapathlengths.type_source);
    serialize_data(buffer, GeoParapathlengths.LT_aper);
    serialize_data(buffer, GeoParapathlengths.dist_T_Cr1);
    serialize_data(buffer, GeoParapathlengths.dist_Cr1_Cr2);
    serialize_data(buffer, GeoParapathlengths.dist_Cr2_Det);
    serialize_data(buffer, GeoParapathlengths.dist_aper_det);

    serialize_data(buffer, Geolengthelements);

    serialize_data(buffer, GeoParameters);

    serialize_data(buffer, CurveVerticalTilt);

    serialize_data(buffer, Graph_options);

    serialize_data(buffer, plotparameters);

    serialize_data(buffer, numberrays);

    serialize_data(buffer, physical_parameters.Unit_energy);
    serialize_data(buffer, physical_parameters.linelamda);
    serialize_data(buffer, physical_parameters.naturalwidth);
    serialize_data(buffer, physical_parameters.gauss_Doop);

    serialize_data(buffer, polarization_parameters);

    serialize_data(buffer, temperature_parameters);

    serialize_data(buffer, fullenergyspectrum);

    serialize_data(buffer, Curved_Crystal);

    serialize_data(buffer, AnalysiesCrystaltilts.make_matrix_full);
    serialize_data(buffer, AnalysiesCrystaltilts.make_graph_widths);
    serialize_data(buffer, AnalysiesCrystaltilts.metafile);
    serialize_data(buffer, AnalysiesCrystaltilts.make_an_C1_ta);
    serialize_data(buffer, AnalysiesCrystaltilts.make_plot_C1_table);

    return buffer;
}

static void deserialize_inputs(std::vector<uint8_t>& buffer)
{
    consume_buffer(buffer, Geometry);

    consume_buffer(buffer, UserSettings);

    consume_buffer(buffer, GeoParapathlengths.type_source);
    consume_buffer(buffer, GeoParapathlengths.LT_aper);
    consume_buffer(buffer, GeoParapathlengths.dist_T_Cr1);
    consume_buffer(buffer, GeoParapathlengths.dist_Cr1_Cr2);
    consume_buffer(buffer, GeoParapathlengths.dist_Cr2_Det);
    consume_buffer(buffer, GeoParapathlengths.dist_aper_det);

    consume_buffer(buffer, Geolengthelements);

    consume_buffer(buffer, GeoParameters);

    consume_buffer(buffer, CurveVerticalTilt);

    consume_buffer(buffer, Graph_options);

    consume_buffer(buffer, plotparameters);

    consume_buffer(buffer, numberrays);

    consume_buffer(buffer, physical_parameters.Unit_energy);
    consume_buffer(buffer, physical_parameters.linelamda);
    consume_buffer(buffer, physical_parameters.naturalwidth);
    consume_buffer(buffer, physical_parameters.gauss_Doop);

    consume_buffer(buffer, polarization_parameters);

    consume_buffer(buffer, temperature_parameters);

    consume_buffer(buffer, fullenergyspectrum);

    consume_buffer(buffer, Curved_Crystal);

    consume_buffer(buffer, AnalysiesCrystaltilts.make_matrix_full);
    consume_buffer(buffer, AnalysiesCrystaltilts.make_graph_widths);
    consume_buffer(buffer, AnalysiesCrystaltilts.metafile);
    consume_buffer(buffer, AnalysiesCrystaltilts.make_an_C1_ta);
    consume_buffer(buffer, AnalysiesCrystaltilts.make_plot_C1_table);
}

static void write_buffer_to_file(const std::vector<uint8_t>& buffer, const std::string& filename)
{
    std::ofstream stream(filename.c_str(), std::ios::binary);
    if(!stream.is_open())
    {
        std::cout << "Failed to write buffer to file." << std::endl;
        return;
    }

    stream.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
}

static std::vector<uint8_t> read_buffer_from_file(const std::string& filename)
{
    std::vector<uint8_t> buffer;

    std::ifstream stream(filename.c_str(), std::ios::binary);
    if(!stream.is_open())
    {
        std::cout << "Failed to read buffer from file." << std::endl;
        return buffer;
    }

    stream.unsetf(std::ios::skipws);

    stream.seekg(0, std::ios::end);
    std::streampos filesize = stream.tellg();
    stream.seekg(0, std::ios::beg);

    buffer.reserve(filesize);

    std::copy(std::istream_iterator<uint8_t>(stream), std::istream_iterator<uint8_t>(), std::back_inserter(buffer));

    assert(buffer.size() > 0);

    return buffer;
}

extern char File_simu[200];

SimulationMain *wsimu;

DCS_GUI::DCS_GUI(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::DCS_GUI)
{
    ui->setupUi(this);

    wsimu = new SimulationMain();

    QValidator *v = new QDoubleValidator(DBL_MIN, DBL_MAX, 6, this);
    QValidator *vi = new QIntValidator(1, INT_MAX, this);

    ui->energy_val->setValidator(v);
    ui->naturalWidth_val->setValidator(v);
    ui->tempC1P_val->setValidator(v);
    ui->tempC1AP_val->setValidator(v);
    ui->tempC2P_val->setValidator(v);
    ui->tempC2AP_val->setValidator(v);
    ui->dopplerFWHMGauss_val->setValidator(v);

    ui->angleScanRange_val->setValidator(v);
    ui->angleDispShift_val->setValidator(v);
    ui->binNum_val->setValidator(vi);
    ui->rayNum_val->setValidator(vi);

    ui->distSAP_val->setValidator(v);
    ui->distAPC1_val->setValidator(v);
    ui->distC1C2_val->setValidator(v);
    ui->distC2D_val->setValidator(v);
    ui->cuTubeDiam_val->setValidator(v);
    ui->lengthCs_val->setValidator(v);
    ui->heightCs_val->setValidator(v);
    ui->lengthD_val->setValidator(v);
    ui->heightD_val->setValidator(v);
    ui->hshiftTubeS_val->setValidator(v);
    ui->hshiftTubeAP_val->setValidator(v);
    ui->vshiftTubeS_val->setValidator(v);
    ui->vshiftTubeAP_val->setValidator(v);
    ui->angleC1_val->setValidator(v);
    ui->angleOffC1_val->setValidator(v);
    ui->angleTable_val->setValidator(v);
    ui->angleDP_val->setValidator(v);
    ui->angleDAP_val->setValidator(v);
    ui->angleC1tilt_val->setValidator(v);
    ui->angleC2tilt_val->setValidator(v);

    update_ui_text();

    char inFileText[234];

    strcpy(inFileText, "More Settings can be changed at: ");
    strcat(inFileText, File_simu);
    strcat(inFileText, "\\DCrystal_input.input");

    ui->fullSettings_path->setText(QString(inFileText));

}

void DCS_GUI::update_ui_text()
{
    ui->makeP_check->setChecked(UserSettings.see_para);
    ui->makeAP_check->setChecked(UserSettings.see_anti);

    if(UserSettings.mask_C1 == 0)
        ui->C1_noMask_radio->toggle();
    else if(UserSettings.mask_C1 == 1)
        ui->C1_downMask_radio->toggle();
    else if(UserSettings.mask_C1 == 2)
        ui->C1_upMask_radio->toggle();

    if(UserSettings.mask_C2 == 0)
        ui->C2_noMask_radio->toggle();
    else if(UserSettings.mask_C2 == 1)
        ui->C2_downMask_radio->toggle();
    else if(UserSettings.mask_C2 == 2)
        ui->C2_upMask_radio->toggle();

    ui->makeImagePlateC2_check->setChecked(Graph_options.make_imageC1_After_refle);
    ui->makeImagePlateC1_check->setChecked(Graph_options.make_imageC2_After_refle);
    
    ui->energy_val->setText(std::to_string(physical_parameters.linelamda).c_str());
    ui->naturalWidth_val->setText(std::to_string(physical_parameters.naturalwidth).c_str());
    ui->tempC1P_val->setText(std::to_string(temperature_parameters.T_crystal_1_para).c_str());
    ui->tempC1AP_val->setText(std::to_string(temperature_parameters.T_crystal_1_anti).c_str());
    ui->tempC2P_val->setText(std::to_string(temperature_parameters.T_crystal_2_para).c_str());
    ui->tempC2AP_val->setText(std::to_string(temperature_parameters.T_crystal_2_anti).c_str());
    ui->dopplerFWHMGauss_val->setText(std::to_string(physical_parameters.gauss_Doop).c_str());

    ui->angleScanRange_val->setText(std::to_string(plotparameters.delta_angl).c_str());
    ui->angleDispShift_val->setText(std::to_string(plotparameters.shift_disp_window).c_str());
    ui->binNum_val->setText(std::to_string(plotparameters.nubins).c_str());
    ui->rayNum_val->setText(std::to_string(numberrays.nbeams).c_str());

    ui->distSAP_val->setText(std::to_string(GeoParapathlengths.LT_aper).c_str());
    ui->distAPC1_val->setText(std::to_string(GeoParapathlengths.dist_T_Cr1).c_str());
    ui->distC1C2_val->setText(std::to_string(GeoParapathlengths.dist_Cr1_Cr2).c_str());
    ui->distC2D_val->setText(std::to_string(GeoParapathlengths.dist_Cr2_Det).c_str());
    ui->cuTubeDiam_val->setText(std::to_string(Geolengthelements.S_aper).c_str());
    ui->lengthCs_val->setText(std::to_string(Geolengthelements.y_first_crys).c_str());
    ui->heightCs_val->setText(std::to_string(Geolengthelements.z_first_crys).c_str());
    ui->lengthD_val->setText(std::to_string(Geolengthelements.ydetc).c_str());
    ui->heightD_val->setText(std::to_string(Geolengthelements.zdetc).c_str());
    ui->hshiftTubeS_val->setText(std::to_string(Geolengthelements.S_shi_hor_B).c_str());
    ui->hshiftTubeAP_val->setText(std::to_string(Geolengthelements.S_shi_hor_A).c_str());
    ui->vshiftTubeS_val->setText(std::to_string(Geolengthelements.S_shi_ver_B).c_str());
    ui->vshiftTubeAP_val->setText(std::to_string(Geolengthelements.S_shi_ver_A).c_str());
    ui->angleC1_val->setText(std::to_string(GeoParameters.Exp_crys1).c_str());
    ui->angleOffC1_val->setText(std::to_string(GeoParameters.OffsetRotCry1).c_str());
    ui->angleTable_val->setText(std::to_string(GeoParameters.teta_table).c_str());
    ui->angleDP_val->setText(std::to_string(GeoParameters.teta_detec_para).c_str());
    ui->angleDAP_val->setText(std::to_string(GeoParameters.teta_detec_anti).c_str());
    ui->angleC1tilt_val->setText(std::to_string(GeoParameters.tilt_C1).c_str());
    ui->angleC2tilt_val->setText(std::to_string(GeoParameters.tilt_C2).c_str());
}

DCS_GUI::~DCS_GUI()
{
    delete ui;
    exit(0);
}


void DCS_GUI::on_makeP_check_stateChanged(int arg1)
{
    UserSettings.see_para = (bool)arg1;
}

void DCS_GUI::on_makeAP_check_stateChanged(int arg1)
{
    UserSettings.see_anti = (bool)arg1;
}

void DCS_GUI::on_C1_noMask_radio_clicked()
{
    UserSettings.mask_C1 = 0;
}

void DCS_GUI::on_C1_downMask_radio_clicked()
{
    UserSettings.mask_C1 = 1;
}

void DCS_GUI::on_C1_upMask_radio_clicked()
{
    UserSettings.mask_C1 = 2;
}

void DCS_GUI::on_C2_noMask_radio_clicked()
{
    UserSettings.mask_C2 = 0;
}

void DCS_GUI::on_C2_downMask_radio_clicked()
{
    UserSettings.mask_C2 = 1;
}

void DCS_GUI::on_C2_upMask_radio_clicked()
{
    UserSettings.mask_C2 = 2;
}

void DCS_GUI::on_makeImagePlateC1_check_stateChanged(int arg1)
{
    Graph_options.make_imageC1_After_refle = (bool) arg1;
}

void DCS_GUI::on_makeImagePlateC2_check_stateChanged(int arg1)
{
    Graph_options.make_imageC2_After_refle = (bool) arg1;
}

void DCS_GUI::on_startSim_button_clicked()
{
    DCS_GUI::close();
    wsimu->show();
}

void DCS_GUI::on_saveSim_button_clicked()
{
    QString filename = QFileDialog::getSaveFileName(this,
        tr("Save Simulation Settings"),
        tr("simu_settings.dsp"),
        tr("DCSsimu params (*.dsp)")
    );

    if(filename.isEmpty())
    {
        return;
    }

    auto buffer = serialize_inputs();
    write_buffer_to_file(buffer, filename.toStdString());
}

void DCS_GUI::on_loadSim_button_clicked()
{
    QString filename = QFileDialog::getOpenFileName(this,
        tr("Open Simulation Settings"),
        QString(),
        tr("DCSsimu params (*.dsp)")
    );

    if(filename.isEmpty())
    {
        return;
    }

    auto buffer = read_buffer_from_file(filename.toStdString());
    deserialize_inputs(buffer);

    update_ui_text();
}

void DCS_GUI::on_energy_val_textEdited(const QString &arg1)
{
    physical_parameters.linelamda = std::stod(arg1.toStdString());
}

void DCS_GUI::on_naturalWidth_val_textEdited(const QString &arg1)
{
    physical_parameters.naturalwidth = std::stod(arg1.toStdString());
}

void DCS_GUI::on_tempC1P_val_textEdited(const QString &arg1)
{
    temperature_parameters.T_crystal_1_para = std::stod(arg1.toStdString());
}

void DCS_GUI::on_tempC1AP_val_textEdited(const QString &arg1)
{
    temperature_parameters.T_crystal_1_anti = std::stod(arg1.toStdString());
}

void DCS_GUI::on_tempC2P_val_textEdited(const QString &arg1)
{
    temperature_parameters.T_crystal_2_para = std::stod(arg1.toStdString());
}

void DCS_GUI::on_tempC2AP_val_textEdited(const QString &arg1)
{
    temperature_parameters.T_crystal_2_anti = std::stod(arg1.toStdString());
}

void DCS_GUI::on_dopplerFWHMGauss_val_textEdited(const QString &arg1)
{
    physical_parameters.gauss_Doop = std::stod(arg1.toStdString());
}

void DCS_GUI::on_angleScanRange_val_textEdited(const QString &arg1)
{
    plotparameters.delta_angl = std::stod(arg1.toStdString());
}

void DCS_GUI::on_angleDispShift_val_textEdited(const QString &arg1)
{
    plotparameters.shift_disp_window = std::stod(arg1.toStdString());
}

void DCS_GUI::on_binNum_val_textEdited(const QString &arg1)
{
    plotparameters.nubins = stoi(arg1.toStdString());
}

void DCS_GUI::on_rayNum_val_textEdited(const QString &arg1)
{
    numberrays.nbeams = stoi(arg1.toStdString());
}

void DCS_GUI::on_distSAP_val_textEdited(const QString &arg1)
{
    GeoParapathlengths.LT_aper = std::stod(arg1.toStdString());
}

void DCS_GUI::on_distAPC1_val_textEdited(const QString &arg1)
{
    GeoParapathlengths.dist_T_Cr1 = std::stod(arg1.toStdString());
}

void DCS_GUI::on_distC1C2_val_textEdited(const QString &arg1)
{
    GeoParapathlengths.dist_Cr1_Cr2 = std::stod(arg1.toStdString());
}

void DCS_GUI::on_distC2D_val_textEdited(const QString &arg1)
{
    GeoParapathlengths.dist_Cr2_Det = std::stod(arg1.toStdString());
}

void DCS_GUI::on_cuTubeDiam_val_textEdited(const QString &arg1)
{
    Geolengthelements.S_aper = std::stod(arg1.toStdString());
}

void DCS_GUI::on_lengthCs_val_textEdited(const QString &arg1)
{
    Geolengthelements.y_first_crys = std::stod(arg1.toStdString());
}

void DCS_GUI::on_heightCs_val_textEdited(const QString &arg1)
{
    Geolengthelements.z_first_crys = std::stod(arg1.toStdString());
}

void DCS_GUI::on_lengthD_val_textEdited(const QString &arg1)
{
    Geolengthelements.ydetc = std::stod(arg1.toStdString());
}

void DCS_GUI::on_heightD_val_textEdited(const QString &arg1)
{
    Geolengthelements.zdetc = std::stod(arg1.toStdString());
}

void DCS_GUI::on_hshiftTubeS_val_textEdited(const QString &arg1)
{
    Geolengthelements.S_shi_hor_B = std::stod(arg1.toStdString());
}

void DCS_GUI::on_hshiftTubeAP_val_textEdited(const QString &arg1)
{
    Geolengthelements.S_shi_hor_A = std::stod(arg1.toStdString());
}

void DCS_GUI::on_vshiftTubeS_val_textEdited(const QString &arg1)
{
    Geolengthelements.S_shi_ver_B = std::stod(arg1.toStdString());
}

void DCS_GUI::on_vshiftTubeAP_val_textEdited(const QString &arg1)
{
    Geolengthelements.S_shi_ver_A = std::stod(arg1.toStdString());
}

void DCS_GUI::on_angleC1_val_textEdited(const QString &arg1)
{
    GeoParameters.Exp_crys1 = std::stod(arg1.toStdString());
}

void DCS_GUI::on_angleOffC1_val_textEdited(const QString &arg1)
{
    GeoParameters.OffsetRotCry1 = std::stod(arg1.toStdString());
}

void DCS_GUI::on_angleTable_val_textEdited(const QString &arg1)
{
    GeoParameters.teta_table = std::stod(arg1.toStdString());
}

void DCS_GUI::on_angleDP_val_textEdited(const QString &arg1)
{
    GeoParameters.teta_detec_para = std::stod(arg1.toStdString());
}

void DCS_GUI::on_angleDAP_val_textEdited(const QString &arg1)
{
    GeoParameters.teta_detec_anti = std::stod(arg1.toStdString());
}

void DCS_GUI::on_angleC1tilt_val_textEdited(const QString &arg1)
{
    GeoParameters.tilt_C1 = std::stod(arg1.toStdString());
}

void DCS_GUI::on_angleC2tilt_val_textEdited(const QString &arg1)
{
    GeoParameters.tilt_C2 = std::stod(arg1.toStdString());
}
