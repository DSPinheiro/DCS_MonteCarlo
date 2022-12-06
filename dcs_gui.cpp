#include <float.h>

#include <QDoubleValidator>

#include "dcs_gui.h"
#include "ui_dcs_gui.h"

#include "simulationmain.h"

using namespace std;


DCS_GUI::DCS_GUI(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::DCS_GUI)
{
    ui->setupUi(this);

    ui->makeP_check->setChecked(UserSettingsInput.see_para);
    ui->makeAP_check->setChecked(UserSettingsInput.see_anti);

    if(UserSettingsInput.mask_C1 == 0)
        ui->C1_noMask_radio->toggle();
    else if(UserSettingsInput.mask_C1 == 1)
        ui->C1_downMask_radio->toggle();
    else if(UserSettingsInput.mask_C1 == 2)
        ui->C1_upMask_radio->toggle();

    if(UserSettingsInput.mask_C2 == 0)
        ui->C2_noMask_radio->toggle();
    else if(UserSettingsInput.mask_C2 == 1)
        ui->C2_downMask_radio->toggle();
    else if(UserSettingsInput.mask_C2 == 2)
        ui->C2_upMask_radio->toggle();

    ui->makeImagePlateC2_check->setChecked(GraphOptionsInput.make_imageC1_After_refle);
    ui->makeImagePlateC1_check->setChecked(GraphOptionsInput.make_imageC2_After_refle);

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

    ui->energy_val->setText(to_string(PhysicalParametersInput.linelamda).c_str());
    ui->naturalWidth_val->setText(to_string(PhysicalParametersInput.naturalwidth).c_str());
    ui->tempC1P_val->setText(to_string(TemperatureParametersInput.T_crystal_1_para).c_str());
    ui->tempC1AP_val->setText(to_string(TemperatureParametersInput.T_crystal_1_anti).c_str());
    ui->tempC2P_val->setText(to_string(TemperatureParametersInput.T_crystal_2_para).c_str());
    ui->tempC2AP_val->setText(to_string(TemperatureParametersInput.T_crystal_2_anti).c_str());
    ui->dopplerFWHMGauss_val->setText(to_string(PhysicalParametersInput.gauss_Doop).c_str());

    ui->angleScanRange_val->setText(to_string(PlotParametersInput.delta_angl).c_str());
    ui->angleDispShift_val->setText(to_string(PlotParametersInput.shift_disp_window).c_str());
    ui->binNum_val->setText(to_string(PlotParametersInput.nubins).c_str());
    ui->rayNum_val->setText(to_string(NumberRaysInput.nbeams).c_str());

    ui->distSAP_val->setText(to_string(GeoParapathlengthsInput.LT_aper).c_str());
    ui->distAPC1_val->setText(to_string(GeoParapathlengthsInput.dist_T_Cr1).c_str());
    ui->distC1C2_val->setText(to_string(GeoParapathlengthsInput.dist_Cr1_Cr2).c_str());
    ui->distC2D_val->setText(to_string(GeoParapathlengthsInput.dist_Cr2_Det).c_str());
    ui->cuTubeDiam_val->setText(to_string(GeolengthelementsInput.S_aper).c_str());
    ui->lengthCs_val->setText(to_string(GeolengthelementsInput.y_first_crys).c_str());
    ui->heightCs_val->setText(to_string(GeolengthelementsInput.z_first_crys).c_str());
    ui->lengthD_val->setText(to_string(GeolengthelementsInput.ydetc).c_str());
    ui->heightD_val->setText(to_string(GeolengthelementsInput.zdetc).c_str());
    ui->hshiftTubeS_val->setText(to_string(GeolengthelementsInput.S_shi_hor_B).c_str());
    ui->hshiftTubeAP_val->setText(to_string(GeolengthelementsInput.S_shi_hor_A).c_str());
    ui->vshiftTubeS_val->setText(to_string(GeolengthelementsInput.S_shi_ver_B).c_str());
    ui->vshiftTubeAP_val->setText(to_string(GeolengthelementsInput.S_shi_ver_A).c_str());
    ui->angleC1_val->setText(to_string(GeoParametersInput.Exp_crys1).c_str());
    ui->angleOffC1_val->setText(to_string(GeoParametersInput.OffsetRotCry1).c_str());
    ui->angleTable_val->setText(to_string(GeoParametersInput.teta_table).c_str());
    ui->angleDP_val->setText(to_string(GeoParametersInput.teta_detec_para).c_str());
    ui->angleDAP_val->setText(to_string(GeoParametersInput.teta_detec_anti).c_str());
    ui->angleC1tilt_val->setText(to_string(GeoParametersInput.tilt_C1).c_str());
    ui->angleC2tilt_val->setText(to_string(GeoParametersInput.tilt_C2).c_str());

    char inFileText[234];

    strcpy(inFileText, "More Settings can be changed at: ");
    strcat(inFileText, File_simu);
    strcat(inFileText, "\\DCrystal_input.input");

    ui->fullSettings_path->setText(QString(inFileText));

}

DCS_GUI::~DCS_GUI()
{
    delete ui;
    exit(0);
}


void DCS_GUI::on_makeP_check_stateChanged(int arg1)
{
    UserSettingsInput.see_para = (bool)arg1;
}

void DCS_GUI::on_makeAP_check_stateChanged(int arg1)
{
    UserSettingsInput.see_anti = (bool)arg1;
}

void DCS_GUI::on_C1_noMask_radio_clicked()
{
    UserSettingsInput.mask_C1 = 0;
}

void DCS_GUI::on_C1_downMask_radio_clicked()
{
    UserSettingsInput.mask_C1 = 1;
}

void DCS_GUI::on_C1_upMask_radio_clicked()
{
    UserSettingsInput.mask_C1 = 2;
}

void DCS_GUI::on_C2_noMask_radio_clicked()
{
    UserSettingsInput.mask_C2 = 0;
}

void DCS_GUI::on_C2_downMask_radio_clicked()
{
    UserSettingsInput.mask_C2 = 1;
}

void DCS_GUI::on_C2_upMask_radio_clicked()
{
    UserSettingsInput.mask_C2 = 2;
}

void DCS_GUI::on_makeImagePlateC1_check_stateChanged(int arg1)
{
    GraphOptionsInput.make_imageC1_After_refle = (bool) arg1;
}

void DCS_GUI::on_makeImagePlateC2_check_stateChanged(int arg1)
{
    GraphOptionsInput.make_imageC2_After_refle = (bool) arg1;
}

void DCS_GUI::on_energy_val_textEdited(const QString &arg1)
{
    PhysicalParametersInput.linelamda = stod(arg1.toStdString());
}

void DCS_GUI::on_naturalWidth_val_textEdited(const QString &arg1)
{
    PhysicalParametersInput.naturalwidth = stod(arg1.toStdString());
}

void DCS_GUI::on_tempC1P_val_textEdited(const QString &arg1)
{
    TemperatureParametersInput.T_crystal_1_para = stod(arg1.toStdString());
}

void DCS_GUI::on_tempC1AP_val_textEdited(const QString &arg1)
{
    TemperatureParametersInput.T_crystal_1_anti = stod(arg1.toStdString());
}

void DCS_GUI::on_tempC2P_val_textEdited(const QString &arg1)
{
    TemperatureParametersInput.T_crystal_2_para = stod(arg1.toStdString());
}

void DCS_GUI::on_tempC2AP_val_textEdited(const QString &arg1)
{
    TemperatureParametersInput.T_crystal_2_anti = stod(arg1.toStdString());
}

void DCS_GUI::on_dopplerFWHMGauss_val_textEdited(const QString &arg1)
{
    PhysicalParametersInput.gauss_Doop = stod(arg1.toStdString());
}

void DCS_GUI::on_angleScanRange_val_textEdited(const QString &arg1)
{
    PlotParametersInput.delta_angl = stod(arg1.toStdString());
}

void DCS_GUI::on_angleDispShift_val_textEdited(const QString &arg1)
{
    PlotParametersInput.shift_disp_window = stod(arg1.toStdString());
}

void DCS_GUI::on_binNum_val_textEdited(const QString &arg1)
{
    PlotParametersInput.nubins = stoi(arg1.toStdString());
}

void DCS_GUI::on_rayNum_val_textEdited(const QString &arg1)
{
    NumberRaysInput.nbeams = stoi(arg1.toStdString());
}

void DCS_GUI::on_distSAP_val_textEdited(const QString &arg1)
{
    GeoParapathlengthsInput.LT_aper = stod(arg1.toStdString());
}

void DCS_GUI::on_distAPC1_val_textEdited(const QString &arg1)
{
    GeoParapathlengthsInput.dist_T_Cr1 = stod(arg1.toStdString());
}

void DCS_GUI::on_distC1C2_val_textEdited(const QString &arg1)
{
    GeoParapathlengthsInput.dist_Cr1_Cr2 = stod(arg1.toStdString());
}

void DCS_GUI::on_distC2D_val_textEdited(const QString &arg1)
{
    GeoParapathlengthsInput.dist_Cr2_Det = stod(arg1.toStdString());
}

void DCS_GUI::on_cuTubeDiam_val_textEdited(const QString &arg1)
{
    GeolengthelementsInput.S_aper = stod(arg1.toStdString());
}

void DCS_GUI::on_lengthCs_val_textEdited(const QString &arg1)
{
    GeolengthelementsInput.y_first_crys = stod(arg1.toStdString());
}

void DCS_GUI::on_heightCs_val_textEdited(const QString &arg1)
{
    GeolengthelementsInput.z_first_crys = stod(arg1.toStdString());
}

void DCS_GUI::on_lengthD_val_textEdited(const QString &arg1)
{
    GeolengthelementsInput.ydetc = stod(arg1.toStdString());
}

void DCS_GUI::on_heightD_val_textEdited(const QString &arg1)
{
    GeolengthelementsInput.zdetc = stod(arg1.toStdString());
}

void DCS_GUI::on_hshiftTubeS_val_textEdited(const QString &arg1)
{
    GeolengthelementsInput.S_shi_hor_B = stod(arg1.toStdString());
}

void DCS_GUI::on_hshiftTubeAP_val_textEdited(const QString &arg1)
{
    GeolengthelementsInput.S_shi_hor_A = stod(arg1.toStdString());
}

void DCS_GUI::on_vshiftTubeS_val_textEdited(const QString &arg1)
{
    GeolengthelementsInput.S_shi_ver_B = stod(arg1.toStdString());
}

void DCS_GUI::on_vshiftTubeAP_val_textEdited(const QString &arg1)
{
    GeolengthelementsInput.S_shi_ver_A = stod(arg1.toStdString());
}

void DCS_GUI::on_angleC1_val_textEdited(const QString &arg1)
{
    GeoParametersInput.Exp_crys1 = stod(arg1.toStdString());
}

void DCS_GUI::on_angleOffC1_val_textEdited(const QString &arg1)
{
    GeoParametersInput.OffsetRotCry1 = stod(arg1.toStdString());
}

void DCS_GUI::on_angleTable_val_textEdited(const QString &arg1)
{
    GeoParametersInput.teta_table = stod(arg1.toStdString());
}

void DCS_GUI::on_angleDP_val_textEdited(const QString &arg1)
{
    GeoParametersInput.teta_detec_para = stod(arg1.toStdString());
}

void DCS_GUI::on_angleDAP_val_textEdited(const QString &arg1)
{
    GeoParametersInput.teta_detec_anti = stod(arg1.toStdString());
}

void DCS_GUI::on_angleC1tilt_val_textEdited(const QString &arg1)
{
    GeoParametersInput.tilt_C1 = stod(arg1.toStdString());
}

void DCS_GUI::on_angleC2tilt_val_textEdited(const QString &arg1)
{
    GeoParametersInput.tilt_C2 = stod(arg1.toStdString());
}


void DCS_GUI::on_startSim_button_clicked()
{
    DCS_GUI::close();
    SimulationMain* wsimu = new SimulationMain(nullptr);
    wsimu->show();
}