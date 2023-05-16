#ifndef DCS_GUI_H
#define DCS_GUI_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE
namespace Ui { class DCS_GUI; }
QT_END_NAMESPACE

class DCS_GUI : public QMainWindow
{
    Q_OBJECT

public:
    DCS_GUI(QWidget *parent = nullptr);
    ~DCS_GUI();

    

private slots:
    void on_makeP_check_stateChanged(int arg1);

    void on_makeAP_check_stateChanged(int arg1);

    void on_C1_noMask_radio_clicked();

    void on_C1_downMask_radio_clicked();

    void on_C1_upMask_radio_clicked();

    void on_C2_noMask_radio_clicked();

    void on_C2_downMask_radio_clicked();

    void on_C2_upMask_radio_clicked();

    void on_makeImagePlateC1_check_stateChanged(int arg1);

    void on_makeImagePlateC2_check_stateChanged(int arg1);

    void on_startSim_button_clicked();

    void on_saveSim_button_clicked();

    void on_loadSim_button_clicked();

    void on_energy_val_textEdited(const QString &arg1);

    void on_naturalWidth_val_textEdited(const QString &arg1);

    void on_tempC1P_val_textEdited(const QString &arg1);

    void on_tempC1AP_val_textEdited(const QString &arg1);

    void on_tempC2P_val_textEdited(const QString &arg1);

    void on_tempC2AP_val_textEdited(const QString &arg1);

    void on_dopplerFWHMGauss_val_textEdited(const QString &arg1);

    void on_angleScanRange_val_textEdited(const QString &arg1);

    void on_angleDispShift_val_textEdited(const QString &arg1);

    void on_binNum_val_textEdited(const QString &arg1);

    void on_rayNum_val_textEdited(const QString &arg1);

    void on_distSAP_val_textEdited(const QString &arg1);

    void on_distAPC1_val_textEdited(const QString &arg1);

    void on_distC1C2_val_textEdited(const QString &arg1);

    void on_distC2D_val_textEdited(const QString &arg1);

    void on_cuTubeDiam_val_textEdited(const QString &arg1);

    void on_lengthCs_val_textEdited(const QString &arg1);

    void on_heightCs_val_textEdited(const QString &arg1);

    void on_lengthD_val_textEdited(const QString &arg1);

    void on_heightD_val_textEdited(const QString &arg1);

    void on_hshiftTubeS_val_textEdited(const QString &arg1);

    void on_hshiftTubeAP_val_textEdited(const QString &arg1);

    void on_vshiftTubeS_val_textEdited(const QString &arg1);

    void on_vshiftTubeAP_val_textEdited(const QString &arg1);

    void on_angleC1_val_textEdited(const QString &arg1);

    void on_angleOffC1_val_textEdited(const QString &arg1);

    void on_angleTable_val_textEdited(const QString &arg1);

    void on_angleDP_val_textEdited(const QString &arg1);

    void on_angleDAP_val_textEdited(const QString &arg1);

    void on_angleC1tilt_val_textEdited(const QString &arg1);

    void on_angleC2tilt_val_textEdited(const QString &arg1);

private:
    Ui::DCS_GUI *ui;

    void update_ui_text();
};
#endif // DCS_GUI_H
