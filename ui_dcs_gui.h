/********************************************************************************
** Form generated from reading UI file 'dcs_gui.ui'
**
** Created by: Qt User Interface Compiler version 5.15.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_DCS_GUI_H
#define UI_DCS_GUI_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QSplitter>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_DCS_GUI
{
public:
    QWidget *centralwidget;
    QGridLayout *gridLayout;
    QLabel *physicalSettings_label;
    QLabel *fullSettings_path;
    QPushButton *startSim_button;
    QSplitter *geometricalSettings_splitter;
    QWidget *layoutWidget;
    QVBoxLayout *geometricalSettingsLabels;
    QLabel *distSAP_label;
    QLabel *distAPC1_label;
    QLabel *distC1C2_label;
    QLabel *distC2D_label;
    QLabel *cuTubeDiam_label;
    QLabel *lengthCs_label;
    QLabel *heightCs_label;
    QLabel *lengthD_label;
    QLabel *heightD_label;
    QLabel *hshiftTubeS_label;
    QLabel *hshiftTubeAP_label;
    QLabel *vshiftTubeS_label;
    QLabel *vshiftTubeAP_label;
    QLabel *angleC1_label;
    QLabel *angleOffC1_label;
    QLabel *angleTable_label;
    QLabel *angleDP_label;
    QLabel *angleDAP_label;
    QLabel *angleC1tilt_label;
    QLabel *angleC2tilt_label;
    QWidget *layoutWidget1;
    QVBoxLayout *geometricalSettingsVals;
    QLineEdit *distSAP_val;
    QLineEdit *distAPC1_val;
    QLineEdit *distC1C2_val;
    QLineEdit *distC2D_val;
    QLineEdit *cuTubeDiam_val;
    QLineEdit *lengthCs_val;
    QLineEdit *heightCs_val;
    QLineEdit *lengthD_val;
    QLineEdit *heightD_val;
    QLineEdit *hshiftTubeS_val;
    QLineEdit *hshiftTubeAP_val;
    QLineEdit *vshiftTubeS_val;
    QLineEdit *vshiftTubeAP_val;
    QLineEdit *angleC1_val;
    QLineEdit *angleOffC1_val;
    QLineEdit *angleTable_val;
    QLineEdit *angleDP_val;
    QLineEdit *angleDAP_val;
    QLineEdit *angleC1tilt_val;
    QLineEdit *angleC2tilt_val;
    QSplitter *physicalSettings_splitter;
    QWidget *layoutWidget2;
    QVBoxLayout *physicalSettingsLabels;
    QLabel *energy_label;
    QLabel *naturalWidth_label;
    QLabel *tempC1P_label;
    QLabel *tempC1AP_label;
    QLabel *tempC2P_label;
    QLabel *tempC2AP_label;
    QLabel *dopplerFWHMGauss_label;
    QWidget *layoutWidget3;
    QVBoxLayout *physicalSettingsVals;
    QLineEdit *energy_val;
    QLineEdit *naturalWidth_val;
    QLineEdit *tempC1P_val;
    QLineEdit *tempC1AP_val;
    QLineEdit *tempC2P_val;
    QLineEdit *tempC2AP_val;
    QLineEdit *dopplerFWHMGauss_val;
    QLabel *plotParameters_label;
    QSplitter *plotParameters_splitter;
    QWidget *layoutWidget4;
    QVBoxLayout *plotParametersLabels;
    QLabel *angleScanRange_label;
    QLabel *angleDispShift_label;
    QLabel *binNum_label;
    QLabel *rayNum_label;
    QWidget *layoutWidget5;
    QVBoxLayout *plotParametersVals;
    QLineEdit *angleScanRange_val;
    QLineEdit *angleDispShift_val;
    QLineEdit *binNum_val;
    QLineEdit *rayNum_val;
    QLabel *geometricalSettings_label;
    QLabel *userSettings_label;
    QSplitter *userSettings_splitter;
    QSplitter *crystalSettings_splitter;
    QSplitter *C1Settings_splitter;
    QWidget *layoutWidget6;
    QVBoxLayout *spectraSimTypeLayout;
    QCheckBox *makeP_check;
    QCheckBox *makeAP_check;
    QWidget *layoutWidget7;
    QVBoxLayout *C1MaskSettingsLayout;
    QLabel *C1Mask_label;
    QVBoxLayout *verticalLayout;
    QRadioButton *C1_noMask_radio;
    QRadioButton *C1_downMask_radio;
    QRadioButton *C1_upMask_radio;
    QWidget *layoutWidget8;
    QVBoxLayout *C2MaskSettingsLayout;
    QLabel *C2Mask_label;
    QVBoxLayout *verticalLayout_2;
    QRadioButton *C2_noMask_radio;
    QRadioButton *C2_downMask_radio;
    QRadioButton *C2_upMask_radio;
    QWidget *layoutWidget9;
    QVBoxLayout *makeImagePlatesLayout;
    QCheckBox *makeImagePlateC1_check;
    QCheckBox *makeImagePlateC2_check;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *DCS_GUI)
    {
        if (DCS_GUI->objectName().isEmpty())
            DCS_GUI->setObjectName(QString::fromUtf8("DCS_GUI"));
        DCS_GUI->setWindowModality(Qt::ApplicationModal);
        DCS_GUI->resize(1044, 591);
        centralwidget = new QWidget(DCS_GUI);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        QSizePolicy sizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
        sizePolicy.setHorizontalStretch(1);
        sizePolicy.setVerticalStretch(1);
        sizePolicy.setHeightForWidth(centralwidget->sizePolicy().hasHeightForWidth());
        centralwidget->setSizePolicy(sizePolicy);
        centralwidget->setMinimumSize(QSize(1044, 569));
        gridLayout = new QGridLayout(centralwidget);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        gridLayout->setSizeConstraint(QLayout::SetNoConstraint);
        physicalSettings_label = new QLabel(centralwidget);
        physicalSettings_label->setObjectName(QString::fromUtf8("physicalSettings_label"));

        gridLayout->addWidget(physicalSettings_label, 0, 1, 1, 1);

        fullSettings_path = new QLabel(centralwidget);
        fullSettings_path->setObjectName(QString::fromUtf8("fullSettings_path"));

        gridLayout->addWidget(fullSettings_path, 4, 0, 1, 2);

        startSim_button = new QPushButton(centralwidget);
        startSim_button->setObjectName(QString::fromUtf8("startSim_button"));
        QFont font;
        font.setPointSize(24);
        font.setBold(true);
        startSim_button->setFont(font);
        startSim_button->setStyleSheet(QString::fromUtf8("border-color: rgb(0, 0, 0);\n"
"background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(0, 100, 0, 255), stop:1 rgba(0, 170, 0, 255));"));

        gridLayout->addWidget(startSim_button, 5, 0, 1, 2);

        geometricalSettings_splitter = new QSplitter(centralwidget);
        geometricalSettings_splitter->setObjectName(QString::fromUtf8("geometricalSettings_splitter"));
        sizePolicy.setHeightForWidth(geometricalSettings_splitter->sizePolicy().hasHeightForWidth());
        geometricalSettings_splitter->setSizePolicy(sizePolicy);
        geometricalSettings_splitter->setMinimumSize(QSize(311, 511));
        geometricalSettings_splitter->setMaximumSize(QSize(16777215, 16777215));
        geometricalSettings_splitter->setOrientation(Qt::Horizontal);
        geometricalSettings_splitter->setOpaqueResize(false);
        geometricalSettings_splitter->setChildrenCollapsible(false);
        layoutWidget = new QWidget(geometricalSettings_splitter);
        layoutWidget->setObjectName(QString::fromUtf8("layoutWidget"));
        geometricalSettingsLabels = new QVBoxLayout(layoutWidget);
        geometricalSettingsLabels->setSpacing(0);
        geometricalSettingsLabels->setObjectName(QString::fromUtf8("geometricalSettingsLabels"));
        geometricalSettingsLabels->setSizeConstraint(QLayout::SetNoConstraint);
        geometricalSettingsLabels->setContentsMargins(0, 0, 0, 0);
        distSAP_label = new QLabel(layoutWidget);
        distSAP_label->setObjectName(QString::fromUtf8("distSAP_label"));

        geometricalSettingsLabels->addWidget(distSAP_label);

        distAPC1_label = new QLabel(layoutWidget);
        distAPC1_label->setObjectName(QString::fromUtf8("distAPC1_label"));

        geometricalSettingsLabels->addWidget(distAPC1_label);

        distC1C2_label = new QLabel(layoutWidget);
        distC1C2_label->setObjectName(QString::fromUtf8("distC1C2_label"));

        geometricalSettingsLabels->addWidget(distC1C2_label);

        distC2D_label = new QLabel(layoutWidget);
        distC2D_label->setObjectName(QString::fromUtf8("distC2D_label"));

        geometricalSettingsLabels->addWidget(distC2D_label);

        cuTubeDiam_label = new QLabel(layoutWidget);
        cuTubeDiam_label->setObjectName(QString::fromUtf8("cuTubeDiam_label"));

        geometricalSettingsLabels->addWidget(cuTubeDiam_label);

        lengthCs_label = new QLabel(layoutWidget);
        lengthCs_label->setObjectName(QString::fromUtf8("lengthCs_label"));

        geometricalSettingsLabels->addWidget(lengthCs_label);

        heightCs_label = new QLabel(layoutWidget);
        heightCs_label->setObjectName(QString::fromUtf8("heightCs_label"));

        geometricalSettingsLabels->addWidget(heightCs_label);

        lengthD_label = new QLabel(layoutWidget);
        lengthD_label->setObjectName(QString::fromUtf8("lengthD_label"));

        geometricalSettingsLabels->addWidget(lengthD_label);

        heightD_label = new QLabel(layoutWidget);
        heightD_label->setObjectName(QString::fromUtf8("heightD_label"));

        geometricalSettingsLabels->addWidget(heightD_label);

        hshiftTubeS_label = new QLabel(layoutWidget);
        hshiftTubeS_label->setObjectName(QString::fromUtf8("hshiftTubeS_label"));

        geometricalSettingsLabels->addWidget(hshiftTubeS_label);

        hshiftTubeAP_label = new QLabel(layoutWidget);
        hshiftTubeAP_label->setObjectName(QString::fromUtf8("hshiftTubeAP_label"));

        geometricalSettingsLabels->addWidget(hshiftTubeAP_label);

        vshiftTubeS_label = new QLabel(layoutWidget);
        vshiftTubeS_label->setObjectName(QString::fromUtf8("vshiftTubeS_label"));

        geometricalSettingsLabels->addWidget(vshiftTubeS_label);

        vshiftTubeAP_label = new QLabel(layoutWidget);
        vshiftTubeAP_label->setObjectName(QString::fromUtf8("vshiftTubeAP_label"));

        geometricalSettingsLabels->addWidget(vshiftTubeAP_label);

        angleC1_label = new QLabel(layoutWidget);
        angleC1_label->setObjectName(QString::fromUtf8("angleC1_label"));

        geometricalSettingsLabels->addWidget(angleC1_label);

        angleOffC1_label = new QLabel(layoutWidget);
        angleOffC1_label->setObjectName(QString::fromUtf8("angleOffC1_label"));

        geometricalSettingsLabels->addWidget(angleOffC1_label);

        angleTable_label = new QLabel(layoutWidget);
        angleTable_label->setObjectName(QString::fromUtf8("angleTable_label"));

        geometricalSettingsLabels->addWidget(angleTable_label);

        angleDP_label = new QLabel(layoutWidget);
        angleDP_label->setObjectName(QString::fromUtf8("angleDP_label"));

        geometricalSettingsLabels->addWidget(angleDP_label);

        angleDAP_label = new QLabel(layoutWidget);
        angleDAP_label->setObjectName(QString::fromUtf8("angleDAP_label"));

        geometricalSettingsLabels->addWidget(angleDAP_label);

        angleC1tilt_label = new QLabel(layoutWidget);
        angleC1tilt_label->setObjectName(QString::fromUtf8("angleC1tilt_label"));

        geometricalSettingsLabels->addWidget(angleC1tilt_label);

        angleC2tilt_label = new QLabel(layoutWidget);
        angleC2tilt_label->setObjectName(QString::fromUtf8("angleC2tilt_label"));

        geometricalSettingsLabels->addWidget(angleC2tilt_label);

        geometricalSettingsLabels->setStretch(0, 1);
        geometricalSettingsLabels->setStretch(1, 1);
        geometricalSettingsLabels->setStretch(2, 1);
        geometricalSettingsLabels->setStretch(3, 1);
        geometricalSettingsLabels->setStretch(4, 1);
        geometricalSettingsLabels->setStretch(5, 1);
        geometricalSettingsLabels->setStretch(6, 1);
        geometricalSettingsLabels->setStretch(7, 1);
        geometricalSettingsLabels->setStretch(8, 1);
        geometricalSettingsLabels->setStretch(9, 1);
        geometricalSettingsLabels->setStretch(10, 1);
        geometricalSettingsLabels->setStretch(11, 1);
        geometricalSettingsLabels->setStretch(12, 1);
        geometricalSettingsLabels->setStretch(13, 1);
        geometricalSettingsLabels->setStretch(14, 1);
        geometricalSettingsLabels->setStretch(15, 1);
        geometricalSettingsLabels->setStretch(16, 1);
        geometricalSettingsLabels->setStretch(17, 1);
        geometricalSettingsLabels->setStretch(18, 1);
        geometricalSettingsLabels->setStretch(19, 1);
        geometricalSettings_splitter->addWidget(layoutWidget);
        layoutWidget1 = new QWidget(geometricalSettings_splitter);
        layoutWidget1->setObjectName(QString::fromUtf8("layoutWidget1"));
        geometricalSettingsVals = new QVBoxLayout(layoutWidget1);
        geometricalSettingsVals->setObjectName(QString::fromUtf8("geometricalSettingsVals"));
        geometricalSettingsVals->setContentsMargins(0, 0, 0, 0);
        distSAP_val = new QLineEdit(layoutWidget1);
        distSAP_val->setObjectName(QString::fromUtf8("distSAP_val"));
        QSizePolicy sizePolicy1(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(distSAP_val->sizePolicy().hasHeightForWidth());
        distSAP_val->setSizePolicy(sizePolicy1);
        distSAP_val->setMaximumSize(QSize(100, 20));

        geometricalSettingsVals->addWidget(distSAP_val);

        distAPC1_val = new QLineEdit(layoutWidget1);
        distAPC1_val->setObjectName(QString::fromUtf8("distAPC1_val"));
        sizePolicy1.setHeightForWidth(distAPC1_val->sizePolicy().hasHeightForWidth());
        distAPC1_val->setSizePolicy(sizePolicy1);
        distAPC1_val->setMaximumSize(QSize(100, 20));

        geometricalSettingsVals->addWidget(distAPC1_val);

        distC1C2_val = new QLineEdit(layoutWidget1);
        distC1C2_val->setObjectName(QString::fromUtf8("distC1C2_val"));
        sizePolicy1.setHeightForWidth(distC1C2_val->sizePolicy().hasHeightForWidth());
        distC1C2_val->setSizePolicy(sizePolicy1);
        distC1C2_val->setMaximumSize(QSize(100, 20));

        geometricalSettingsVals->addWidget(distC1C2_val);

        distC2D_val = new QLineEdit(layoutWidget1);
        distC2D_val->setObjectName(QString::fromUtf8("distC2D_val"));
        sizePolicy1.setHeightForWidth(distC2D_val->sizePolicy().hasHeightForWidth());
        distC2D_val->setSizePolicy(sizePolicy1);
        distC2D_val->setMaximumSize(QSize(100, 20));

        geometricalSettingsVals->addWidget(distC2D_val);

        cuTubeDiam_val = new QLineEdit(layoutWidget1);
        cuTubeDiam_val->setObjectName(QString::fromUtf8("cuTubeDiam_val"));
        sizePolicy1.setHeightForWidth(cuTubeDiam_val->sizePolicy().hasHeightForWidth());
        cuTubeDiam_val->setSizePolicy(sizePolicy1);
        cuTubeDiam_val->setMaximumSize(QSize(100, 20));

        geometricalSettingsVals->addWidget(cuTubeDiam_val);

        lengthCs_val = new QLineEdit(layoutWidget1);
        lengthCs_val->setObjectName(QString::fromUtf8("lengthCs_val"));
        sizePolicy1.setHeightForWidth(lengthCs_val->sizePolicy().hasHeightForWidth());
        lengthCs_val->setSizePolicy(sizePolicy1);
        lengthCs_val->setMaximumSize(QSize(100, 20));

        geometricalSettingsVals->addWidget(lengthCs_val);

        heightCs_val = new QLineEdit(layoutWidget1);
        heightCs_val->setObjectName(QString::fromUtf8("heightCs_val"));
        sizePolicy1.setHeightForWidth(heightCs_val->sizePolicy().hasHeightForWidth());
        heightCs_val->setSizePolicy(sizePolicy1);
        heightCs_val->setMaximumSize(QSize(100, 20));

        geometricalSettingsVals->addWidget(heightCs_val);

        lengthD_val = new QLineEdit(layoutWidget1);
        lengthD_val->setObjectName(QString::fromUtf8("lengthD_val"));
        sizePolicy1.setHeightForWidth(lengthD_val->sizePolicy().hasHeightForWidth());
        lengthD_val->setSizePolicy(sizePolicy1);
        lengthD_val->setMaximumSize(QSize(100, 20));

        geometricalSettingsVals->addWidget(lengthD_val);

        heightD_val = new QLineEdit(layoutWidget1);
        heightD_val->setObjectName(QString::fromUtf8("heightD_val"));
        sizePolicy1.setHeightForWidth(heightD_val->sizePolicy().hasHeightForWidth());
        heightD_val->setSizePolicy(sizePolicy1);
        heightD_val->setMaximumSize(QSize(100, 20));

        geometricalSettingsVals->addWidget(heightD_val);

        hshiftTubeS_val = new QLineEdit(layoutWidget1);
        hshiftTubeS_val->setObjectName(QString::fromUtf8("hshiftTubeS_val"));
        sizePolicy1.setHeightForWidth(hshiftTubeS_val->sizePolicy().hasHeightForWidth());
        hshiftTubeS_val->setSizePolicy(sizePolicy1);
        hshiftTubeS_val->setMaximumSize(QSize(100, 20));

        geometricalSettingsVals->addWidget(hshiftTubeS_val);

        hshiftTubeAP_val = new QLineEdit(layoutWidget1);
        hshiftTubeAP_val->setObjectName(QString::fromUtf8("hshiftTubeAP_val"));
        sizePolicy1.setHeightForWidth(hshiftTubeAP_val->sizePolicy().hasHeightForWidth());
        hshiftTubeAP_val->setSizePolicy(sizePolicy1);
        hshiftTubeAP_val->setMaximumSize(QSize(100, 20));

        geometricalSettingsVals->addWidget(hshiftTubeAP_val);

        vshiftTubeS_val = new QLineEdit(layoutWidget1);
        vshiftTubeS_val->setObjectName(QString::fromUtf8("vshiftTubeS_val"));
        sizePolicy1.setHeightForWidth(vshiftTubeS_val->sizePolicy().hasHeightForWidth());
        vshiftTubeS_val->setSizePolicy(sizePolicy1);
        vshiftTubeS_val->setMaximumSize(QSize(100, 20));

        geometricalSettingsVals->addWidget(vshiftTubeS_val);

        vshiftTubeAP_val = new QLineEdit(layoutWidget1);
        vshiftTubeAP_val->setObjectName(QString::fromUtf8("vshiftTubeAP_val"));
        sizePolicy1.setHeightForWidth(vshiftTubeAP_val->sizePolicy().hasHeightForWidth());
        vshiftTubeAP_val->setSizePolicy(sizePolicy1);
        vshiftTubeAP_val->setMaximumSize(QSize(100, 20));

        geometricalSettingsVals->addWidget(vshiftTubeAP_val);

        angleC1_val = new QLineEdit(layoutWidget1);
        angleC1_val->setObjectName(QString::fromUtf8("angleC1_val"));
        sizePolicy1.setHeightForWidth(angleC1_val->sizePolicy().hasHeightForWidth());
        angleC1_val->setSizePolicy(sizePolicy1);
        angleC1_val->setMaximumSize(QSize(100, 20));

        geometricalSettingsVals->addWidget(angleC1_val);

        angleOffC1_val = new QLineEdit(layoutWidget1);
        angleOffC1_val->setObjectName(QString::fromUtf8("angleOffC1_val"));
        sizePolicy1.setHeightForWidth(angleOffC1_val->sizePolicy().hasHeightForWidth());
        angleOffC1_val->setSizePolicy(sizePolicy1);
        angleOffC1_val->setMaximumSize(QSize(100, 20));

        geometricalSettingsVals->addWidget(angleOffC1_val);

        angleTable_val = new QLineEdit(layoutWidget1);
        angleTable_val->setObjectName(QString::fromUtf8("angleTable_val"));
        sizePolicy1.setHeightForWidth(angleTable_val->sizePolicy().hasHeightForWidth());
        angleTable_val->setSizePolicy(sizePolicy1);
        angleTable_val->setMaximumSize(QSize(100, 20));

        geometricalSettingsVals->addWidget(angleTable_val);

        angleDP_val = new QLineEdit(layoutWidget1);
        angleDP_val->setObjectName(QString::fromUtf8("angleDP_val"));
        sizePolicy1.setHeightForWidth(angleDP_val->sizePolicy().hasHeightForWidth());
        angleDP_val->setSizePolicy(sizePolicy1);
        angleDP_val->setMaximumSize(QSize(100, 20));

        geometricalSettingsVals->addWidget(angleDP_val);

        angleDAP_val = new QLineEdit(layoutWidget1);
        angleDAP_val->setObjectName(QString::fromUtf8("angleDAP_val"));
        sizePolicy1.setHeightForWidth(angleDAP_val->sizePolicy().hasHeightForWidth());
        angleDAP_val->setSizePolicy(sizePolicy1);
        angleDAP_val->setMaximumSize(QSize(100, 20));

        geometricalSettingsVals->addWidget(angleDAP_val);

        angleC1tilt_val = new QLineEdit(layoutWidget1);
        angleC1tilt_val->setObjectName(QString::fromUtf8("angleC1tilt_val"));
        sizePolicy1.setHeightForWidth(angleC1tilt_val->sizePolicy().hasHeightForWidth());
        angleC1tilt_val->setSizePolicy(sizePolicy1);
        angleC1tilt_val->setMaximumSize(QSize(100, 20));

        geometricalSettingsVals->addWidget(angleC1tilt_val);

        angleC2tilt_val = new QLineEdit(layoutWidget1);
        angleC2tilt_val->setObjectName(QString::fromUtf8("angleC2tilt_val"));
        sizePolicy1.setHeightForWidth(angleC2tilt_val->sizePolicy().hasHeightForWidth());
        angleC2tilt_val->setSizePolicy(sizePolicy1);
        angleC2tilt_val->setMaximumSize(QSize(100, 20));

        geometricalSettingsVals->addWidget(angleC2tilt_val);

        geometricalSettingsVals->setStretch(0, 1);
        geometricalSettingsVals->setStretch(1, 1);
        geometricalSettingsVals->setStretch(2, 1);
        geometricalSettingsVals->setStretch(3, 1);
        geometricalSettingsVals->setStretch(4, 1);
        geometricalSettingsVals->setStretch(5, 1);
        geometricalSettingsVals->setStretch(6, 1);
        geometricalSettingsVals->setStretch(7, 1);
        geometricalSettingsVals->setStretch(8, 1);
        geometricalSettingsVals->setStretch(9, 1);
        geometricalSettingsVals->setStretch(10, 1);
        geometricalSettingsVals->setStretch(11, 1);
        geometricalSettingsVals->setStretch(12, 1);
        geometricalSettingsVals->setStretch(13, 1);
        geometricalSettingsVals->setStretch(14, 1);
        geometricalSettingsVals->setStretch(15, 1);
        geometricalSettingsVals->setStretch(16, 1);
        geometricalSettingsVals->setStretch(17, 1);
        geometricalSettingsVals->setStretch(18, 1);
        geometricalSettingsVals->setStretch(19, 1);
        geometricalSettings_splitter->addWidget(layoutWidget1);

        gridLayout->addWidget(geometricalSettings_splitter, 1, 2, 5, 1);

        physicalSettings_splitter = new QSplitter(centralwidget);
        physicalSettings_splitter->setObjectName(QString::fromUtf8("physicalSettings_splitter"));
        sizePolicy.setHeightForWidth(physicalSettings_splitter->sizePolicy().hasHeightForWidth());
        physicalSettings_splitter->setSizePolicy(sizePolicy);
        physicalSettings_splitter->setMinimumSize(QSize(351, 178));
        physicalSettings_splitter->setMaximumSize(QSize(16777215, 16777215));
        physicalSettings_splitter->setOrientation(Qt::Horizontal);
        physicalSettings_splitter->setOpaqueResize(false);
        physicalSettings_splitter->setChildrenCollapsible(false);
        layoutWidget2 = new QWidget(physicalSettings_splitter);
        layoutWidget2->setObjectName(QString::fromUtf8("layoutWidget2"));
        physicalSettingsLabels = new QVBoxLayout(layoutWidget2);
        physicalSettingsLabels->setSpacing(14);
        physicalSettingsLabels->setObjectName(QString::fromUtf8("physicalSettingsLabels"));
        physicalSettingsLabels->setSizeConstraint(QLayout::SetNoConstraint);
        physicalSettingsLabels->setContentsMargins(0, 0, 0, 0);
        energy_label = new QLabel(layoutWidget2);
        energy_label->setObjectName(QString::fromUtf8("energy_label"));

        physicalSettingsLabels->addWidget(energy_label);

        naturalWidth_label = new QLabel(layoutWidget2);
        naturalWidth_label->setObjectName(QString::fromUtf8("naturalWidth_label"));

        physicalSettingsLabels->addWidget(naturalWidth_label);

        tempC1P_label = new QLabel(layoutWidget2);
        tempC1P_label->setObjectName(QString::fromUtf8("tempC1P_label"));

        physicalSettingsLabels->addWidget(tempC1P_label);

        tempC1AP_label = new QLabel(layoutWidget2);
        tempC1AP_label->setObjectName(QString::fromUtf8("tempC1AP_label"));

        physicalSettingsLabels->addWidget(tempC1AP_label);

        tempC2P_label = new QLabel(layoutWidget2);
        tempC2P_label->setObjectName(QString::fromUtf8("tempC2P_label"));

        physicalSettingsLabels->addWidget(tempC2P_label);

        tempC2AP_label = new QLabel(layoutWidget2);
        tempC2AP_label->setObjectName(QString::fromUtf8("tempC2AP_label"));

        physicalSettingsLabels->addWidget(tempC2AP_label);

        dopplerFWHMGauss_label = new QLabel(layoutWidget2);
        dopplerFWHMGauss_label->setObjectName(QString::fromUtf8("dopplerFWHMGauss_label"));

        physicalSettingsLabels->addWidget(dopplerFWHMGauss_label);

        physicalSettingsLabels->setStretch(0, 1);
        physicalSettingsLabels->setStretch(1, 1);
        physicalSettingsLabels->setStretch(2, 1);
        physicalSettingsLabels->setStretch(3, 1);
        physicalSettingsLabels->setStretch(4, 1);
        physicalSettingsLabels->setStretch(5, 1);
        physicalSettingsLabels->setStretch(6, 1);
        physicalSettings_splitter->addWidget(layoutWidget2);
        layoutWidget3 = new QWidget(physicalSettings_splitter);
        layoutWidget3->setObjectName(QString::fromUtf8("layoutWidget3"));
        physicalSettingsVals = new QVBoxLayout(layoutWidget3);
        physicalSettingsVals->setObjectName(QString::fromUtf8("physicalSettingsVals"));
        physicalSettingsVals->setSizeConstraint(QLayout::SetNoConstraint);
        physicalSettingsVals->setContentsMargins(0, 0, 0, 0);
        energy_val = new QLineEdit(layoutWidget3);
        energy_val->setObjectName(QString::fromUtf8("energy_val"));
        sizePolicy1.setHeightForWidth(energy_val->sizePolicy().hasHeightForWidth());
        energy_val->setSizePolicy(sizePolicy1);
        energy_val->setMaximumSize(QSize(100, 20));

        physicalSettingsVals->addWidget(energy_val);

        naturalWidth_val = new QLineEdit(layoutWidget3);
        naturalWidth_val->setObjectName(QString::fromUtf8("naturalWidth_val"));
        sizePolicy1.setHeightForWidth(naturalWidth_val->sizePolicy().hasHeightForWidth());
        naturalWidth_val->setSizePolicy(sizePolicy1);
        naturalWidth_val->setMaximumSize(QSize(100, 20));

        physicalSettingsVals->addWidget(naturalWidth_val);

        tempC1P_val = new QLineEdit(layoutWidget3);
        tempC1P_val->setObjectName(QString::fromUtf8("tempC1P_val"));
        sizePolicy1.setHeightForWidth(tempC1P_val->sizePolicy().hasHeightForWidth());
        tempC1P_val->setSizePolicy(sizePolicy1);
        tempC1P_val->setMaximumSize(QSize(100, 20));

        physicalSettingsVals->addWidget(tempC1P_val);

        tempC1AP_val = new QLineEdit(layoutWidget3);
        tempC1AP_val->setObjectName(QString::fromUtf8("tempC1AP_val"));
        sizePolicy1.setHeightForWidth(tempC1AP_val->sizePolicy().hasHeightForWidth());
        tempC1AP_val->setSizePolicy(sizePolicy1);
        tempC1AP_val->setMaximumSize(QSize(100, 20));

        physicalSettingsVals->addWidget(tempC1AP_val);

        tempC2P_val = new QLineEdit(layoutWidget3);
        tempC2P_val->setObjectName(QString::fromUtf8("tempC2P_val"));
        sizePolicy1.setHeightForWidth(tempC2P_val->sizePolicy().hasHeightForWidth());
        tempC2P_val->setSizePolicy(sizePolicy1);
        tempC2P_val->setMaximumSize(QSize(100, 20));

        physicalSettingsVals->addWidget(tempC2P_val);

        tempC2AP_val = new QLineEdit(layoutWidget3);
        tempC2AP_val->setObjectName(QString::fromUtf8("tempC2AP_val"));
        sizePolicy1.setHeightForWidth(tempC2AP_val->sizePolicy().hasHeightForWidth());
        tempC2AP_val->setSizePolicy(sizePolicy1);
        tempC2AP_val->setMaximumSize(QSize(100, 20));

        physicalSettingsVals->addWidget(tempC2AP_val);

        dopplerFWHMGauss_val = new QLineEdit(layoutWidget3);
        dopplerFWHMGauss_val->setObjectName(QString::fromUtf8("dopplerFWHMGauss_val"));
        sizePolicy1.setHeightForWidth(dopplerFWHMGauss_val->sizePolicy().hasHeightForWidth());
        dopplerFWHMGauss_val->setSizePolicy(sizePolicy1);
        dopplerFWHMGauss_val->setMaximumSize(QSize(100, 20));

        physicalSettingsVals->addWidget(dopplerFWHMGauss_val);

        physicalSettingsVals->setStretch(0, 1);
        physicalSettingsVals->setStretch(1, 1);
        physicalSettingsVals->setStretch(2, 1);
        physicalSettingsVals->setStretch(3, 1);
        physicalSettingsVals->setStretch(4, 1);
        physicalSettingsVals->setStretch(5, 1);
        physicalSettingsVals->setStretch(6, 1);
        physicalSettings_splitter->addWidget(layoutWidget3);

        gridLayout->addWidget(physicalSettings_splitter, 1, 1, 1, 1);

        plotParameters_label = new QLabel(centralwidget);
        plotParameters_label->setObjectName(QString::fromUtf8("plotParameters_label"));

        gridLayout->addWidget(plotParameters_label, 2, 1, 1, 1);

        plotParameters_splitter = new QSplitter(centralwidget);
        plotParameters_splitter->setObjectName(QString::fromUtf8("plotParameters_splitter"));
        sizePolicy.setHeightForWidth(plotParameters_splitter->sizePolicy().hasHeightForWidth());
        plotParameters_splitter->setSizePolicy(sizePolicy);
        plotParameters_splitter->setMinimumSize(QSize(311, 100));
        plotParameters_splitter->setMaximumSize(QSize(16777215, 16777215));
        plotParameters_splitter->setOrientation(Qt::Horizontal);
        plotParameters_splitter->setOpaqueResize(false);
        plotParameters_splitter->setChildrenCollapsible(false);
        layoutWidget4 = new QWidget(plotParameters_splitter);
        layoutWidget4->setObjectName(QString::fromUtf8("layoutWidget4"));
        plotParametersLabels = new QVBoxLayout(layoutWidget4);
        plotParametersLabels->setSpacing(14);
        plotParametersLabels->setObjectName(QString::fromUtf8("plotParametersLabels"));
        plotParametersLabels->setSizeConstraint(QLayout::SetNoConstraint);
        plotParametersLabels->setContentsMargins(0, 0, 0, 0);
        angleScanRange_label = new QLabel(layoutWidget4);
        angleScanRange_label->setObjectName(QString::fromUtf8("angleScanRange_label"));

        plotParametersLabels->addWidget(angleScanRange_label);

        angleDispShift_label = new QLabel(layoutWidget4);
        angleDispShift_label->setObjectName(QString::fromUtf8("angleDispShift_label"));

        plotParametersLabels->addWidget(angleDispShift_label);

        binNum_label = new QLabel(layoutWidget4);
        binNum_label->setObjectName(QString::fromUtf8("binNum_label"));

        plotParametersLabels->addWidget(binNum_label);

        rayNum_label = new QLabel(layoutWidget4);
        rayNum_label->setObjectName(QString::fromUtf8("rayNum_label"));

        plotParametersLabels->addWidget(rayNum_label);

        plotParametersLabels->setStretch(0, 1);
        plotParametersLabels->setStretch(1, 1);
        plotParametersLabels->setStretch(2, 1);
        plotParametersLabels->setStretch(3, 1);
        plotParameters_splitter->addWidget(layoutWidget4);
        layoutWidget5 = new QWidget(plotParameters_splitter);
        layoutWidget5->setObjectName(QString::fromUtf8("layoutWidget5"));
        plotParametersVals = new QVBoxLayout(layoutWidget5);
        plotParametersVals->setObjectName(QString::fromUtf8("plotParametersVals"));
        plotParametersVals->setSizeConstraint(QLayout::SetNoConstraint);
        plotParametersVals->setContentsMargins(0, 0, 0, 0);
        angleScanRange_val = new QLineEdit(layoutWidget5);
        angleScanRange_val->setObjectName(QString::fromUtf8("angleScanRange_val"));
        sizePolicy1.setHeightForWidth(angleScanRange_val->sizePolicy().hasHeightForWidth());
        angleScanRange_val->setSizePolicy(sizePolicy1);
        angleScanRange_val->setMaximumSize(QSize(100, 20));

        plotParametersVals->addWidget(angleScanRange_val);

        angleDispShift_val = new QLineEdit(layoutWidget5);
        angleDispShift_val->setObjectName(QString::fromUtf8("angleDispShift_val"));
        sizePolicy1.setHeightForWidth(angleDispShift_val->sizePolicy().hasHeightForWidth());
        angleDispShift_val->setSizePolicy(sizePolicy1);
        angleDispShift_val->setMaximumSize(QSize(100, 20));

        plotParametersVals->addWidget(angleDispShift_val);

        binNum_val = new QLineEdit(layoutWidget5);
        binNum_val->setObjectName(QString::fromUtf8("binNum_val"));
        sizePolicy1.setHeightForWidth(binNum_val->sizePolicy().hasHeightForWidth());
        binNum_val->setSizePolicy(sizePolicy1);
        binNum_val->setMaximumSize(QSize(100, 20));

        plotParametersVals->addWidget(binNum_val);

        rayNum_val = new QLineEdit(layoutWidget5);
        rayNum_val->setObjectName(QString::fromUtf8("rayNum_val"));
        sizePolicy1.setHeightForWidth(rayNum_val->sizePolicy().hasHeightForWidth());
        rayNum_val->setSizePolicy(sizePolicy1);
        rayNum_val->setMaximumSize(QSize(100, 20));

        plotParametersVals->addWidget(rayNum_val);

        plotParametersVals->setStretch(0, 1);
        plotParametersVals->setStretch(1, 1);
        plotParametersVals->setStretch(2, 1);
        plotParametersVals->setStretch(3, 1);
        plotParameters_splitter->addWidget(layoutWidget5);

        gridLayout->addWidget(plotParameters_splitter, 3, 1, 1, 1);

        geometricalSettings_label = new QLabel(centralwidget);
        geometricalSettings_label->setObjectName(QString::fromUtf8("geometricalSettings_label"));

        gridLayout->addWidget(geometricalSettings_label, 0, 2, 1, 1);

        userSettings_label = new QLabel(centralwidget);
        userSettings_label->setObjectName(QString::fromUtf8("userSettings_label"));

        gridLayout->addWidget(userSettings_label, 0, 0, 1, 1);

        userSettings_splitter = new QSplitter(centralwidget);
        userSettings_splitter->setObjectName(QString::fromUtf8("userSettings_splitter"));
        sizePolicy.setHeightForWidth(userSettings_splitter->sizePolicy().hasHeightForWidth());
        userSettings_splitter->setSizePolicy(sizePolicy);
        userSettings_splitter->setMinimumSize(QSize(202, 341));
        userSettings_splitter->setMaximumSize(QSize(16777215, 16777215));
        userSettings_splitter->setOrientation(Qt::Vertical);
        userSettings_splitter->setOpaqueResize(false);
        userSettings_splitter->setHandleWidth(40);
        userSettings_splitter->setChildrenCollapsible(false);
        crystalSettings_splitter = new QSplitter(userSettings_splitter);
        crystalSettings_splitter->setObjectName(QString::fromUtf8("crystalSettings_splitter"));
        sizePolicy.setHeightForWidth(crystalSettings_splitter->sizePolicy().hasHeightForWidth());
        crystalSettings_splitter->setSizePolicy(sizePolicy);
        crystalSettings_splitter->setMinimumSize(QSize(202, 259));
        crystalSettings_splitter->setMaximumSize(QSize(16777215, 16777215));
        crystalSettings_splitter->setOrientation(Qt::Vertical);
        crystalSettings_splitter->setOpaqueResize(false);
        crystalSettings_splitter->setHandleWidth(15);
        crystalSettings_splitter->setChildrenCollapsible(false);
        C1Settings_splitter = new QSplitter(crystalSettings_splitter);
        C1Settings_splitter->setObjectName(QString::fromUtf8("C1Settings_splitter"));
        sizePolicy.setHeightForWidth(C1Settings_splitter->sizePolicy().hasHeightForWidth());
        C1Settings_splitter->setSizePolicy(sizePolicy);
        C1Settings_splitter->setMinimumSize(QSize(202, 158));
        C1Settings_splitter->setMaximumSize(QSize(16777215, 16777215));
        C1Settings_splitter->setOrientation(Qt::Vertical);
        C1Settings_splitter->setOpaqueResize(false);
        C1Settings_splitter->setHandleWidth(23);
        C1Settings_splitter->setChildrenCollapsible(false);
        layoutWidget6 = new QWidget(C1Settings_splitter);
        layoutWidget6->setObjectName(QString::fromUtf8("layoutWidget6"));
        spectraSimTypeLayout = new QVBoxLayout(layoutWidget6);
        spectraSimTypeLayout->setObjectName(QString::fromUtf8("spectraSimTypeLayout"));
        spectraSimTypeLayout->setSizeConstraint(QLayout::SetNoConstraint);
        spectraSimTypeLayout->setContentsMargins(0, 0, 0, 0);
        makeP_check = new QCheckBox(layoutWidget6);
        makeP_check->setObjectName(QString::fromUtf8("makeP_check"));

        spectraSimTypeLayout->addWidget(makeP_check);

        makeAP_check = new QCheckBox(layoutWidget6);
        makeAP_check->setObjectName(QString::fromUtf8("makeAP_check"));

        spectraSimTypeLayout->addWidget(makeAP_check);

        spectraSimTypeLayout->setStretch(0, 1);
        spectraSimTypeLayout->setStretch(1, 1);
        C1Settings_splitter->addWidget(layoutWidget6);
        layoutWidget7 = new QWidget(C1Settings_splitter);
        layoutWidget7->setObjectName(QString::fromUtf8("layoutWidget7"));
        C1MaskSettingsLayout = new QVBoxLayout(layoutWidget7);
        C1MaskSettingsLayout->setObjectName(QString::fromUtf8("C1MaskSettingsLayout"));
        C1MaskSettingsLayout->setSizeConstraint(QLayout::SetNoConstraint);
        C1MaskSettingsLayout->setContentsMargins(0, 0, 0, 0);
        C1Mask_label = new QLabel(layoutWidget7);
        C1Mask_label->setObjectName(QString::fromUtf8("C1Mask_label"));

        C1MaskSettingsLayout->addWidget(C1Mask_label);

        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        C1_noMask_radio = new QRadioButton(layoutWidget7);
        C1_noMask_radio->setObjectName(QString::fromUtf8("C1_noMask_radio"));

        verticalLayout->addWidget(C1_noMask_radio);

        C1_downMask_radio = new QRadioButton(layoutWidget7);
        C1_downMask_radio->setObjectName(QString::fromUtf8("C1_downMask_radio"));

        verticalLayout->addWidget(C1_downMask_radio);

        C1_upMask_radio = new QRadioButton(layoutWidget7);
        C1_upMask_radio->setObjectName(QString::fromUtf8("C1_upMask_radio"));

        verticalLayout->addWidget(C1_upMask_radio);


        C1MaskSettingsLayout->addLayout(verticalLayout);

        C1MaskSettingsLayout->setStretch(0, 1);
        C1MaskSettingsLayout->setStretch(1, 1);
        C1Settings_splitter->addWidget(layoutWidget7);
        crystalSettings_splitter->addWidget(C1Settings_splitter);
        layoutWidget8 = new QWidget(crystalSettings_splitter);
        layoutWidget8->setObjectName(QString::fromUtf8("layoutWidget8"));
        C2MaskSettingsLayout = new QVBoxLayout(layoutWidget8);
        C2MaskSettingsLayout->setObjectName(QString::fromUtf8("C2MaskSettingsLayout"));
        C2MaskSettingsLayout->setSizeConstraint(QLayout::SetNoConstraint);
        C2MaskSettingsLayout->setContentsMargins(0, 0, 0, 0);
        C2Mask_label = new QLabel(layoutWidget8);
        C2Mask_label->setObjectName(QString::fromUtf8("C2Mask_label"));

        C2MaskSettingsLayout->addWidget(C2Mask_label);

        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        verticalLayout_2->setSizeConstraint(QLayout::SetNoConstraint);
        C2_noMask_radio = new QRadioButton(layoutWidget8);
        C2_noMask_radio->setObjectName(QString::fromUtf8("C2_noMask_radio"));

        verticalLayout_2->addWidget(C2_noMask_radio);

        C2_downMask_radio = new QRadioButton(layoutWidget8);
        C2_downMask_radio->setObjectName(QString::fromUtf8("C2_downMask_radio"));

        verticalLayout_2->addWidget(C2_downMask_radio);

        C2_upMask_radio = new QRadioButton(layoutWidget8);
        C2_upMask_radio->setObjectName(QString::fromUtf8("C2_upMask_radio"));

        verticalLayout_2->addWidget(C2_upMask_radio);

        verticalLayout_2->setStretch(0, 1);
        verticalLayout_2->setStretch(1, 1);
        verticalLayout_2->setStretch(2, 1);

        C2MaskSettingsLayout->addLayout(verticalLayout_2);

        C2MaskSettingsLayout->setStretch(0, 1);
        C2MaskSettingsLayout->setStretch(1, 1);
        crystalSettings_splitter->addWidget(layoutWidget8);
        userSettings_splitter->addWidget(crystalSettings_splitter);
        layoutWidget9 = new QWidget(userSettings_splitter);
        layoutWidget9->setObjectName(QString::fromUtf8("layoutWidget9"));
        makeImagePlatesLayout = new QVBoxLayout(layoutWidget9);
        makeImagePlatesLayout->setObjectName(QString::fromUtf8("makeImagePlatesLayout"));
        makeImagePlatesLayout->setSizeConstraint(QLayout::SetNoConstraint);
        makeImagePlatesLayout->setContentsMargins(0, 0, 0, 0);
        makeImagePlateC1_check = new QCheckBox(layoutWidget9);
        makeImagePlateC1_check->setObjectName(QString::fromUtf8("makeImagePlateC1_check"));

        makeImagePlatesLayout->addWidget(makeImagePlateC1_check);

        makeImagePlateC2_check = new QCheckBox(layoutWidget9);
        makeImagePlateC2_check->setObjectName(QString::fromUtf8("makeImagePlateC2_check"));

        makeImagePlatesLayout->addWidget(makeImagePlateC2_check);

        makeImagePlatesLayout->setStretch(0, 1);
        makeImagePlatesLayout->setStretch(1, 1);
        userSettings_splitter->addWidget(layoutWidget9);

        gridLayout->addWidget(userSettings_splitter, 1, 0, 3, 1);

        gridLayout->setRowStretch(0, 1);
        gridLayout->setRowStretch(1, 1);
        gridLayout->setRowStretch(2, 1);
        gridLayout->setRowStretch(3, 1);
        gridLayout->setRowStretch(4, 1);
        gridLayout->setRowStretch(5, 1);
        gridLayout->setColumnStretch(0, 1);
        gridLayout->setColumnStretch(1, 1);
        gridLayout->setColumnStretch(2, 1);
        DCS_GUI->setCentralWidget(centralwidget);
        statusbar = new QStatusBar(DCS_GUI);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        statusbar->setAcceptDrops(false);
        DCS_GUI->setStatusBar(statusbar);

        retranslateUi(DCS_GUI);

        QMetaObject::connectSlotsByName(DCS_GUI);
    } // setupUi

    void retranslateUi(QMainWindow *DCS_GUI)
    {
        DCS_GUI->setWindowTitle(QCoreApplication::translate("DCS_GUI", "DCS Simulation", nullptr));
        physicalSettings_label->setText(QCoreApplication::translate("DCS_GUI", "Physical settings", nullptr));
        fullSettings_path->setText(QCoreApplication::translate("DCS_GUI", "More Settings can be changed at: ", nullptr));
        startSim_button->setText(QCoreApplication::translate("DCS_GUI", "Start Simulation", nullptr));
        distSAP_label->setText(QCoreApplication::translate("DCS_GUI", "Distance from source to aperture (cm)", nullptr));
        distAPC1_label->setText(QCoreApplication::translate("DCS_GUI", "Distance from aperture to crystal 1 (cm)", nullptr));
        distC1C2_label->setText(QCoreApplication::translate("DCS_GUI", "Distance between crystals (cm)", nullptr));
        distC2D_label->setText(QCoreApplication::translate("DCS_GUI", "Distance from crystal 2 to detector (cm)", nullptr));
        cuTubeDiam_label->setText(QCoreApplication::translate("DCS_GUI", "Diameter of copper tube (cm)", nullptr));
        lengthCs_label->setText(QCoreApplication::translate("DCS_GUI", "Length of crystals (cm)", nullptr));
        heightCs_label->setText(QCoreApplication::translate("DCS_GUI", "Height of crystals (cm)", nullptr));
        lengthD_label->setText(QCoreApplication::translate("DCS_GUI", "Length of detector (cm)", nullptr));
        heightD_label->setText(QCoreApplication::translate("DCS_GUI", "Height of detector (cm)", nullptr));
        hshiftTubeS_label->setText(QCoreApplication::translate("DCS_GUI", "Horizontal shift in tube at source (cm)", nullptr));
        hshiftTubeAP_label->setText(QCoreApplication::translate("DCS_GUI", "Horizontal shift in tube at aperture (cm)", nullptr));
        vshiftTubeS_label->setText(QCoreApplication::translate("DCS_GUI", "Vertical shift in tube at source (cm)", nullptr));
        vshiftTubeAP_label->setText(QCoreApplication::translate("DCS_GUI", "Vertical shift in tube at aperture (cm)", nullptr));
        angleC1_label->setText(QCoreApplication::translate("DCS_GUI", "Angle of crystal 1 (deg.)", nullptr));
        angleOffC1_label->setText(QCoreApplication::translate("DCS_GUI", "Offset of crystal 1 (deg.)", nullptr));
        angleTable_label->setText(QCoreApplication::translate("DCS_GUI", "Angle of table (deg.)", nullptr));
        angleDP_label->setText(QCoreApplication::translate("DCS_GUI", "Angle of detector in parallel (deg.)", nullptr));
        angleDAP_label->setText(QCoreApplication::translate("DCS_GUI", "Angle of detector in antiparallel (deg.)", nullptr));
        angleC1tilt_label->setText(QCoreApplication::translate("DCS_GUI", "Tilt angle in crystal 1 (deg.)", nullptr));
        angleC2tilt_label->setText(QCoreApplication::translate("DCS_GUI", "Tilt angle in crystal 2 (deg.)", nullptr));
        distSAP_val->setProperty("plainText", QVariant(QString()));
        distAPC1_val->setProperty("plainText", QVariant(QString()));
        distC1C2_val->setProperty("plainText", QVariant(QString()));
        distC2D_val->setProperty("plainText", QVariant(QString()));
        cuTubeDiam_val->setProperty("plainText", QVariant(QString()));
        lengthCs_val->setProperty("plainText", QVariant(QString()));
        heightCs_val->setProperty("plainText", QVariant(QString()));
        lengthD_val->setProperty("plainText", QVariant(QString()));
        heightD_val->setProperty("plainText", QVariant(QString()));
        hshiftTubeS_val->setProperty("plainText", QVariant(QString()));
        hshiftTubeAP_val->setProperty("plainText", QVariant(QString()));
        vshiftTubeS_val->setProperty("plainText", QVariant(QString()));
        vshiftTubeAP_val->setProperty("plainText", QVariant(QString()));
        angleC1_val->setProperty("plainText", QVariant(QString()));
        angleOffC1_val->setProperty("plainText", QVariant(QString()));
        angleTable_val->setProperty("plainText", QVariant(QString()));
        angleDP_val->setProperty("plainText", QVariant(QString()));
        angleDAP_val->setProperty("plainText", QVariant(QString()));
        angleC1tilt_val->setProperty("plainText", QVariant(QString()));
        angleC2tilt_val->setProperty("plainText", QVariant(QString()));
#if QT_CONFIG(statustip)
        energy_label->setStatusTip(QCoreApplication::translate("DCS_GUI", "Source energy", nullptr));
#endif // QT_CONFIG(statustip)
        energy_label->setText(QCoreApplication::translate("DCS_GUI", "Energy (eV)", nullptr));
#if QT_CONFIG(statustip)
        naturalWidth_label->setStatusTip(QCoreApplication::translate("DCS_GUI", "Source natural width", nullptr));
#endif // QT_CONFIG(statustip)
        naturalWidth_label->setText(QCoreApplication::translate("DCS_GUI", "Natural width FWHM (eV)", nullptr));
#if QT_CONFIG(statustip)
        tempC1P_label->setStatusTip(QCoreApplication::translate("DCS_GUI", "Temperature of the first crystal to be considered while simulating the parallel scan", nullptr));
#endif // QT_CONFIG(statustip)
        tempC1P_label->setText(QCoreApplication::translate("DCS_GUI", "Temperature on crystal 1 in parallel scan (\302\272C)", nullptr));
#if QT_CONFIG(statustip)
        tempC1AP_label->setStatusTip(QCoreApplication::translate("DCS_GUI", "Temperature of the first crystal to be considered while simulating the antiparallel scan", nullptr));
#endif // QT_CONFIG(statustip)
        tempC1AP_label->setText(QCoreApplication::translate("DCS_GUI", "Temperature on crystal 1 in antiparallel scan (\302\272C)", nullptr));
#if QT_CONFIG(statustip)
        tempC2P_label->setStatusTip(QCoreApplication::translate("DCS_GUI", "Temperature of the second crystal to be considered while simulating the parallel scan", nullptr));
#endif // QT_CONFIG(statustip)
        tempC2P_label->setText(QCoreApplication::translate("DCS_GUI", "Temperature on crystal 2 in parallel scan (\302\272C)", nullptr));
#if QT_CONFIG(statustip)
        tempC2AP_label->setStatusTip(QCoreApplication::translate("DCS_GUI", "Temperature of the second crystal to be considered while simulating the antiparallel scan", nullptr));
#endif // QT_CONFIG(statustip)
        tempC2AP_label->setText(QCoreApplication::translate("DCS_GUI", "Temperature on crystal 2 in antiparallel scan (\302\272C)", nullptr));
#if QT_CONFIG(statustip)
        dopplerFWHMGauss_label->setStatusTip(QCoreApplication::translate("DCS_GUI", "Gaussian FWHM of the doppler broadening effect considered in the simulation", nullptr));
#endif // QT_CONFIG(statustip)
        dopplerFWHMGauss_label->setText(QCoreApplication::translate("DCS_GUI", "Gaussian FWHM due to Doppler broadening (eV)", nullptr));
        naturalWidth_val->setProperty("plainText", QVariant(QString()));
        tempC1P_val->setProperty("plainText", QVariant(QString()));
        tempC1AP_val->setProperty("plainText", QVariant(QString()));
        tempC2P_val->setProperty("plainText", QVariant(QString()));
        tempC2AP_val->setProperty("plainText", QVariant(QString()));
        dopplerFWHMGauss_val->setProperty("plainText", QVariant(QString()));
        plotParameters_label->setText(QCoreApplication::translate("DCS_GUI", "Plot parameters", nullptr));
#if QT_CONFIG(statustip)
        angleScanRange_label->setStatusTip(QCoreApplication::translate("DCS_GUI", "Angle range of the plotted scan", nullptr));
#endif // QT_CONFIG(statustip)
        angleScanRange_label->setText(QCoreApplication::translate("DCS_GUI", "Angle scan range (deg.)", nullptr));
#if QT_CONFIG(statustip)
        angleDispShift_label->setStatusTip(QCoreApplication::translate("DCS_GUI", "Angular shift of the plots", nullptr));
#endif // QT_CONFIG(statustip)
        angleDispShift_label->setText(QCoreApplication::translate("DCS_GUI", "Shift of display window (deg.)", nullptr));
#if QT_CONFIG(statustip)
        binNum_label->setStatusTip(QCoreApplication::translate("DCS_GUI", "Number of bins of the plots", nullptr));
#endif // QT_CONFIG(statustip)
        binNum_label->setText(QCoreApplication::translate("DCS_GUI", "Number of bins", nullptr));
#if QT_CONFIG(statustip)
        rayNum_label->setStatusTip(QCoreApplication::translate("DCS_GUI", "Number of simulated rays", nullptr));
#endif // QT_CONFIG(statustip)
        rayNum_label->setText(QCoreApplication::translate("DCS_GUI", "Number of rays in Monte Carlo simulation", nullptr));
        angleScanRange_val->setProperty("plainText", QVariant(QString()));
        angleDispShift_val->setProperty("plainText", QVariant(QString()));
        binNum_val->setProperty("plainText", QVariant(QString()));
        rayNum_val->setProperty("plainText", QVariant(QString()));
        geometricalSettings_label->setText(QCoreApplication::translate("DCS_GUI", "Geometrical settings", nullptr));
        userSettings_label->setText(QCoreApplication::translate("DCS_GUI", "User Settings", nullptr));
#if QT_CONFIG(statustip)
        makeP_check->setStatusTip(QCoreApplication::translate("DCS_GUI", "Perform simulation for the parallel configuration peak", "Perform simulation for the parallel configuration peak"));
#endif // QT_CONFIG(statustip)
        makeP_check->setText(QCoreApplication::translate("DCS_GUI", "Make Parallel peak", nullptr));
#if QT_CONFIG(statustip)
        makeAP_check->setStatusTip(QCoreApplication::translate("DCS_GUI", "Perform simulation for the antiparallel configuration peak", "Perform simulation for the antiparallel configuration peak"));
#endif // QT_CONFIG(statustip)
        makeAP_check->setText(QCoreApplication::translate("DCS_GUI", "Make Antiparallel peak", nullptr));
        C1Mask_label->setText(QCoreApplication::translate("DCS_GUI", "Crystal 1 Mask", nullptr));
#if QT_CONFIG(statustip)
        C1_noMask_radio->setStatusTip(QCoreApplication::translate("DCS_GUI", "No mask will be placed on the first crystal", "Mask testing on the first crystal will not be performed"));
#endif // QT_CONFIG(statustip)
        C1_noMask_radio->setText(QCoreApplication::translate("DCS_GUI", "no", nullptr));
#if QT_CONFIG(statustip)
        C1_downMask_radio->setStatusTip(QCoreApplication::translate("DCS_GUI", "Mask will be placed on the first crystal in the down position", "Mask testing on the first crystal will be performed for the down mask setting"));
#endif // QT_CONFIG(statustip)
        C1_downMask_radio->setText(QCoreApplication::translate("DCS_GUI", "down", nullptr));
#if QT_CONFIG(tooltip)
        C1_upMask_radio->setToolTip(QString());
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(statustip)
        C1_upMask_radio->setStatusTip(QCoreApplication::translate("DCS_GUI", "Mask will be placed on the first crystal in the up position", nullptr));
#endif // QT_CONFIG(statustip)
        C1_upMask_radio->setText(QCoreApplication::translate("DCS_GUI", "up", nullptr));
        C2Mask_label->setText(QCoreApplication::translate("DCS_GUI", "Crystal 2 Mask", nullptr));
#if QT_CONFIG(statustip)
        C2_noMask_radio->setStatusTip(QCoreApplication::translate("DCS_GUI", "No mask will be placed on the second crystal", "Mask testing on the first crystal will not be performed"));
#endif // QT_CONFIG(statustip)
        C2_noMask_radio->setText(QCoreApplication::translate("DCS_GUI", "no", nullptr));
#if QT_CONFIG(statustip)
        C2_downMask_radio->setStatusTip(QCoreApplication::translate("DCS_GUI", "Mask will be placed on the second crystal in the down position", "Mask testing on the first crystal will be performed for the down mask setting"));
#endif // QT_CONFIG(statustip)
        C2_downMask_radio->setText(QCoreApplication::translate("DCS_GUI", "down", nullptr));
#if QT_CONFIG(tooltip)
        C2_upMask_radio->setToolTip(QString());
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(statustip)
        C2_upMask_radio->setStatusTip(QCoreApplication::translate("DCS_GUI", "Mask will be placed on the second crystal in the up position", "Mask testing on the first crystal will be performed for the up mask setting"));
#endif // QT_CONFIG(statustip)
        C2_upMask_radio->setText(QCoreApplication::translate("DCS_GUI", "up", nullptr));
#if QT_CONFIG(statustip)
        makeImagePlateC1_check->setStatusTip(QCoreApplication::translate("DCS_GUI", "Display the image of hits on the first crystal after reflection", "Perform simulation for the parallel configuration peak"));
#endif // QT_CONFIG(statustip)
        makeImagePlateC1_check->setText(QCoreApplication::translate("DCS_GUI", "Make Image after reflection crystal 1", nullptr));
#if QT_CONFIG(statustip)
        makeImagePlateC2_check->setStatusTip(QCoreApplication::translate("DCS_GUI", "Display the image of hits on the second crystal after reflection", "Perform simulation for the antiparallel configuration peak"));
#endif // QT_CONFIG(statustip)
        makeImagePlateC2_check->setText(QCoreApplication::translate("DCS_GUI", "Make Image after reflection crystal 2", nullptr));
    } // retranslateUi

};

namespace Ui {
    class DCS_GUI: public Ui_DCS_GUI {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DCS_GUI_H
