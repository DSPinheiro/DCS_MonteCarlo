/********************************************************************************
** Form generated from reading UI file 'simulationmain.ui'
**
** Created by: Qt User Interface Compiler version 5.15.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_SIMULATIONMAIN_H
#define UI_SIMULATIONMAIN_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPlainTextEdit>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include "qcustomplot.h"
#include "qdcs3dvis.h"

QT_BEGIN_NAMESPACE

class Ui_SimulationMain
{
public:
    QWidget *centralwidget;
    QGridLayout *gridLayout;
    QVBoxLayout *mainLayout;
    QHBoxLayout *timesLayout;
    QLabel *simstart;
    QLabel *simremain;
    QVBoxLayout *simElementsLayout;
    QHBoxLayout *imageplates3DvisLayout;
    QVBoxLayout *imageplatesLayout;
    QHBoxLayout *E_C2A_C2PLayout;
    QHBoxLayout *EPlate;
    QLabel *E_ImagePlate;
    QHBoxLayout *EPlateScale;
    QLabel *E_cScale;
    QLabel *E_vScale;
    QHBoxLayout *C2APlate;
    QLabel *C2A_ImagePlate;
    QHBoxLayout *C2AplateScale;
    QLabel *C2A_cScale;
    QLabel *C2A_vScale;
    QHBoxLayout *C2PPlate;
    QLabel *C2P_ImagePlate;
    QHBoxLayout *C2PplateScale;
    QLabel *C2P_cScale;
    QLabel *C2P_vScale;
    QHBoxLayout *C1_DA_DPLayout;
    QHBoxLayout *C1Plate;
    QLabel *C1_ImagePlate;
    QHBoxLayout *C1plateScale;
    QLabel *C1_cScale;
    QLabel *C1_vScale;
    QHBoxLayout *DAPlate;
    QLabel *DA_ImagePlate;
    QHBoxLayout *DAplateScale;
    QLabel *DA_cScale;
    QLabel *DA_vScale;
    QHBoxLayout *DPPlate;
    QLabel *DP_ImagePlate;
    QHBoxLayout *DPplateScale;
    QLabel *DP_cScale;
    QLabel *DP_vScale;
    QDCS3Dvis *GL3Dvis;
    QHBoxLayout *stats_graphs_Layout;
    QHBoxLayout *statsLayout;
    QVBoxLayout *statesLabels;
    QVBoxLayout *statsLabels2;
    QLabel *Ecnts_label;
    QLabel *C1cnts_label;
    QLabel *C2Pcnts_label;
    QLabel *DPcnts_label;
    QLabel *C2Acnts_label;
    QLabel *DAcnts_label;
    QLabel *currRot_label;
    QVBoxLayout *statsValues;
    QVBoxLayout *statsValues2;
    QLabel *Ecnts_val;
    QLabel *C1cnts_val;
    QLabel *C2Pcnts_val;
    QLabel *DPcnts_val;
    QLabel *C2Acnts_val;
    QLabel *DAcnts_val;
    QLabel *currRot_val;
    QHBoxLayout *histogramsLayout;
    QCustomPlot *AP_histogram;
    QCustomPlot *P_histogram;
    QPlainTextEdit *logBox;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *SimulationMain)
    {
        if (SimulationMain->objectName().isEmpty())
            SimulationMain->setObjectName(QString::fromUtf8("SimulationMain"));
        SimulationMain->setWindowModality(Qt::ApplicationModal);
        SimulationMain->resize(1042, 644);
        QSizePolicy sizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
        sizePolicy.setHorizontalStretch(1);
        sizePolicy.setVerticalStretch(1);
        sizePolicy.setHeightForWidth(SimulationMain->sizePolicy().hasHeightForWidth());
        SimulationMain->setSizePolicy(sizePolicy);
        SimulationMain->setMinimumSize(QSize(1042, 604));
        SimulationMain->setMaximumSize(QSize(16777215, 16777215));
        SimulationMain->setSizeIncrement(QSize(50, 20));
        SimulationMain->setBaseSize(QSize(1042, 604));
        SimulationMain->setToolButtonStyle(Qt::ToolButtonIconOnly);
        centralwidget = new QWidget(SimulationMain);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        centralwidget->setEnabled(true);
        sizePolicy.setHeightForWidth(centralwidget->sizePolicy().hasHeightForWidth());
        centralwidget->setSizePolicy(sizePolicy);
        centralwidget->setMinimumSize(QSize(1042, 582));
        centralwidget->setSizeIncrement(QSize(50, 20));
        centralwidget->setBaseSize(QSize(1042, 582));
        gridLayout = new QGridLayout(centralwidget);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        mainLayout = new QVBoxLayout();
        mainLayout->setSpacing(30);
        mainLayout->setObjectName(QString::fromUtf8("mainLayout"));
        mainLayout->setSizeConstraint(QLayout::SetNoConstraint);
        timesLayout = new QHBoxLayout();
        timesLayout->setSpacing(100);
        timesLayout->setObjectName(QString::fromUtf8("timesLayout"));
        timesLayout->setSizeConstraint(QLayout::SetMinimumSize);
        timesLayout->setContentsMargins(-1, -1, 350, -1);
        simstart = new QLabel(centralwidget);
        simstart->setObjectName(QString::fromUtf8("simstart"));

        timesLayout->addWidget(simstart);

        simremain = new QLabel(centralwidget);
        simremain->setObjectName(QString::fromUtf8("simremain"));

        timesLayout->addWidget(simremain);

        timesLayout->setStretch(0, 1);
        timesLayout->setStretch(1, 1);

        mainLayout->addLayout(timesLayout);

        simElementsLayout = new QVBoxLayout();
        simElementsLayout->setSpacing(23);
        simElementsLayout->setObjectName(QString::fromUtf8("simElementsLayout"));
        simElementsLayout->setSizeConstraint(QLayout::SetNoConstraint);
        imageplates3DvisLayout = new QHBoxLayout();
        imageplates3DvisLayout->setObjectName(QString::fromUtf8("imageplates3DvisLayout"));
        imageplatesLayout = new QVBoxLayout();
        imageplatesLayout->setSpacing(50);
        imageplatesLayout->setObjectName(QString::fromUtf8("imageplatesLayout"));
        imageplatesLayout->setSizeConstraint(QLayout::SetNoConstraint);
        imageplatesLayout->setContentsMargins(20, -1, 20, -1);
        E_C2A_C2PLayout = new QHBoxLayout();
        E_C2A_C2PLayout->setSpacing(110);
        E_C2A_C2PLayout->setObjectName(QString::fromUtf8("E_C2A_C2PLayout"));
        E_C2A_C2PLayout->setSizeConstraint(QLayout::SetNoConstraint);
        EPlate = new QHBoxLayout();
        EPlate->setSpacing(14);
        EPlate->setObjectName(QString::fromUtf8("EPlate"));
        EPlate->setSizeConstraint(QLayout::SetFixedSize);
        E_ImagePlate = new QLabel(centralwidget);
        E_ImagePlate->setObjectName(QString::fromUtf8("E_ImagePlate"));
        QSizePolicy sizePolicy1(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(E_ImagePlate->sizePolicy().hasHeightForWidth());
        E_ImagePlate->setSizePolicy(sizePolicy1);
        E_ImagePlate->setMinimumSize(QSize(100, 100));
        E_ImagePlate->setMaximumSize(QSize(100, 100));
        E_ImagePlate->setSizeIncrement(QSize(1, 1));

        EPlate->addWidget(E_ImagePlate);

        EPlateScale = new QHBoxLayout();
        EPlateScale->setObjectName(QString::fromUtf8("EPlateScale"));
        E_cScale = new QLabel(centralwidget);
        E_cScale->setObjectName(QString::fromUtf8("E_cScale"));
        sizePolicy1.setHeightForWidth(E_cScale->sizePolicy().hasHeightForWidth());
        E_cScale->setSizePolicy(sizePolicy1);
        E_cScale->setMinimumSize(QSize(10, 100));
        E_cScale->setMaximumSize(QSize(10, 100));
        E_cScale->setBaseSize(QSize(10, 100));

        EPlateScale->addWidget(E_cScale);

        E_vScale = new QLabel(centralwidget);
        E_vScale->setObjectName(QString::fromUtf8("E_vScale"));
        sizePolicy1.setHeightForWidth(E_vScale->sizePolicy().hasHeightForWidth());
        E_vScale->setSizePolicy(sizePolicy1);
        E_vScale->setMinimumSize(QSize(30, 100));
        E_vScale->setMaximumSize(QSize(50, 100));
        E_vScale->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignTop);

        EPlateScale->addWidget(E_vScale);


        EPlate->addLayout(EPlateScale);


        E_C2A_C2PLayout->addLayout(EPlate);

        C2APlate = new QHBoxLayout();
        C2APlate->setSpacing(14);
        C2APlate->setObjectName(QString::fromUtf8("C2APlate"));
        C2APlate->setSizeConstraint(QLayout::SetFixedSize);
        C2A_ImagePlate = new QLabel(centralwidget);
        C2A_ImagePlate->setObjectName(QString::fromUtf8("C2A_ImagePlate"));
        sizePolicy1.setHeightForWidth(C2A_ImagePlate->sizePolicy().hasHeightForWidth());
        C2A_ImagePlate->setSizePolicy(sizePolicy1);
        C2A_ImagePlate->setMinimumSize(QSize(100, 100));
        C2A_ImagePlate->setMaximumSize(QSize(100, 100));
        C2A_ImagePlate->setSizeIncrement(QSize(1, 1));

        C2APlate->addWidget(C2A_ImagePlate);

        C2AplateScale = new QHBoxLayout();
        C2AplateScale->setObjectName(QString::fromUtf8("C2AplateScale"));
        C2A_cScale = new QLabel(centralwidget);
        C2A_cScale->setObjectName(QString::fromUtf8("C2A_cScale"));
        sizePolicy1.setHeightForWidth(C2A_cScale->sizePolicy().hasHeightForWidth());
        C2A_cScale->setSizePolicy(sizePolicy1);
        C2A_cScale->setMinimumSize(QSize(10, 100));
        C2A_cScale->setMaximumSize(QSize(10, 100));
        C2A_cScale->setBaseSize(QSize(10, 100));

        C2AplateScale->addWidget(C2A_cScale);

        C2A_vScale = new QLabel(centralwidget);
        C2A_vScale->setObjectName(QString::fromUtf8("C2A_vScale"));
        sizePolicy1.setHeightForWidth(C2A_vScale->sizePolicy().hasHeightForWidth());
        C2A_vScale->setSizePolicy(sizePolicy1);
        C2A_vScale->setMinimumSize(QSize(30, 100));
        C2A_vScale->setMaximumSize(QSize(50, 100));
        C2A_vScale->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignTop);

        C2AplateScale->addWidget(C2A_vScale);


        C2APlate->addLayout(C2AplateScale);


        E_C2A_C2PLayout->addLayout(C2APlate);

        C2PPlate = new QHBoxLayout();
        C2PPlate->setSpacing(14);
        C2PPlate->setObjectName(QString::fromUtf8("C2PPlate"));
        C2PPlate->setSizeConstraint(QLayout::SetFixedSize);
        C2P_ImagePlate = new QLabel(centralwidget);
        C2P_ImagePlate->setObjectName(QString::fromUtf8("C2P_ImagePlate"));
        sizePolicy1.setHeightForWidth(C2P_ImagePlate->sizePolicy().hasHeightForWidth());
        C2P_ImagePlate->setSizePolicy(sizePolicy1);
        C2P_ImagePlate->setMinimumSize(QSize(100, 100));
        C2P_ImagePlate->setMaximumSize(QSize(100, 100));
        C2P_ImagePlate->setSizeIncrement(QSize(1, 1));

        C2PPlate->addWidget(C2P_ImagePlate);

        C2PplateScale = new QHBoxLayout();
        C2PplateScale->setObjectName(QString::fromUtf8("C2PplateScale"));
        C2P_cScale = new QLabel(centralwidget);
        C2P_cScale->setObjectName(QString::fromUtf8("C2P_cScale"));
        sizePolicy1.setHeightForWidth(C2P_cScale->sizePolicy().hasHeightForWidth());
        C2P_cScale->setSizePolicy(sizePolicy1);
        C2P_cScale->setMinimumSize(QSize(10, 100));
        C2P_cScale->setMaximumSize(QSize(10, 100));
        C2P_cScale->setBaseSize(QSize(10, 100));

        C2PplateScale->addWidget(C2P_cScale);

        C2P_vScale = new QLabel(centralwidget);
        C2P_vScale->setObjectName(QString::fromUtf8("C2P_vScale"));
        sizePolicy1.setHeightForWidth(C2P_vScale->sizePolicy().hasHeightForWidth());
        C2P_vScale->setSizePolicy(sizePolicy1);
        C2P_vScale->setMinimumSize(QSize(30, 100));
        C2P_vScale->setMaximumSize(QSize(50, 100));
        C2P_vScale->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignTop);

        C2PplateScale->addWidget(C2P_vScale);


        C2PPlate->addLayout(C2PplateScale);


        E_C2A_C2PLayout->addLayout(C2PPlate);

        E_C2A_C2PLayout->setStretch(0, 1);
        E_C2A_C2PLayout->setStretch(1, 1);
        E_C2A_C2PLayout->setStretch(2, 1);

        imageplatesLayout->addLayout(E_C2A_C2PLayout);

        C1_DA_DPLayout = new QHBoxLayout();
        C1_DA_DPLayout->setSpacing(110);
        C1_DA_DPLayout->setObjectName(QString::fromUtf8("C1_DA_DPLayout"));
        C1_DA_DPLayout->setSizeConstraint(QLayout::SetNoConstraint);
        C1Plate = new QHBoxLayout();
        C1Plate->setSpacing(14);
        C1Plate->setObjectName(QString::fromUtf8("C1Plate"));
        C1Plate->setSizeConstraint(QLayout::SetFixedSize);
        C1_ImagePlate = new QLabel(centralwidget);
        C1_ImagePlate->setObjectName(QString::fromUtf8("C1_ImagePlate"));
        sizePolicy1.setHeightForWidth(C1_ImagePlate->sizePolicy().hasHeightForWidth());
        C1_ImagePlate->setSizePolicy(sizePolicy1);
        C1_ImagePlate->setMinimumSize(QSize(100, 100));
        C1_ImagePlate->setMaximumSize(QSize(100, 100));
        C1_ImagePlate->setSizeIncrement(QSize(1, 1));

        C1Plate->addWidget(C1_ImagePlate);

        C1plateScale = new QHBoxLayout();
        C1plateScale->setObjectName(QString::fromUtf8("C1plateScale"));
        C1_cScale = new QLabel(centralwidget);
        C1_cScale->setObjectName(QString::fromUtf8("C1_cScale"));
        sizePolicy1.setHeightForWidth(C1_cScale->sizePolicy().hasHeightForWidth());
        C1_cScale->setSizePolicy(sizePolicy1);
        C1_cScale->setMinimumSize(QSize(10, 100));
        C1_cScale->setMaximumSize(QSize(10, 100));
        C1_cScale->setBaseSize(QSize(10, 100));

        C1plateScale->addWidget(C1_cScale);

        C1_vScale = new QLabel(centralwidget);
        C1_vScale->setObjectName(QString::fromUtf8("C1_vScale"));
        sizePolicy1.setHeightForWidth(C1_vScale->sizePolicy().hasHeightForWidth());
        C1_vScale->setSizePolicy(sizePolicy1);
        C1_vScale->setMinimumSize(QSize(30, 100));
        C1_vScale->setMaximumSize(QSize(50, 100));
        C1_vScale->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignTop);

        C1plateScale->addWidget(C1_vScale);


        C1Plate->addLayout(C1plateScale);


        C1_DA_DPLayout->addLayout(C1Plate);

        DAPlate = new QHBoxLayout();
        DAPlate->setSpacing(14);
        DAPlate->setObjectName(QString::fromUtf8("DAPlate"));
        DAPlate->setSizeConstraint(QLayout::SetFixedSize);
        DA_ImagePlate = new QLabel(centralwidget);
        DA_ImagePlate->setObjectName(QString::fromUtf8("DA_ImagePlate"));
        sizePolicy1.setHeightForWidth(DA_ImagePlate->sizePolicy().hasHeightForWidth());
        DA_ImagePlate->setSizePolicy(sizePolicy1);
        DA_ImagePlate->setMinimumSize(QSize(100, 100));
        DA_ImagePlate->setMaximumSize(QSize(100, 100));
        DA_ImagePlate->setSizeIncrement(QSize(1, 1));

        DAPlate->addWidget(DA_ImagePlate);

        DAplateScale = new QHBoxLayout();
        DAplateScale->setObjectName(QString::fromUtf8("DAplateScale"));
        DA_cScale = new QLabel(centralwidget);
        DA_cScale->setObjectName(QString::fromUtf8("DA_cScale"));
        sizePolicy1.setHeightForWidth(DA_cScale->sizePolicy().hasHeightForWidth());
        DA_cScale->setSizePolicy(sizePolicy1);
        DA_cScale->setMinimumSize(QSize(10, 100));
        DA_cScale->setMaximumSize(QSize(10, 100));
        DA_cScale->setBaseSize(QSize(10, 100));

        DAplateScale->addWidget(DA_cScale);

        DA_vScale = new QLabel(centralwidget);
        DA_vScale->setObjectName(QString::fromUtf8("DA_vScale"));
        sizePolicy1.setHeightForWidth(DA_vScale->sizePolicy().hasHeightForWidth());
        DA_vScale->setSizePolicy(sizePolicy1);
        DA_vScale->setMinimumSize(QSize(30, 100));
        DA_vScale->setMaximumSize(QSize(50, 100));
        DA_vScale->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignTop);

        DAplateScale->addWidget(DA_vScale);


        DAPlate->addLayout(DAplateScale);


        C1_DA_DPLayout->addLayout(DAPlate);

        DPPlate = new QHBoxLayout();
        DPPlate->setSpacing(14);
        DPPlate->setObjectName(QString::fromUtf8("DPPlate"));
        DPPlate->setSizeConstraint(QLayout::SetFixedSize);
        DP_ImagePlate = new QLabel(centralwidget);
        DP_ImagePlate->setObjectName(QString::fromUtf8("DP_ImagePlate"));
        sizePolicy1.setHeightForWidth(DP_ImagePlate->sizePolicy().hasHeightForWidth());
        DP_ImagePlate->setSizePolicy(sizePolicy1);
        DP_ImagePlate->setMinimumSize(QSize(100, 100));
        DP_ImagePlate->setMaximumSize(QSize(100, 100));
        DP_ImagePlate->setSizeIncrement(QSize(1, 1));

        DPPlate->addWidget(DP_ImagePlate);

        DPplateScale = new QHBoxLayout();
        DPplateScale->setObjectName(QString::fromUtf8("DPplateScale"));
        DP_cScale = new QLabel(centralwidget);
        DP_cScale->setObjectName(QString::fromUtf8("DP_cScale"));
        sizePolicy1.setHeightForWidth(DP_cScale->sizePolicy().hasHeightForWidth());
        DP_cScale->setSizePolicy(sizePolicy1);
        DP_cScale->setMinimumSize(QSize(10, 100));
        DP_cScale->setMaximumSize(QSize(10, 100));
        DP_cScale->setBaseSize(QSize(10, 100));

        DPplateScale->addWidget(DP_cScale);

        DP_vScale = new QLabel(centralwidget);
        DP_vScale->setObjectName(QString::fromUtf8("DP_vScale"));
        sizePolicy1.setHeightForWidth(DP_vScale->sizePolicy().hasHeightForWidth());
        DP_vScale->setSizePolicy(sizePolicy1);
        DP_vScale->setMinimumSize(QSize(30, 100));
        DP_vScale->setMaximumSize(QSize(50, 100));
        DP_vScale->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignTop);

        DPplateScale->addWidget(DP_vScale);


        DPPlate->addLayout(DPplateScale);


        C1_DA_DPLayout->addLayout(DPPlate);

        C1_DA_DPLayout->setStretch(0, 1);
        C1_DA_DPLayout->setStretch(1, 1);
        C1_DA_DPLayout->setStretch(2, 1);

        imageplatesLayout->addLayout(C1_DA_DPLayout);

        imageplatesLayout->setStretch(0, 1);
        imageplatesLayout->setStretch(1, 1);

        imageplates3DvisLayout->addLayout(imageplatesLayout);

        GL3Dvis = new QDCS3Dvis(centralwidget);
        GL3Dvis->setObjectName(QString::fromUtf8("GL3Dvis"));
        sizePolicy.setHeightForWidth(GL3Dvis->sizePolicy().hasHeightForWidth());
        GL3Dvis->setSizePolicy(sizePolicy);
        GL3Dvis->setMinimumSize(QSize(300, 264));

        imageplates3DvisLayout->addWidget(GL3Dvis);

        imageplates3DvisLayout->setStretch(0, 1);
        imageplates3DvisLayout->setStretch(1, 1);

        simElementsLayout->addLayout(imageplates3DvisLayout);

        stats_graphs_Layout = new QHBoxLayout();
        stats_graphs_Layout->setSpacing(0);
        stats_graphs_Layout->setObjectName(QString::fromUtf8("stats_graphs_Layout"));
        stats_graphs_Layout->setSizeConstraint(QLayout::SetNoConstraint);
        stats_graphs_Layout->setContentsMargins(5, -1, 5, -1);
        statsLayout = new QHBoxLayout();
        statsLayout->setObjectName(QString::fromUtf8("statsLayout"));
        statsLayout->setSizeConstraint(QLayout::SetFixedSize);
        statsLayout->setContentsMargins(-1, -1, 0, -1);
        statesLabels = new QVBoxLayout();
        statesLabels->setObjectName(QString::fromUtf8("statesLabels"));
        statesLabels->setSizeConstraint(QLayout::SetFixedSize);
        statsLabels2 = new QVBoxLayout();
        statsLabels2->setObjectName(QString::fromUtf8("statsLabels2"));
        Ecnts_label = new QLabel(centralwidget);
        Ecnts_label->setObjectName(QString::fromUtf8("Ecnts_label"));
        sizePolicy1.setHeightForWidth(Ecnts_label->sizePolicy().hasHeightForWidth());
        Ecnts_label->setSizePolicy(sizePolicy1);
        Ecnts_label->setMinimumSize(QSize(154, 13));
        Ecnts_label->setMaximumSize(QSize(154, 13));

        statsLabels2->addWidget(Ecnts_label);

        C1cnts_label = new QLabel(centralwidget);
        C1cnts_label->setObjectName(QString::fromUtf8("C1cnts_label"));
        sizePolicy1.setHeightForWidth(C1cnts_label->sizePolicy().hasHeightForWidth());
        C1cnts_label->setSizePolicy(sizePolicy1);
        C1cnts_label->setMinimumSize(QSize(154, 13));
        C1cnts_label->setMaximumSize(QSize(154, 13));

        statsLabels2->addWidget(C1cnts_label);

        C2Pcnts_label = new QLabel(centralwidget);
        C2Pcnts_label->setObjectName(QString::fromUtf8("C2Pcnts_label"));
        sizePolicy1.setHeightForWidth(C2Pcnts_label->sizePolicy().hasHeightForWidth());
        C2Pcnts_label->setSizePolicy(sizePolicy1);
        C2Pcnts_label->setMinimumSize(QSize(154, 13));
        C2Pcnts_label->setMaximumSize(QSize(154, 13));

        statsLabels2->addWidget(C2Pcnts_label);

        DPcnts_label = new QLabel(centralwidget);
        DPcnts_label->setObjectName(QString::fromUtf8("DPcnts_label"));
        sizePolicy1.setHeightForWidth(DPcnts_label->sizePolicy().hasHeightForWidth());
        DPcnts_label->setSizePolicy(sizePolicy1);
        DPcnts_label->setMinimumSize(QSize(154, 13));
        DPcnts_label->setMaximumSize(QSize(154, 13));

        statsLabels2->addWidget(DPcnts_label);

        C2Acnts_label = new QLabel(centralwidget);
        C2Acnts_label->setObjectName(QString::fromUtf8("C2Acnts_label"));
        sizePolicy1.setHeightForWidth(C2Acnts_label->sizePolicy().hasHeightForWidth());
        C2Acnts_label->setSizePolicy(sizePolicy1);
        C2Acnts_label->setMinimumSize(QSize(154, 13));
        C2Acnts_label->setMaximumSize(QSize(154, 13));

        statsLabels2->addWidget(C2Acnts_label);

        DAcnts_label = new QLabel(centralwidget);
        DAcnts_label->setObjectName(QString::fromUtf8("DAcnts_label"));
        sizePolicy1.setHeightForWidth(DAcnts_label->sizePolicy().hasHeightForWidth());
        DAcnts_label->setSizePolicy(sizePolicy1);
        DAcnts_label->setMinimumSize(QSize(154, 13));
        DAcnts_label->setMaximumSize(QSize(154, 13));

        statsLabels2->addWidget(DAcnts_label);


        statesLabels->addLayout(statsLabels2);

        currRot_label = new QLabel(centralwidget);
        currRot_label->setObjectName(QString::fromUtf8("currRot_label"));
        sizePolicy1.setHeightForWidth(currRot_label->sizePolicy().hasHeightForWidth());
        currRot_label->setSizePolicy(sizePolicy1);
        currRot_label->setMinimumSize(QSize(156, 53));
        currRot_label->setMaximumSize(QSize(156, 53));

        statesLabels->addWidget(currRot_label);


        statsLayout->addLayout(statesLabels);

        statsValues = new QVBoxLayout();
        statsValues->setObjectName(QString::fromUtf8("statsValues"));
        statsValues->setSizeConstraint(QLayout::SetFixedSize);
        statsValues2 = new QVBoxLayout();
        statsValues2->setObjectName(QString::fromUtf8("statsValues2"));
        Ecnts_val = new QLabel(centralwidget);
        Ecnts_val->setObjectName(QString::fromUtf8("Ecnts_val"));
        sizePolicy1.setHeightForWidth(Ecnts_val->sizePolicy().hasHeightForWidth());
        Ecnts_val->setSizePolicy(sizePolicy1);
        Ecnts_val->setMinimumSize(QSize(107, 13));
        Ecnts_val->setMaximumSize(QSize(107, 13));

        statsValues2->addWidget(Ecnts_val);

        C1cnts_val = new QLabel(centralwidget);
        C1cnts_val->setObjectName(QString::fromUtf8("C1cnts_val"));
        sizePolicy1.setHeightForWidth(C1cnts_val->sizePolicy().hasHeightForWidth());
        C1cnts_val->setSizePolicy(sizePolicy1);
        C1cnts_val->setMinimumSize(QSize(107, 13));
        C1cnts_val->setMaximumSize(QSize(107, 13));

        statsValues2->addWidget(C1cnts_val);

        C2Pcnts_val = new QLabel(centralwidget);
        C2Pcnts_val->setObjectName(QString::fromUtf8("C2Pcnts_val"));
        sizePolicy1.setHeightForWidth(C2Pcnts_val->sizePolicy().hasHeightForWidth());
        C2Pcnts_val->setSizePolicy(sizePolicy1);
        C2Pcnts_val->setMinimumSize(QSize(107, 13));
        C2Pcnts_val->setMaximumSize(QSize(107, 13));

        statsValues2->addWidget(C2Pcnts_val);

        DPcnts_val = new QLabel(centralwidget);
        DPcnts_val->setObjectName(QString::fromUtf8("DPcnts_val"));
        sizePolicy1.setHeightForWidth(DPcnts_val->sizePolicy().hasHeightForWidth());
        DPcnts_val->setSizePolicy(sizePolicy1);
        DPcnts_val->setMinimumSize(QSize(107, 13));
        DPcnts_val->setMaximumSize(QSize(107, 13));

        statsValues2->addWidget(DPcnts_val);

        C2Acnts_val = new QLabel(centralwidget);
        C2Acnts_val->setObjectName(QString::fromUtf8("C2Acnts_val"));
        sizePolicy1.setHeightForWidth(C2Acnts_val->sizePolicy().hasHeightForWidth());
        C2Acnts_val->setSizePolicy(sizePolicy1);
        C2Acnts_val->setMinimumSize(QSize(107, 13));
        C2Acnts_val->setMaximumSize(QSize(107, 13));

        statsValues2->addWidget(C2Acnts_val);

        DAcnts_val = new QLabel(centralwidget);
        DAcnts_val->setObjectName(QString::fromUtf8("DAcnts_val"));
        sizePolicy1.setHeightForWidth(DAcnts_val->sizePolicy().hasHeightForWidth());
        DAcnts_val->setSizePolicy(sizePolicy1);
        DAcnts_val->setMinimumSize(QSize(107, 13));
        DAcnts_val->setMaximumSize(QSize(107, 13));

        statsValues2->addWidget(DAcnts_val);


        statsValues->addLayout(statsValues2);

        currRot_val = new QLabel(centralwidget);
        currRot_val->setObjectName(QString::fromUtf8("currRot_val"));
        sizePolicy1.setHeightForWidth(currRot_val->sizePolicy().hasHeightForWidth());
        currRot_val->setSizePolicy(sizePolicy1);
        currRot_val->setMinimumSize(QSize(109, 53));
        currRot_val->setMaximumSize(QSize(109, 53));

        statsValues->addWidget(currRot_val);


        statsLayout->addLayout(statsValues);


        stats_graphs_Layout->addLayout(statsLayout);

        histogramsLayout = new QHBoxLayout();
        histogramsLayout->setSpacing(160);
        histogramsLayout->setObjectName(QString::fromUtf8("histogramsLayout"));
        histogramsLayout->setSizeConstraint(QLayout::SetNoConstraint);
        AP_histogram = new QCustomPlot(centralwidget);
        AP_histogram->setObjectName(QString::fromUtf8("AP_histogram"));
        sizePolicy.setHeightForWidth(AP_histogram->sizePolicy().hasHeightForWidth());
        AP_histogram->setSizePolicy(sizePolicy);
        AP_histogram->setMinimumSize(QSize(222, 200));
        AP_histogram->setMaximumSize(QSize(800, 800));
        AP_histogram->setSizeIncrement(QSize(20, 20));
        AP_histogram->setBaseSize(QSize(222, 200));

        histogramsLayout->addWidget(AP_histogram);

        P_histogram = new QCustomPlot(centralwidget);
        P_histogram->setObjectName(QString::fromUtf8("P_histogram"));
        sizePolicy.setHeightForWidth(P_histogram->sizePolicy().hasHeightForWidth());
        P_histogram->setSizePolicy(sizePolicy);
        P_histogram->setMinimumSize(QSize(222, 200));
        P_histogram->setMaximumSize(QSize(800, 800));
        P_histogram->setSizeIncrement(QSize(20, 20));
        P_histogram->setBaseSize(QSize(222, 200));

        histogramsLayout->addWidget(P_histogram);

        histogramsLayout->setStretch(0, 1);
        histogramsLayout->setStretch(1, 1);

        stats_graphs_Layout->addLayout(histogramsLayout);

        stats_graphs_Layout->setStretch(0, 1);
        stats_graphs_Layout->setStretch(1, 2);

        simElementsLayout->addLayout(stats_graphs_Layout);

        simElementsLayout->setStretch(0, 1);
        simElementsLayout->setStretch(1, 1);

        mainLayout->addLayout(simElementsLayout);

        mainLayout->setStretch(0, 1);
        mainLayout->setStretch(1, 20);

        gridLayout->addLayout(mainLayout, 0, 0, 1, 1);

        logBox = new QPlainTextEdit(centralwidget);
        logBox->setObjectName(QString::fromUtf8("logBox"));
        sizePolicy.setHeightForWidth(logBox->sizePolicy().hasHeightForWidth());
        logBox->setSizePolicy(sizePolicy);
        logBox->setMinimumSize(QSize(1024, 50));

        gridLayout->addWidget(logBox, 1, 0, 1, 1);

        SimulationMain->setCentralWidget(centralwidget);
        statusbar = new QStatusBar(SimulationMain);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        SimulationMain->setStatusBar(statusbar);

        retranslateUi(SimulationMain);

        QMetaObject::connectSlotsByName(SimulationMain);
    } // setupUi

    void retranslateUi(QMainWindow *SimulationMain)
    {
        SimulationMain->setWindowTitle(QCoreApplication::translate("SimulationMain", "Double Crystal Spectrometer Simulation", nullptr));
        simstart->setText(QCoreApplication::translate("SimulationMain", "Simulation start at:      xxh    xxm     xxs", nullptr));
        simremain->setText(QCoreApplication::translate("SimulationMain", "Remaining time estimate:      xxh    xxm     xxs", nullptr));
#if QT_CONFIG(statustip)
        E_ImagePlate->setStatusTip(QCoreApplication::translate("SimulationMain", "Counts image plate at entrance", nullptr));
#endif // QT_CONFIG(statustip)
        E_ImagePlate->setText(QString());
        E_cScale->setText(QString());
        E_vScale->setText(QString());
#if QT_CONFIG(statustip)
        C2A_ImagePlate->setStatusTip(QCoreApplication::translate("SimulationMain", "Counts image plate after second crystal reflection in antiparallel", nullptr));
#endif // QT_CONFIG(statustip)
        C2A_ImagePlate->setText(QString());
        C2A_cScale->setText(QString());
        C2A_vScale->setText(QString());
#if QT_CONFIG(statustip)
        C2P_ImagePlate->setStatusTip(QCoreApplication::translate("SimulationMain", "Counts image plate after second crystal reflection in parallel", nullptr));
#endif // QT_CONFIG(statustip)
        C2P_ImagePlate->setText(QString());
        C2P_cScale->setText(QString());
        C2P_vScale->setText(QString());
#if QT_CONFIG(statustip)
        C1_ImagePlate->setStatusTip(QCoreApplication::translate("SimulationMain", "Counts image plate after first crystal reflection", nullptr));
#endif // QT_CONFIG(statustip)
        C1_ImagePlate->setText(QString());
        C1_cScale->setText(QString());
        C1_vScale->setText(QString());
#if QT_CONFIG(statustip)
        DA_ImagePlate->setStatusTip(QCoreApplication::translate("SimulationMain", "Counts image plate at detector in antiparallel", nullptr));
#endif // QT_CONFIG(statustip)
        DA_ImagePlate->setText(QString());
        DA_cScale->setText(QString());
        DA_vScale->setText(QString());
#if QT_CONFIG(statustip)
        DP_ImagePlate->setStatusTip(QCoreApplication::translate("SimulationMain", "Counts image plate at detector in parallel", nullptr));
#endif // QT_CONFIG(statustip)
        DP_ImagePlate->setText(QString());
        DP_cScale->setText(QString());
        DP_vScale->setText(QString());
#if QT_CONFIG(tooltip)
        GL3Dvis->setToolTip(QString());
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(statustip)
        GL3Dvis->setStatusTip(QCoreApplication::translate("SimulationMain", "3D Visualization of the geometry and events being simulated.", nullptr));
#endif // QT_CONFIG(statustip)
        Ecnts_label->setText(QCoreApplication::translate("SimulationMain", "Number of counts at entrance:", nullptr));
        C1cnts_label->setText(QCoreApplication::translate("SimulationMain", "Number of counts at C1:", nullptr));
        C2Pcnts_label->setText(QCoreApplication::translate("SimulationMain", "Number of counts at C2_para:", nullptr));
        DPcnts_label->setText(QCoreApplication::translate("SimulationMain", "Number of counts at detc_para:", nullptr));
        C2Acnts_label->setText(QCoreApplication::translate("SimulationMain", "Number of counts at C2_anti:", nullptr));
        DAcnts_label->setText(QCoreApplication::translate("SimulationMain", "Number of counts at detc_anti:", nullptr));
        currRot_label->setText(QCoreApplication::translate("SimulationMain", "Current rot:", nullptr));
        Ecnts_val->setText(QString());
        C1cnts_val->setText(QString());
        C2Pcnts_val->setText(QString());
        DPcnts_val->setText(QString());
        C2Acnts_val->setText(QString());
        DAcnts_val->setText(QString());
        currRot_val->setText(QString());
#if QT_CONFIG(statustip)
        AP_histogram->setStatusTip(QCoreApplication::translate("SimulationMain", "Antiparallel spectrum. The energy axis is only accurate after suficient statistics have been accumulated (>~100000 events per bin)", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(statustip)
        P_histogram->setStatusTip(QCoreApplication::translate("SimulationMain", "Parallel spectrum", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(tooltip)
        logBox->setToolTip(QString());
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(statustip)
        logBox->setStatusTip(QCoreApplication::translate("SimulationMain", "LogBox for the console output.", nullptr));
#endif // QT_CONFIG(statustip)
    } // retranslateUi

};

namespace Ui {
    class SimulationMain: public Ui_SimulationMain {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SIMULATIONMAIN_H
