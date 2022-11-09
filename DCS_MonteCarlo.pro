QT       += opengl core gui
QT       += printsupport

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11
CONFIG += console

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    CheckInputSpectrum.cpp \
    analysie_voigt.cpp \
    calculate_geo_corr.cpp \
    dcs_montecarlo.cpp \
    double_crystal_diffraction.cpp \
    gaussi_rockin.cpp \
    get_new_temp_for_bin.cpp \
    intensitysource.cpp \
    limits_horizontal_diverge.cpp \
    dcs_gui.cpp \
    make_graph_imageplates.cpp \
    make_plot_profiles.cpp \
    make_pointcryst.cpp \
    makefitvoigt.cpp \
    mask_test_c2.cpp \
    obtain_curve_responce.cpp \
    obtain_curved_hor_corr.cpp \
    obtain_data_fitting.cpp \
    obtain_energyspectrum.cpp \
    obtain_misalign.cpp \
    obtain_time.cpp \
    obtainvertical.cpp \
    qcustomplot.cpp \
    qdcs3dvis.cpp \
    set_ang_rad.cpp \
    simulationmain.cpp \
    source_complex.cpp \
    source_simple.cpp \
    test_input.cpp

HEADERS += \
    Boxmuller.hh \
    CheckInputSpectrum.hh \
    FindLoc.hh \
    First_crystal_angle.hh \
    First_crystal_angle_1_Approximation.hh \
    First_crystal_angle_Full.hh \
    Obtain_d_Lattice_temp.hh \
    Obtain_ypzp.hh \
    PseuVoigt.hh \
    Reach_detector.hh \
    Second_crystal_angle.hh \
    Second_crystal_angle_Full.hh \
    Second_crystal_angle_aproximation.hh \
    TrueVoig.hh \
    analysie_voigt.hh \
    calculate_geo_corr.hh \
    covsrt.hh \
    cw.hh \
    dcs_gui.h \
    dcs_montecarlo.hh \
    double_crystal_diffraction.hh \
    gaussi_rockin.hh \
    gaussj.hh \
    get_new_temp_for_bin.hh \
    intensitysource.hh \
    limits_horizontal_diverge.hh \
    make_graph_imageplates.hh \
    make_plot_profiles.hh \
    make_pointcryst.hh \
    makefitvoigt.hh \
    mask_test_c2.hh \
    mrqcof.hh \
    mrqmin.hh \
    obtain_curve_responce.hh \
    obtain_curved_hor_corr.hh \
    obtain_data_fitting.hh \
    obtain_energyspectrum.hh \
    obtain_misalign.hh \
    obtain_time.hh \
    obtainvertical.hh \
    qcustomplot.h \
    qdcs3dvis.h \
    set_ang_rad.hh \
    simuGlobals.hh \
    simulationmain.h \
    source_complex.hh \
    source_simple.hh \
    spline.hh \
    splint.hh \
    test_input.hh

FORMS += \
    dcs_gui.ui \
    simulationmain.ui

LIBS += -LC:\Qt\5.15.1\mingw81_64\lib\libQt5OpenGL.a -lopengl32

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

DISTFILES +=
