#ifndef PARAMETER_SETTINGS_H
#define PARAMETER_SETTINGS_H

#include "simuGlobals.hh"
#include "simulation_interface.h"

#include <QMainWindow>
#include <QTimer>


QT_BEGIN_NAMESPACE
namespace Ui { class GUISettingsWindow; }
QT_END_NAMESPACE

class GUISettingsWindow : public QMainWindow
{
    Q_OBJECT

public:
    GUISettingsWindow(QWidget *parent = nullptr);
    ~GUISettingsWindow();

    inline SimulationInterface* getSimulationWindow() { return wsimu; }
    void setup();
    void updateElements();
    void energyFileMissingDialog();

private:
    SimulationInterface* wsimu;
    Ui::GUISettingsWindow* ui;

    QTimer* timer;
};
#endif // PARAMETER_SETTINGS_H
