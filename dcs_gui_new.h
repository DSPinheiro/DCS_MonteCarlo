#ifndef DCS_GUI_NEW_H
#define DCS_GUI_NEW_H

#include <QMainWindow>
#include <QTimer>

#include "simuGlobals.hh"
#include "simulationmain.h"

QT_BEGIN_NAMESPACE
namespace Ui { class GUISettingsWindow; }
QT_END_NAMESPACE

class GUISettingsWindow : public QMainWindow
{
    Q_OBJECT

public:
    GUISettingsWindow(QWidget *parent = nullptr);
    ~GUISettingsWindow();

    inline SimulationMain* getSimulationWindow() { return wsimu; }

private:
    SimulationMain* wsimu;
    Ui::GUISettingsWindow* ui;
    void updateElements();

    QTimer* timer;
};
#endif // DCS_GUI_NEW_H
