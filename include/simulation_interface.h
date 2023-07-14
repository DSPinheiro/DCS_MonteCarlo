#ifndef SIMULATION_INTERFACE_H
#define SIMULATION_INTERFACE_H

#include <algorithm>
#include <chrono>

#include "simuGlobals.hh"

#ifdef QT_EXISTS
#include <QMainWindow>
#include "../include/qcustomplot.h"

namespace Ui {
class SimulationWindow;
}

class SimulationInterface : public QMainWindow
{
    Q_OBJECT

public:
    explicit SimulationInterface(QWidget *parent = nullptr);
    ~SimulationInterface();
    void showEvent(QShowEvent *);
    void closeEvent(QCloseEvent *event);
    void guiSimu();
    static void headlessSimu();

    inline bool isRunning() const { return running; }
    inline float getPctDone() const { return simulation_pct.load(); }
    inline void setPctDone(double pct) { simulation_pct.store(pct); }
    inline bool isOpen() const { return open; }
    
    struct Stats
    {
        int c_sour;
        int c_cr1;
        int c_cr2_para;
        int c_cr2_anti;
        int c_detc_para;
        int c_detc_anti;
        double delrot;
        std::vector<std::vector<double>> events_para;
        std::vector<std::vector<double>> events_anti;
    };

    struct Plots
    {
        std::vector<plot> para;
        std::vector<plot> anti;
    };

    struct Plates
    {
        double hist_image[n_his_ima][n_his_ima];
        double max_z;
        int crystal;
    };

    struct Times
    {
        int timeSlot;
        int h;
        int m;
        int s;
    };

    QCustomPlot *P_hist;
    QCustomPlot *AP_hist;

signals:
    void changeStatsSignal(Stats stats);
    void changePlotsSignal(Plots plots);
    void changePlatesSignal(Plates plates);
    void changeTimesSignal(Times times);
    void LogLineSignal(std::string line);
    void setTetaTableSignal(double teta);
    void showDoneDialogSignal();
    
public slots:
    void changeStats(Stats stats);
    void changePlots(Plots plots);
    void changePlates(Plates plates);
    void changeTimes(Times times);
    void LogLine(std::string line);
    void setTetaTable(double teta);
    void showDoneDialog();
    void startSimulationThread();
    
private:
    Ui::SimulationWindow *ui;
    bool running;
    bool open;
    std::atomic<double> simulation_pct;

    long int lastHistogramUpdate;
};
#else
class SimulationInterface
{
public:
    static void headlessSimu();


    struct Times
    {
        int timeSlot;
        int h;
        int m;
        int s;
    };
    
private:
    static inline double interpolate( double val, double y0, double x0, double y1, double x1 )
    {
      return (val-x0)*(y1-y0)/(x1-x0) + y0;
    }
    static inline double blue( double grayscale )
    {
      if ( grayscale < -0.33 )
          return 0.0;
      else if ( grayscale < 0.33 )
          return interpolate( grayscale, 1.0, -0.33, 0.0, 0.33 );
      else
          return 0.0;
    }
    static inline double green( double grayscale )
    {
      if ( grayscale < -1.0 )
          return 0.0; // unexpected grayscale value
      if  ( grayscale < -0.33 )
          return interpolate( grayscale, 0.0, -1.0, 1.0, -0.33 );
      else if ( grayscale < 0.33 )
          return 1.0;
      else if ( grayscale <= 1.0 )
          return interpolate( grayscale, 1.0, 0.33, 0.0, 1.0 );
      else
          return 1.0; // unexpected grayscale value
    }
    static inline double red( double grayscale )
    {
      if ( grayscale < -0.33 )
          return 0.0;
      else if ( grayscale < 0.33 )
          return interpolate( grayscale, 0.0, -0.33, 1.0, 0.33 );
      else
          return 1.0;
    }

};
#endif
#endif // SIMULATION_INTERFACE_H
