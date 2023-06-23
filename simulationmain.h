#ifndef SIMULATIONMAIN_H
#define SIMULATIONMAIN_H

#include <algorithm>
#include <chrono>

#include <QMainWindow>

#include "simuGlobals.hh"

namespace Ui {
class SimulationMain;
}

class SimulationMain : public QMainWindow
{
    Q_OBJECT

public:
    explicit SimulationMain(QWidget *parent = nullptr);
    ~SimulationMain();
    void showEvent(QShowEvent *);
    void closeEvent(QCloseEvent *event);
    void guiSimu();

    inline bool isRunning() const { return running; }
    inline float getPctDone() const { return simulation_pct.load(); }
    inline void setPctDone(float pct) { simulation_pct.store(pct); }
    inline bool isOpen() const { return open; }
    
    struct Stats
    {
        double c_sour;
        double c_cr1;
        double c_cr2_para;
        double c_cr2_anti;
        double c_detc_para;
        double c_detc_anti;
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

signals:
    void changeStatsSignal(Stats stats);
    void changePlotsSignal(Plots plots);
    void changePlatesSignal(Plates plates);
    void changeTimesSignal(Times times);
    void LogLineSignal(std::string line);
    void showDoneDialogSignal();
    
public slots:
    void changeStats(Stats stats);
    void changePlots(Plots plots);
    void changePlates(Plates plates);
    void changeTimes(Times times);
    void LogLine(std::string line);
    void showDoneDialog();
    void startSimulationThread();
    
private:
    Ui::SimulationMain *ui;
    bool running;
    bool open;
    std::atomic<float> simulation_pct;

    long int lastHistogramUpdate;

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

#endif // SIMULATIONMAIN_H
