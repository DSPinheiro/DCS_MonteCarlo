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

    struct Stats
    {
        double c_sour;
        double c_cr1;
        double c_cr2_para;
        double c_cr2_anti;
        double c_detc_para;
        double c_detc_anti;
        double delrot;
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

public slots:
    void changeStats(Stats stats);
    void changePlots(Plots plots);
    void changePlates(Plates plates);
    void changeTimes(Times times);

private:
    Ui::SimulationMain *ui;

    long int lastHistogramUpdate;

    static inline std::vector<std::string> split(std::string s, std::string delimiter)
    {
        size_t last = 0;
        size_t next = std::string::npos;

        std::vector<std::string> tokens;
        std::string token;

        while ((next = s.find(delimiter, last)) != std::string::npos)
        {
            token = s.substr(last, next - last);

            last = next + delimiter.length();

            tokens.push_back(token);
        }

        tokens.push_back(s.substr(last, next));

        return tokens;
    }
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
