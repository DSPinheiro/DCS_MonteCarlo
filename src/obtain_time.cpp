/*
 * Obtain_time.cpp
 *
 *  Created on: Nov 1, 2020
 *      Author: danip
 */


#include "../include/obtain_time.hh"


Obtain_time::Obtain_time(SimulationInterface *w)
{
    startingTime = std::chrono::system_clock::now();
    
    time_t tt = std::chrono::system_clock::to_time_t(startingTime);

    tm local_tm = *localtime(&tt);

    int Hour = local_tm.tm_hour;
    int Min = local_tm.tm_min;
    int Sec = local_tm.tm_sec;


    SimulationInterface::Times times = { 0, Hour, Min, Sec };

    #ifdef QT_EXISTS
        emit w->changeTimesSignal(times);
    #endif

    std::stringstream logString;
    
    logString.clear();
    logString << std::endl;
    logString << "Simulation start at: " << Hour << " h " << Min << " m " << Sec << " s" << std::endl;
    logString << std::endl;

    #ifdef QT_EXISTS
        emit w->LogLineSignal(logString.str());
    #else
        std::cout << logString.str();
    #endif
    
    gener_out << std::endl;
    gener_out << "Simulation start at: " << Hour << " h " << Min << " m " << Sec << " s" << std::endl;
    gener_out << std::endl;
}


void Obtain_time::simuTime(bool finished, double pctDone, SimulationInterface *w)
{
    
    std::stringstream logString;

    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();

    updateTotalDuration = now - startingTime;

    std::chrono::duration<float, std::nano> totalSimuTime = updateTotalDuration / pctDone;
    std::chrono::duration<float, std::nano> remainSimuTime = totalSimuTime - updateTotalDuration;


    if(!finished)
    {
        int Hour = std::chrono::duration_cast<std::chrono::hours>(remainSimuTime).count();
        int Min = std::chrono::duration_cast<std::chrono::minutes>(remainSimuTime).count() - Hour * 60;
        int Sec = std::chrono::duration_cast<std::chrono::seconds>(remainSimuTime).count() - Hour * 3600 - Min * 60;

        SimulationInterface::Times times = { 1, Hour, Min, Sec };

        #ifdef QT_EXISTS
            emit w->changeTimesSignal(times);
        #endif

        logString.clear();
        logString << "Remainder time estimate: " << Hour << " h " << Min << " m " << Sec << " s" << std::endl;

        #ifdef QT_EXISTS
            logString << std::endl;
            emit w->LogLineSignal(logString.str());
        #else
            std::cout << "\r" << logString.str() << std::flush;
        #endif

        gener_out << std::endl;
        gener_out << "Remainder time estimate: " << Hour << " h " << Min << " m " << Sec << " s" << std::endl;
        gener_out << std::endl;
    }
    else
    {
        time_t tt = std::chrono::system_clock::to_time_t(now);

        tm local_tm = *localtime(&tt);

        int Hour = local_tm.tm_hour;
        int Min = local_tm.tm_min;
        int Sec = local_tm.tm_sec;
    
        int HourT = std::chrono::duration_cast<std::chrono::hours>(updateTotalDuration).count();
        int MinT = std::chrono::duration_cast<std::chrono::minutes>(updateTotalDuration).count() - HourT * 60;
        int SecT = std::chrono::duration_cast<std::chrono::seconds>(updateTotalDuration).count() - HourT * 3600 - MinT * 60;

        logString.clear();
        logString << "Simulation end at: " << Hour << " h " << Min << " m " << Sec << " s" << std::endl;
        logString << std::endl;
        logString << "Total time of simulation: " << HourT << " h " << MinT << " m " << SecT << " s" << std::endl;
        logString << std::endl;

        #ifdef QT_EXISTS
            emit w->LogLineSignal(logString.str());
        #else
            std::cout << logString.str();
        #endif

        gener_out << "Simulation end at: " << Hour << " h " << Min << " m " << Sec << " s" << std::endl;
        gener_out << std::endl;
        gener_out << "Total time of simulation: " << HourT << " h " << MinT << " m " << SecT << " s" << std::endl;
        gener_out << std::endl;
    }
}
