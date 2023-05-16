/*
 * Obtain_time.cpp
 *
 *  Created on: Nov 1, 2020
 *      Author: danip
 */


#include "obtain_time.hh"


using namespace std;

extern UserSettings UserSettings;
extern Graph_options Graph_options;

extern ofstream gener_out;

vector<int> Obtain_time::simuTime(int First_call, int process_remain, int int_time, int int_time_mili, SimulationMain *w){

    int int_time_temp, int_time_mili_temp, dif_time_a[3], int_time_out, int_time_mili_out;

    double dif_time, dif_mili;

    if(First_call == 0){

        auto now = chrono::system_clock::now();
        auto timer = chrono::system_clock::to_time_t(now);

        tm localTime = *localtime(&timer);

        int Hour   = localTime.tm_hour;
        int Min    = localTime.tm_min;
        int Sec    = localTime.tm_sec;

        if(Graph_options.MakeDislin)
        {
            SimulationMain::Times times = { 0, Hour, Min, Sec };
            emit w->changeTimesSignal(times);
        }

        int_time_out = 60 * ((60 * Hour) + Min) + Sec;
        int_time_mili_out = chrono::duration_cast<chrono::milliseconds>(now.time_since_epoch()).count();

        if(UserSettings.Simple_simu){
            cout << endl;
            cout << "Simulation start at: " << Hour << " h " << Min << " m " << Sec << " s" << endl;
            cout << endl;
        }

        gener_out << endl;
        gener_out << "Simulation start at: " << Hour << " h " << Min << " m " << Sec << " s" << endl;
        gener_out << endl;

        //TODO implement gui
        //if(Graph_options.make_image_plates){
        //	if(UserSettings.Simple_simu){
        //
        //	}else{
        //
        //	}
        //}

    }else{
        int_time_temp = int_time;
        int_time_mili_temp = int_time_mili;

        auto now = chrono::system_clock::now();
        auto timer = chrono::system_clock::to_time_t(now);

        tm localTime = *localtime(&timer);

        int Hour   = localTime.tm_hour;
        int Min    = localTime.tm_min;
        int Sec    = localTime.tm_sec;

        int_time_out = 60 * ((60 * Hour) + Min) + Sec;
        int_time_mili_out = chrono::duration_cast<chrono::milliseconds>(now.time_since_epoch()).count();


        dif_time = int_time_out - int_time_temp;
        dif_mili = (int_time_mili_out - int_time_mili_temp) / 1000;
        dif_time += dif_mili;
        dif_time *= process_remain;

        dif_time_a[0] = 0;
        dif_time_a[1] = 0;
        dif_time_a[2] = 0;

        if(dif_time > 3599){
            dif_time_a[0] = (int)(dif_time / 3600);
            dif_time -= dif_time_a[0] * 3600;
        }

        if(dif_time > 59){
            dif_time_a[1] = (int)(dif_time / 60);
            dif_time -= dif_time_a[1] * 60;
        }

        dif_time_a[2] = dif_time;

        if(First_call == 1){
            if(Graph_options.MakeDislin)
            {
                SimulationMain::Times times = { 1, dif_time_a[0], dif_time_a[1], dif_time_a[2] };
                emit w->changeTimesSignal(times);
            }

            // (César) : This is on the GUI already no need to polute the console (if displaying one)
            // cout << "remainder time estimate: " << dif_time_a[0] << " h " << dif_time_a[1] << " m " << dif_time_a[2] << " s" << endl;
            // cout << endl;
            static int rotate = 0;
            if(rotate % 7 == 0)      cout << "Working       \r";
            else if(rotate % 7 == 1) cout << "Working .     \r";
            else if(rotate % 7 == 2) cout << "Working ..    \r";
            else if(rotate % 7 == 3) cout << "Working ...   \r";
            else if(rotate % 7 == 4) cout << "Working ....  \r";
            else if(rotate % 7 == 5) cout << "Working ..... \r";
            else if(rotate % 7 == 6) cout << "Working ......\r";
            rotate++;

            //TODO implement gui
            //if(Graph_options.make_image_plates){
            //	if(UserSettings.Simple_simu){
            //
            //	}else{
            //
            //	}
            //}
        }else{
            cout << "Simulation end at: " << Hour << " h " << Min << " m " << Sec << " s" << endl;
            cout << endl;
            cout << "Total time of simulation: " << dif_time_a[0] << " h " << dif_time_a[1] << " m " << dif_time_a[2] << " s" << endl;
            cout << endl;

            gener_out << "Simulation end at: " << Hour << " h " << Min << " m " << Sec << " s" << endl;
            gener_out << endl;
            gener_out << "Total time of simulation: " << dif_time_a[0] << " h " << dif_time_a[1] << " m " << dif_time_a[2] << " s" << endl;
            gener_out << endl;
        }

    }

    vector<int> res;
    res.push_back(int_time_out);
    res.push_back(int_time_mili_out);

    return res;

}
