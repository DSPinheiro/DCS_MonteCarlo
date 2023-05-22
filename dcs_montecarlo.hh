/*
 * DCS_MonteCarlo.hh
 *
 *  Created on: Oct 29, 2020
 *      Author: danip, cgodinho
 */

#ifndef DCS_MONTECARLO_HH_
#define DCS_MONTECARLO_HH_

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdio.h>
#include <vector>
#include <sstream>
#include <windows.h>
#include <sys/stat.h>
#include <math.h>
#include <algorithm>
#include <cctype>
#include <locale>

#include <QApplication>

#include "obtain_energyspectrum.hh"
#include "CheckInputSpectrum.hh"
#include "double_crystal_diffraction.hh"
#include "simuGlobals.hh"
#include "dcs_gui.h"
#include "dcs_gui_new.h"


Geometry Geometry;
UserSettings UserSettings;
GeoParapathlengths GeoParapathlengths;
Geolengthelements Geolengthelements;
GeoParameters GeoParameters;
CurveVerticalTilt CurveVerticalTilt;
Graph_options Graph_options;
plotparameters plotparameters;
numberrays numberrays;
physical_parameters physical_parameters;
polarization_parameters polarization_parameters;
temperature_parameters temperature_parameters;
fullenergyspectrum fullenergyspectrum;
Curved_Crystal Curved_Crystal;
AnalysiesCrystaltilts AnalysiesCrystaltilts;

double reques_energ[4];
double reques_width[4];

double linelamda, naturalwidth;

double linelambda1, naturalwidth1, linelambda2, naturalwidth2, linelambda3, naturalwidth3, linelambda4, naturalwidth4;

double request_width[4], request_energy[4];

double teta_crys1;

int i;

bool exist_file, usable;

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


// trim from start (in place)
static inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

// trim from both ends (in place)
static inline void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}


#endif /* DCS_MONTECARLO_HH_ */
