/*
 * ObtainVertical.cpp
 *
 *  Created on: Oct 31, 2020
 *      Author: danip
 */


#include "obtainvertical.hh"

extern GeoParameters GeoParameters;
extern CurveVerticalTilt CurveVerticalTilt;

double ObtainVertical::ObtainVert(int crystal, double angle){
    double angle_temp, temp, rad, offsettilt, phase_temp, consttilt;

    rad = M_PI / 180.0;

    if(crystal == 1){
        angle_temp = GeoParameters.Exp_crys1 * rad;
        phase_temp = CurveVerticalTilt.phas_tilt1 * rad;
        offsettilt = CurveVerticalTilt.offsettilt1 * rad;
        consttilt = CurveVerticalTilt.consttilt1 * rad;
    }else{
        angle_temp = angle * rad;
        phase_temp = CurveVerticalTilt.phas_tilt2 * rad;
        offsettilt = CurveVerticalTilt.offsettilt2 * rad;
        consttilt = CurveVerticalTilt.consttilt2 * rad;
    }

    temp = cos(angle_temp + phase_temp);

    double tilt;

    if(temp > 0){
        tilt = - acos(cos(offsettilt) * pow(cos(angle_temp + phase_temp), 2) + pow(sin(angle_temp + phase_temp), 2));
    }else{
        tilt = acos(cos(offsettilt) * pow(cos(angle_temp + phase_temp), 2) + pow(sin(angle_temp + phase_temp), 2));
    }

    return tilt + consttilt;
}
