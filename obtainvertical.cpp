/*
 * ObtainVertical.cpp
 *
 *  Created on: Oct 31, 2020
 *      Author: danip
 */


#include "obtainvertical.hh"

double ObtainVertical::ObtainVert(int crystal, double angle){
    
    double angle_temp, temp, rad, offsettilt, phase_temp, consttilt;

    rad = M_PI / 180.0;

    if(crystal == 1){
        angle_temp = GeoParametersInput.Exp_crys1 * rad;
        phase_temp = CurveVerticalTiltInput.phas_tilt1 * rad;
        offsettilt = CurveVerticalTiltInput.offsettilt1 * rad;
        consttilt = CurveVerticalTiltInput.consttilt1 * rad;
    }else{
        angle_temp = angle * rad;
        phase_temp = CurveVerticalTiltInput.phas_tilt2 * rad;
        offsettilt = CurveVerticalTiltInput.offsettilt2 * rad;
        consttilt = CurveVerticalTiltInput.consttilt2 * rad;
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
