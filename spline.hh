/*
 * spline.hh
 *
 *  Created on: Oct 29, 2020
 *      Author: danip
 */

#ifndef SPLINE_HH_
#define SPLINE_HH_

#include <vector>

using namespace std;

static vector<double> spline(vector<double> x, vector<double> y, double yp1, double ypn){

        vector <double> y2, u;

        double p, qn, sig, un;

        int n = x.size();

        if(yp1 > 0.99E30){
                y2.push_back(0);
                u.push_back(0);
        }else{
                y2.push_back(0.5);
                u.push_back((3.0 / (x[1] - x[0])) * ((y[1] - y[0]) / (x[1] - x[0]) - yp1));
        }

        for(int i = 1; i < n - 1; i++){
                sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
                p = sig * y2[i - 1] + 2;

                y2.push_back((sig - 1) / p);
                u.push_back((6 * ((y[i + 1] - y [i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1])) / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p);
        }

        if(ypn > 0.99E30){
                qn = 0;
                un = 0;
        }else{
                qn = 0.5;
                un = (3.0 / (x[n - 1] - x[n - 2])) * (ypn - (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]));
        }

        y2.push_back((un - qn * u[n - 2]) / (qn * y2[n - 2] + 1));

        for(int k = n - 2; k >= 0; k--){
                y2[k] = y2[k] * y2[k + 1] + u[k];
        }

        return y2;
}

#endif /* SPLINE_HH_ */
