 /*
 * Utils to create data
 * dinupa3@gmail.com
 *
 */


#ifndef _DRELL_YAN_NAMESPACE__HH_
#define _DRELL_YAN_NAMESPACE__HH_

#include <TFile.h>
#include <TTree.h>
#include <TMath.h>
#include <TCanvas.h>
#include <iostream>

namespace drellYan {

    // function to calculate event weights with lambda, mu, nu
    double weight_fn(double lambda, double mu, double nu, double phi, double costh)
    {
        double weight = 1. + lambda* costh* costh + 2.* mu* costh* sqrt(1. - costh* costh) *cos(phi) + 0.5* nu* (1. - costh* costh)* cos(2.* phi);
        return weight/(1. + costh* costh);
    }

    // variables
    double true_mass, true_pT, true_xF, true_x1, true_x2, true_phi, true_costh, mass, pT, xF, x1, x2, phi, costh;
    float occuD1;

    double pi = TMath::Pi();

}

#endif /*_DRELL_YAN_NAMESPACE__HH_*/
