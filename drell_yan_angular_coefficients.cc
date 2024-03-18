R__LOAD_LIBRARY(src/build/libdrell_yan_ang_coef.dylib)

#include <TFile.h>
#include <TTree.h>
#include <TCanvas.h>
#include <TString.h>
#include <TH1D.h>
#include <TF1.h>
#include <TSystem.h>
#include <iostream>

//#include "src/reweighting_plots.cc"
//#include "src/reweighting_mc.cc"

void drell_yan_angular_coefficients()
{
//  gSystem->Load("/Users/dinupa/seaquest/drell_yan_angular_coefficients/src/build/libdrell_yan_ang_coef.dylib");
  reweighting_mc* rmc = new reweighting_mc();
  rmc->ana();
  rmc->plot();
}