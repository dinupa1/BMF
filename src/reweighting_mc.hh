#ifndef _REWEIGHTING_MC__HH_
#define _REWEIGHTING_MC__HH_

#include <TFile.h>
#include <TTree.h>
#include <TString.h>
#include "reweighting_plots.hh"

class reweighting_mc
{
  int bins = 20;
  reweighting_plots* mass_plots = new reweighting_plots("hmass", "; mass [GeV]; counts", bins, 4.5, 9.0);
  reweighting_plots* pT_plots = new reweighting_plots("hpT", "; pT [GeV]; counts", bins, 0.0, 2.5);
//   reweighting_plots* xB_plots;
//   reweighting_plots* xT_plots;
//   reweighting_plots* xF_plots;

public:
  reweighting_mc();
  virtual ~reweighting_mc(){;}
  void cuts();
  void train_model();
  void ana();
  void plot();
};

#endif // _REWEIGHTING_MC__HH_