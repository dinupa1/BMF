#ifndef _REWEIGHTING_PLOTS__HH_
#define _REWEIGHTING_PLOTS__HH_

#include <TH1D.h>
#include <TF1.h>
#include <TCanvas.h>
#include <TString.h>
#include <TSystem.h>
#include <TStyle.h>

class reweighting_plots
{
  TH1D* hist;
  TH1D* hist_mc;
  TH1D* reweighted_hist;
  TF1* fit;
  TCanvas* can;

public:
  reweighting_plots(TString hname, TString htitle, int nbins, double xmin, double xmax);
  virtual ~reweighting_plots();
  void fill_hist(double xval, double xmc, double xweight, double xweight_mc, double xreweight);
  void plot_hist();
  void plot_ratio();
};
#endif // _REWEIGHTING_PLOTS__HH_