#include <TFile.h>
#include <TTree.h>
#include <TSystem.h>
#include <TString.h>
#include <iostream>
#include "reweighting_mc.hh"

reweighting_mc::reweighting_mc(){;}

void reweighting_mc::cuts()
{
  gSystem->Exec("python e906_data_cuts.py");
}

void reweighting_mc::train_model()
{
  gSystem->Exec("python reweighting_messy_mc.py");
}

void reweighting_mc::ana()
{
  gSystem->Exec("python compare_plots.py");

  int bins = 20;

  auto infile = TFile::Open("/Users/dinupa/seaquest/e906-LH2-data/output.root", "READ");
  auto tree = (TTree*)infile->Get("tree");
  auto tree_mc = (TTree*)infile->Get("tree_mc");

  int nevents = tree->GetEntries();
  int nevents_mc = tree_mc->GetEntries();

  std::cout << "---> mc events " << nevents_mc << " real events " <<  nevents <<std::endl;

  double mass, pT, xT, xB, xF, weight;
  double mass_mc, pT_mc, xT_mc, xB_mc, xF_mc, weight_mc, weight2_mc;

  tree->SetBranchAddress("mass", &mass);
  tree->SetBranchAddress("pT", &pT);
  tree->SetBranchAddress("xT", &xT);
  tree->SetBranchAddress("xB", &xB);
  tree->SetBranchAddress("xF", &xF);
  tree->SetBranchAddress("weight", &weight);

  tree_mc->SetBranchAddress("mass", &mass_mc);
  tree_mc->SetBranchAddress("pT", &pT_mc);
  tree_mc->SetBranchAddress("xT", &xT_mc);
  tree_mc->SetBranchAddress("xB", &xB_mc);
  tree_mc->SetBranchAddress("xF", &xF_mc);
  tree_mc->SetBranchAddress("weight", &weight_mc);
  tree_mc->SetBranchAddress("weight2", &weight2_mc);

//  mass_plots = new reweighting_plots("hmass", "; mass [GeV]; counts", bins, 4.5, 9.0);
//  pT_plots = new reweighting_plots("hpT", "; pT [GeV]; counts", bins, 0.0, 2.5);
//  xT_plots = new reweighting_plots("hxT", "; xT; counts", bins, 0.1, 0.6);
//  xB_plots = new reweighting_plots("hxB", "; xB; counts", bins, 0.3, 1.0);
//  xF_plots = new reweighting_plots("hxF", "; xF; counts", bins, -0.1, 1.0);

  for(int ii = 0; ii < nevents; ii++)
  {
    tree->GetEntry(ii);
    tree_mc->GetEntry(ii);

    mass_plots->fill_hist(mass, mass_mc, weight, weight_mc, weight2_mc);
    pT_plots->fill_hist(pT, pT_mc, weight, weight_mc, weight2_mc);
//    xT_plots->fill_hist(xT, xT_mc, weight, weight_mc, weight2_mc);
//    xB_plots->fill_hist(xB, xB_mc, weight, weight_mc, weight2_mc);
//    xF_plots->fill_hist(xF, xF_mc, weight, weight_mc, weight2_mc);
  }
}

void reweighting_mc::plot()
{
  mass_plots->plot_hist();
  mass_plots->plot_ratio();

  pT_plots->plot_hist();
  pT_plots->plot_ratio();
//
//  xT_plots->plot_hist();
//  xT_plots->plot_ratio();
//
//  xB_plots->plot_hist();
//  xB_plots->plot_ratio();
//
//  xF_plots->plot_hist();
//  xF_plots->plot_ratio();
}