//
// plot the binned histogram
// dinupa3@gmail.com
// 04-12-2023
//

#include <TFile.h>
#include <TH1D.h>
#include <TCanvas.h>
#include <iostream>

using namespace std;

void PlotBinnedHist()
{
  auto file = TFile::Open("clean_plot.root", "read");

  auto can = new TCanvas();

  auto Hmass0 = (TH1D*)file->Get("mass0");
  Hmass0->Draw("COLZ");
  can->SaveAs("imgs/phi_costh_mass0.png");

  auto Hmass1 = (TH1D*)file->Get("mass1");
  Hmass1->Draw("COLZ");
  can->SaveAs("imgs/phi_costh_mass1.png");

  auto HpT0 = (TH1D*)file->Get("pT0");
  HpT0->Draw("COLZ");
  can->SaveAs("imgs/phi_costh_pT0.png");

  auto HpT1 = (TH1D*)file->Get("pT1");
  HpT1->Draw("COLZ");
  can->SaveAs("imgs/phi_costh_pT1.png");

  auto HpT2 = (TH1D*)file->Get("pT2");
  HpT2->Draw("COLZ");
  can->SaveAs("imgs/phi_costh_pT2.png");

  auto HxF0 = (TH1D*)file->Get("xF0");
  HxF0->Draw("COLZ");
  can->SaveAs("imgs/phi_costh_xF0.png");

  auto HxF1 = (TH1D*)file->Get("xF1");
  HxF1->Draw("COLZ");
  can->SaveAs("imgs/phi_costh_xF1.png");

  auto HxF2 = (TH1D*)file->Get("xF2");
  HxF2->Draw("COLZ");
  can->SaveAs("imgs/phi_costh_xF2.png");

  auto HxF3 = (TH1D*)file->Get("xF3");
  HxF3->Draw("COLZ");
  can->SaveAs("imgs/phi_costh_xF3.png");
}