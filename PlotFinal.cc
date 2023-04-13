#include <TH1D.h>
#include <TLegend.h>
#include <TCanvas.h>


void PlotFinal()
{

// custom style
//ROOT.gStyle.SetHistFillColor(422)
//  gStyle->SetHistLineStyle(2);
//  gStyle->SetHistLineWidth(2);
//ROOT.gStyle.SetStatFontSize(0.1)
  gStyle->SetStatY(0.85);
  gStyle->SetStatX(0.85);
  gStyle->SetStatW(0.3);
  gStyle->SetStatH(0.3);
  gStyle->SetOptStat(0);
  gStyle->SetStatBorderSize(0);
  gStyle->SetStatFont(42);
  gStyle->SetCanvasDefH(800);
  gStyle->SetCanvasDefW(800);
  gStyle->SetLegendBorderSize(0);
//  gStyle->SetHistMinimumZero(1);

  auto can = new TCanvas();
//
//
//
  auto mass_lambda = new TH1D("mass_lambda", "; mass [GeV/c^2]; value [a.u.]", 2, 4.5, 6.5);
  auto mass_mu = new TH1D("mass_mu", "; mass [GeV/c^2]; value [a.u.]", 2, 4.5, 6.5);
  auto mass_nu = new TH1D("mass_nu", "; mass [GeV/c^2]; value [a.u.]", 2, 4.5, 6.5);

  mass_lambda->SetMinimum(-3.);
  mass_lambda->SetMaximum(3.);

  mass_lambda->SetBinContent(1, 0.5138);
  mass_lambda->SetBinContent(2, 1.535);
  mass_mu->SetBinContent(1, 0.1245);
  mass_mu->SetBinContent(2, 0.03438);
  mass_nu->SetBinContent(1, 0.101);
  mass_nu->SetBinContent(2, 0.2508);

  mass_lambda->SetMarkerStyle(8);
  mass_lambda->SetMarkerColor(2);

  mass_mu->SetMarkerStyle(8);
  mass_mu->SetMarkerColor(3);

  mass_nu->SetMarkerStyle(8);
  mass_nu->SetMarkerColor(4);

  auto L0 = new TLegend(0.8, 0.1, 0.9, 0.3);
  L0->AddEntry(mass_lambda, "#lambda");
  L0->AddEntry(mass_mu, "#mu");
  L0->AddEntry(mass_nu, "#nu");

  mass_lambda->Draw("P0");
  mass_mu->Draw("SAME P0");
  mass_nu->Draw("SAME P0");
  L0->Draw();
  can->SaveAs("imgs/mass_final.png");

//
//
//
  auto pT_lambda = new TH1D("pT_lambda", "; pT [GeV/c]; value [a.u.]", 3, 0., 1.5);
  auto pT_mu = new TH1D("pT_mu", "; pT [GeV/c]; value [a.u.]", 3, 0., 1.5);
  auto pT_nu = new TH1D("pT_nu", "; pT [GeV/c]; value [a.u.]", 3, 0., 1.5);

  pT_lambda->SetMinimum(-3.);
  pT_lambda->SetMaximum(3.);

  pT_lambda->SetBinContent(1, 0.2832);
  pT_lambda->SetBinContent(2, 0.467);
  pT_lambda->SetBinContent(3, 0.8842);

  pT_mu->SetBinContent(1, -0.07717);
  pT_mu->SetBinContent(2, 0.03002);
  pT_mu->SetBinContent(3, 0.07164);

  pT_nu->SetBinContent(1, 0.1841);
  pT_nu->SetBinContent(2, -0.006442);
  pT_nu->SetBinContent(2, 0.08638);

  pT_lambda->SetMarkerStyle(8);
  pT_lambda->SetMarkerColor(2);

  pT_mu->SetMarkerStyle(8);
  pT_mu->SetMarkerColor(3);

  pT_nu->SetMarkerStyle(8);
  pT_nu->SetMarkerColor(4);

  auto L1 = new TLegend(0.8, 0.1, 0.9, 0.3);
  L1->AddEntry(pT_lambda, "#lambda");
  L1->AddEntry(pT_mu, "#mu");
  L1->AddEntry(pT_nu, "#nu");

  pT_lambda->Draw("P0");
  pT_mu->Draw("SAME P0");
  pT_nu->Draw("SAME P0");
  L1->Draw();
  can->SaveAs("imgs/pT_final.png");

//
//
//
  auto xF_lambda = new TH1D("xF_lambda", "; xF; value [a.u.]", 4, 0., 0.8);
  auto xF_mu = new TH1D("xF_mu", "; xF; value [a.u.]", 4, 0., 0.8);
  auto xF_nu = new TH1D("xF_nu", "; xF; value [a.u.]", 4, 0., 0.8);

  xF_lambda->SetMinimum(-3.);
  xF_lambda->SetMaximum(3.);

  xF_lambda->SetBinContent(1, 1.135);
  xF_lambda->SetBinContent(2, 0.9452);
  xF_lambda->SetBinContent(3, -1.059);
  xF_lambda->SetBinContent(4, -0.4628);

  xF_mu->SetBinContent(1, 0.1927);
  xF_mu->SetBinContent(2, -0.1985);
  xF_mu->SetBinContent(3, -0.04861);
  xF_mu->SetBinContent(4, -0.3923);

  xF_nu->SetBinContent(1, -0.06175);
  xF_nu->SetBinContent(2, 0.1525);
  xF_nu->SetBinContent(2, 0.04097);
  xF_nu->SetBinContent(2, 0.07034);

  xF_lambda->SetMarkerStyle(8);
  xF_lambda->SetMarkerColor(2);

  xF_mu->SetMarkerStyle(8);
  xF_mu->SetMarkerColor(3);

  xF_nu->SetMarkerStyle(8);
  xF_nu->SetMarkerColor(4);

  auto L2 = new TLegend(0.8, 0.1, 0.9, 0.3);
  L2->AddEntry(xF_lambda, "#lambda");
  L2->AddEntry(xF_mu, "#mu");
  L2->AddEntry(xF_nu, "#nu");

  xF_lambda->Draw("P0");
  xF_mu->Draw("SAME P0");
  xF_nu->Draw("SAME P0");
  L2->Draw();
  can->SaveAs("imgs/xF_final.png");

}