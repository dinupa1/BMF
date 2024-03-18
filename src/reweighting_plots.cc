#include <TH1D.h>
#include <TF1.h>
#include <TCanvas.h>
#include <TString.h>
#include <TSystem.h>
#include <TStyle.h>
#include "reweighting_plots.hh"

reweighting_plots::reweighting_plots(TString hname, TString htitle, int nbins, double xmin, double xmax)
{

  gStyle->SetOptStat(0);
  gStyle->SetOptFit(1);

  TH1::SetDefaultSumw2();

  hist = new TH1D(hname.Data(), htitle.Data(), nbins, xmin, xmax);

  TString hname_mc = Form("%s_mc", hname.Data());
  hist_mc = new TH1D(hname_mc.Data(), htitle.Data(), nbins, xmin, xmax);

  TString hname_mc2 = Form("%s_reweighted", hname.Data());
  reweighted_hist = new TH1D(hname_mc2.Data(), htitle.Data(), nbins, xmin, xmax);

  TString nfit = Form("%s_fit", hname.Data());
  fit = new TF1(nfit.Data(), "[0]* x + [1]");
  fit->SetParName(0, "a");
  fit->SetParName(1, "b");

  TString ncan = Form("%s_can", hname.Data());
  can = new TCanvas(ncan.Data(), ncan.Data(), 800, 800);
}

reweighting_plots::~reweighting_plots()
{
  delete hist;
  delete hist_mc;
  delete reweighted_hist;
  delete fit;
  delete can;
}

void reweighting_plots::fill_hist(double xval, double xmc, double xweight, double xweight_mc, double xreweight)
{
  hist->Fill(xval, xweight);
  hist_mc->Fill(xmc, xweight_mc);
  reweighted_hist->Fill(xmc, xreweight);
}

void reweighting_plots::plot_hist()
{
  std::cout << "integral " << hist->Integral() << std::endl;

  hist->Scale(1./hist->Integral());
  hist_mc->Scale(1./hist_mc->Integral());
  reweighted_hist->Scale(1./reweighted_hist->Integral());

  hist->SetFillColorAlpha(kTeal+1, 0.3);

  hist_mc->SetMarkerColor(kViolet+1);
  hist_mc->SetMarkerStyle(20);

  reweighted_hist->SetMarkerColor(kViolet+1);
  reweighted_hist->SetMarkerStyle(20);

  hist->Draw("HIST");
  hist_mc->Draw("SAME E1");

  //can->Update();
  TString save_before = Form("imgs/%s.png", hist->GetName());
  can->SaveAs(save_before.Data());

  hist->Draw("HIST");
  reweighted_hist->Draw("SAME E1");

  //can->Update();
  TString save_after = Form("imgs/%s_reweighted.png", hist->GetName());
  can->SaveAs(save_after.Data());
}


void reweighting_plots::plot_ratio()
{
  TH1D* ratio_mc = (TH1D*)hist_mc->Clone("hratio");

  ratio_mc->Divide(hist);

  ratio_mc->SetMaximum(3.0);
  ratio_mc->SetMinimum(-1.0);

  ratio_mc->Fit(fit);
  ratio_mc->Draw("E1");

  TString ratio_before = Form("imgs/%s_ratio.png", hist->GetName());
//  can->Update();
  can->SaveAs(ratio_before.Data());

  TH1D* ratio_reweighted = (TH1D*)reweighted_hist->Clone("hratio_reweighted");

  ratio_reweighted->Divide(hist);

  ratio_reweighted->SetMaximum(3.0);
  ratio_reweighted->SetMinimum(-1.0);

  ratio_reweighted->Fit(fit);
  ratio_reweighted->Draw("E1");

//  can->Update();
  TString ratio_after = Form("imgs/%s_ratio_reweighted.png", hist->GetName());
  can->SaveAs(ratio_after.Data());

  delete ratio_mc;
  delete ratio_reweighted;
}