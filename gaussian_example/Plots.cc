#include<TFile.h>
#include<TTree.h>
#include<TH1D.h>
#include<TH2D.h>
#include<TMath.h>
#include<TText.h>
#include<TCanvas.h>
#include<iostream>

using namespace std;

TCanvas* can;

void Plots()
{
    gStyle->SetOptStat(0);

    auto inFile = TFile::Open("results.root", "READ");
    auto save = (TTree*)inFile->Get("save");
    int num_bins = 30;
    double xmin = -6.;
    double xmax = 6.;

    float x0, weight0, x0_err, reweight, x1, weight1, x1_err;

    save->SetBranchAddress("x0", &x0);
    save->SetBranchAddress("weight0", &weight0);
    save->SetBranchAddress("x0_err", &x0_err);
    save->SetBranchAddress("reweight", &reweight);
    save->SetBranchAddress("x1", &x1);
    save->SetBranchAddress("weight1", &weight1);
    save->SetBranchAddress("x1_err", &x1_err);

    auto H0 = new TH1D("H0", "; x [a.u.]; normalized to unity [a.u.]", num_bins, xmin, xmax);
    auto H01 = new TH1D("H01", "; x [a.u.]; normalized to unity [a.u.]", num_bins, xmin, xmax);
    auto H1 = new TH1D("H1", "; x [a.u.]; normalized to unity [a.u.]", num_bins, xmin, xmax);

    for(int i = 0; i < num_bins; i++)
    {
        save->GetEntry(i);
        H0->SetBinContent(i+1, weight0);
        H0->SetBinError(i+1, x0_err);

        H01->SetBinContent(i+1, weight0*reweight);
        H01->SetBinError(i+1, x0_err*reweight);

        H1->SetBinContent(i+1, weight1);
        H1->SetBinError(i+1, x1_err);
    }

    can = new TCanvas();

    H0->SetFillColor(kViolet-2);

    H1->SetFillColor(kAzure+6);
    H1->SetFillStyle(3004);

    //X1_hist->SetLineWidth(2);
    H01->SetLineColor(kSpring-2);
    H01->SetLineWidth(2);

    H0->Draw("HIST");
    H1->Draw("HIST SAME");
    H01->Draw("E1 SAME");

    can->SaveAs("imgs/weights.png");


    auto tree2 = (TTree*)inFile->Get("mu_vals");
    const int nevents = tree2->GetEntries();

    double mu_inits[1];
    float mu_fits;

    tree2->SetBranchAddress("mu_fits", &mu_fits);
    tree2->SetBranchAddress("mu_inits", mu_inits);

    auto hmu = new TH2D("hmu", "; #mu_{init} [a.u.]; #mu_{fits} [a.u.]", 20, -2., 2., 20, -1.25, -1.15);

    double mu_vals[nevents];

    for(int i = 0; i < nevents; i++)
    {
        tree2->GetEntry(i);
        hmu->Fill(mu_inits[0], mu_fits);
        //cout << mu_inits[0] << endl;
        mu_vals[i] = mu_fits;
    }

    double mu_fit_mean = TMath::Mean(nevents, mu_vals);
    double mu_fit_std = TMath::RMS(nevents, mu_vals);

    TString mu_results = Form("mu = %.3f +/- %.3f", mu_fit_mean, mu_fit_std);

    cout << mu_results.Data() << endl;

    hmu->Draw("COLZ");

    TText* t = new TText(.2, .95, mu_results.Data());
    t->SetNDC();
    t->SetTextAlign(22);
    t->SetTextColor(kRed+2);
    t->SetTextFont(43);
    t->SetTextSize(20);
    t->Draw();
    //can->Update();
    can->SaveAs("imgs/mu_2D.png");
}
