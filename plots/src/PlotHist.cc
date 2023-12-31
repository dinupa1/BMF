/*
* dinupa3@gmail.com
*/

#include<TFile.h>
#include<TTree.h>
#include<TCanvas.h>
#include<TH1D.h>
#include<TH2D.h>
#include<TString.h>
#include<iostream>

#include "../include/PlotHist.hh"

using namespace std;

PlotHist::PlotHist()
{
	TFile* inFile = TFile::Open("results.root", "read");
	inTree = (TTree*)inFile->Get("save");
	nevents = inTree->GetEntries();

	inTree->SetBranchAddress("X_par",		X_par);
	inTree->SetBranchAddress("X_pred_mu",	X_pred_mu);
    inTree->SetBranchAddress("X_pred_std",	X_pred_std);
}

void PlotHist::DrawResolution()
{
    TCanvas* can = new TCanvas();

    TH1D* lambda_hist = new TH1D("lambda_hist", "; #lambda_{injected} - #lambda_{prediction} [a.u.]; counts [a.u.]", 20, -2., 2.);
    TH1D* mu_hist = new TH1D("mu_hist", "; #mu_{injected} - #mu_{prediction} [a.u.]; counts [a.u.]", 20, -1., 1.);
    TH1D* nu_hist = new TH1D("nu_hist", "; #nu_{injected} - #nu_{prediction} [a.u.]; counts [a.u.]", 20, -1., 1.);

    TH2D* delta_lambda = new TH2D("delta_lambda", "; #lambda_{injected} [a.u.]; #lambda_{prediction} [a.u.]", 20, -1., 1., 20, -1., 1.);
    TH2D* delta_mu = new TH2D("delta_mu", "; #mu_{injected} [a.u.]; #mu_{prediction} [a.u.]", 20, -0.5, 0.5, 20, -0.5, 0.5);
    TH2D* delta_nu = new TH2D("delta_nu", "; #nu_{injected} [a.u.]; #nu_{prediction} [a.u.]", 20, -0.5, 0.5, 20, -0.5, 0.5);

    for(int i = 0; i < nevents; i++)
    {
        inTree->GetEntry(i);
        
        lambda_hist->Fill(X_par[0] - X_pred_mu[0]);
        mu_hist->Fill(X_par[1] - X_pred_mu[1]);
        nu_hist->Fill(X_par[2] - X_pred_mu[2]);

        delta_lambda->Fill(X_par[0], X_pred_mu[0]);
        delta_mu->Fill(X_par[1], X_pred_mu[1]);
        delta_nu->Fill(X_par[2], X_pred_mu[2]);
    }

    lambda_hist->SetMarkerStyle(8);
    lambda_hist->SetMarkerColor(4);

    lambda_hist->Draw("E1");
    can->SaveAs("imgs/delta_lambda.png");

    mu_hist->SetMarkerStyle(8);
    mu_hist->SetMarkerColor(4);

    mu_hist->Draw("E1");
    can->SaveAs("imgs/delta_mu.png");

    nu_hist->SetMarkerStyle(8);
    nu_hist->SetMarkerColor(4);

    nu_hist->Draw("E1");
    can->SaveAs("imgs/delta_nu.png");

    cout << "lambda covariance : " << delta_lambda->GetCovariance() << endl;
    delta_lambda->Draw("COLZ");
    can->SaveAs("imgs/cov_lambda.png");

    cout << "mu covariance : " << delta_mu->GetCovariance() << endl;
    delta_mu->Draw("COLZ");
    can->SaveAs("imgs/cov_mu.png");

    cout << "nu covariance : " << delta_nu->GetCovariance() << endl;
    delta_nu->Draw("COLZ");
    can->SaveAs("imgs/cov_nu.png");
}

void PlotHist::Print()
{
    for(int i = 0; i < 5; i++)
    {
        inTree->GetEntry(i);
        cout << X_par[0] << " -> " << X_pred_mu[0] << " +/- " << X_pred_std[0] << endl;
        cout << X_par[1] << " -> " << X_pred_mu[1] << " +/- " << X_pred_std[1] << endl;
        cout << X_par[2] << " -> " << X_pred_mu[2] << " +/- " << X_pred_std[2] << endl;
        cout << " *** " << endl;
    }
}