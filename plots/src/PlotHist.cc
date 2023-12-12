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
	TFile* file = TFile::Open("results.root", "read");
	tree = (TTree*)file->Get("save");
	nevents = tree->GetEntries();

	tree->SetBranchAddress("X_par",		X_par);
	tree->SetBranchAddress("X_pred",	X_pred);
}

void PlotHist::DrawResolution()
{
	TCanvas* can = new TCanvas();

	TH1D* lambda_hist[4];
	TH1D* mu_hist[4];
	TH1D* nu_hist[4];


	TH2D* delta_lambda[4];
	TH2D* delta_mu[4];
	TH2D* delta_nu[4];

	for(int i = 0; i < 4; i++)
	{
		TString lambda_name = Form("lambda_pT_%d", i);
		lambda_hist[i] = new TH1D(lambda_name.Data(), "; #lambda_{injected} - #lambda_{predicted} [a.u.]; counts [a.u.]", 50, -2., 2.);

		TString mu_name = Form("mu_pT_%d", i);
		mu_hist[i] = new TH1D(mu_name.Data(), "; #mu_{injected} - #mu_{predicted} [a.u.]; counts [a.u.]", 50, -1., 1.);

		TString nu_name = Form("nu_pT_%d", i);
		nu_hist[i] = new TH1D(nu_name.Data(), "; #nu_{injected} - #nu_{predicted} [a.u.]; counts [a.u.]", 50, -1., 1.);

		TString delta_lambda_name = Form("delta_lambda_pT_%d", i);
		delta_lambda[i] = new TH2D(delta_lambda_name.Data(), "; #lambda_{injected} [a.u.]; #lambda_{injected} - #lambda_{predicted}", 50, -1., 1., 50, -2., 2.);

		TString delta_mu_name = Form("delta_mu_pT_%d", i);
		delta_mu[i] = new TH2D(delta_mu_name.Data(), "; #mu_{injected} [a.u.]; #mu_{injected} - #mu_{predicted}", 50, -0.5, 0.5, 50, -1., 1.);

		TString delta_nu_name = Form("delta_nu_pT_%d", i);
		delta_nu[i] = new TH2D(delta_nu_name.Data(), "; #nu_{injected} [a.u.]; #nu_{injected} - #nu_{predicted}", 50, -0.5, 0.5, 50, -1., 1.);

		for(int j = 0; j < nevents; j++)
		{
			tree->GetEntry(j);

			lambda_hist[i]->Fill(X_par[i][0] - X_pred[i][0]);
			mu_hist[i]->Fill(X_par[i][1] - X_pred[i][1]);
			nu_hist[i]->Fill(X_par[i][2] - X_pred[i][2]);

			delta_lambda[i]->Fill(X_par[i][0], X_par[i][0] - X_pred[i][0]);
			delta_mu[i]->Fill(X_par[i][1], X_par[i][1] - X_pred[i][1]);
			delta_nu[i]->Fill(X_par[i][2], X_par[i][2] - X_pred[i][2]);
		}

		lambda_hist[i]->Draw("HIST");
		TString lambda_can = Form("imgs/lambda_pT_%d.png", i);
		can->SaveAs(lambda_can.Data());

		mu_hist[i]->Draw("HIST");
		TString mu_can = Form("imgs/mu_pT_%d.png", i);
		can->SaveAs(mu_can.Data());

		nu_hist[i]->Draw("HIST");
		TString nu_can = Form("imgs/nu_pT_%d.png", i);
		can->SaveAs(nu_can.Data());


		delta_lambda[i]->Draw("COLZ");
		TString delta_lambda_can = Form("imgs/delta_lambda_pT_%d.png", i);
		can->SaveAs(delta_lambda_can.Data());

		delta_mu[i]->Draw("COLZ");
		TString delta_mu_can = Form("imgs/delta_mu_pT_%d.png", i);
		can->SaveAs(delta_mu_can.Data());

		delta_nu[i]->Draw("COLZ");
		TString delta_nu_can = Form("imgs/delta_nu_pT_%d.png", i);
		can->SaveAs(delta_nu_can.Data());
	}
}