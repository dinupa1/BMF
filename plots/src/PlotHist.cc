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
	tree = (TTree*)inFile->Get("save");
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
		lambda_hist[i] = new TH1D(lambda_name.Data(), "; #lambda_{injected} - #lambda_{predicted} [a.u.]; counts [a.u.]", 20, -1.5, 1.5);

		TString mu_name = Form("mu_pT_%d", i);
		mu_hist[i] = new TH1D(mu_name.Data(), "; #mu_{injected} - #mu_{predicted} [a.u.]; counts [a.u.]", 20, -0.6, 0.6);

		TString nu_name = Form("nu_pT_%d", i);
		nu_hist[i] = new TH1D(nu_name.Data(), "; #nu_{injected} - #nu_{predicted} [a.u.]; counts [a.u.]", 20, -0.8, 0.8);

		TString delta_lambda_name = Form("delta_lambda_pT_%d", i);
		delta_lambda[i] = new TH2D(delta_lambda_name.Data(), "; #lambda_{injected} [a.u.]; #lambda_{injected} - #lambda_{predicted}", 20, -1., 1., 20, -1.5, 1.5);

		TString delta_mu_name = Form("delta_mu_pT_%d", i);
		delta_mu[i] = new TH2D(delta_mu_name.Data(), "; #mu_{injected} [a.u.]; #mu_{injected} - #mu_{predicted}", 20, -0.5, 0.5, 20, -0.6, 0.6);

		TString delta_nu_name = Form("delta_nu_pT_%d", i);
		delta_nu[i] = new TH2D(delta_nu_name.Data(), "; #nu_{injected} [a.u.]; #nu_{injected} - #nu_{predicted}", 20, -0.5, 0.5, 20, -0.8, 0.8);

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

void PlotHist::DrawHist()
{
	TFile* inFile1 = TFile::Open("examples.root", "read");
	TTree* tree1 = (TTree*)inFile1->Get("save2");
	int nexamples = tree1->GetEntries();

	float X_par_mean[4][3];
	float X_par_std[4][3];
	float X_pred_mean[4][3];
	float X_pred_std[4][3];

	tree1->SetBranchAddress("X_par_mean", X_par_mean);
	tree1->SetBranchAddress("X_par_std", X_par_std);
	tree1->SetBranchAddress("X_pred_mean", X_pred_mean);
	tree1->SetBranchAddress("X_pred_std", X_pred_std);

	TH1D* lambda_pT_par[5];
	TH1D* mu_pT_par[5];
	TH1D* nu_pT_par[5];

	TH1D* lambda_pT_pred[5];
	TH1D* mu_pT_pred[5];
	TH1D* nu_pT_pred[5];

	TCanvas* c1 = new TCanvas();

	for(int i = 0; i < nexamples; i++)
	{
		tree1->GetEntry(i);

		TString lambda_name1 = Form("lambda_pT_par_%d", i);
		lambda_pT_par[i] = new TH1D(lambda_name1.Data(), "; p_{T} [GeV]; #lambda [a.u.]", 4, pT_edges);

		TString mu_name1 = Form("mu_pT_par_%d", i);
		mu_pT_par[i] = new TH1D(mu_name1.Data(), "; p_{T} [GeV]; #mu [a.u.]", 4, pT_edges);

		TString nu_name1 = Form("nu_pT_par_%d", i);
		nu_pT_par[i] = new TH1D(nu_name1.Data(), "; p_{T} [GeV]; #nu [a.u.]", 4, pT_edges);

		TString lambda_name2 = Form("lambda_pT_pred_%d", i);
		lambda_pT_pred[i] = new TH1D(lambda_name2.Data(), "; p_{T} [GeV]; #lambda [a.u.]", 4, pT_edges);

		TString mu_name2 = Form("mu_pT_pred_%d", i);
		mu_pT_pred[i] = new TH1D(mu_name2.Data(), "; p_{T} [GeV]; #mu [a.u.]", 4, pT_edges);

		TString nu_name2 = Form("nu_pT_pred_%d", i);
		nu_pT_pred[i] = new TH1D(nu_name2.Data(), "; p_{T} [GeV]; #nu [a.u.]", 4, pT_edges);


		for(int j = 0; j < 4; j++)
		{
			lambda_pT_par[i]->SetBinContent(j+1, X_par_mean[j][0]);
			lambda_pT_par[i]->SetBinError(j+1, X_par_std[j][0]);

			mu_pT_par[i]->SetBinContent(j+1, X_par_mean[j][1]);
			mu_pT_par[i]->SetBinError(j+1, X_par_std[j][1]);

			nu_pT_par[i]->SetBinContent(j+1, X_par_mean[j][2]);
			nu_pT_par[i]->SetBinError(j+1, X_par_std[j][2]);


			lambda_pT_pred[i]->SetBinContent(j+1, X_pred_mean[j][0]);
			lambda_pT_pred[i]->SetBinError(j+1, X_pred_std[j][0]);

			mu_pT_pred[i]->SetBinContent(j+1, X_pred_mean[j][1]);
			mu_pT_pred[i]->SetBinError(j+1, X_pred_std[j][1]);

			nu_pT_pred[i]->SetBinContent(j+1, X_pred_mean[j][2]);
			nu_pT_pred[i]->SetBinError(j+1, X_pred_std[j][2]);

			//cout << X_par_mean[j][1] << " " << X_pred_mean[j][1] << " +/- " << X_pred_std[j][1] << endl;
		}

		lambda_pT_par[i]->SetMarkerStyle(8);
		lambda_pT_par[i]->SetMarkerColor(2);

		lambda_pT_pred[i]->SetMarkerStyle(21);
		lambda_pT_pred[i]->SetMarkerColor(4);

		lambda_pT_par[i]->Draw("E1");
		lambda_pT_pred[i]->Draw("SAME");

		TString can_name_lambda = Form("imgs/pred_lambda_pT_%d.png", i);
		c1->SaveAs(can_name_lambda.Data());


		mu_pT_par[i]->SetMarkerStyle(8);
		mu_pT_par[i]->SetMarkerColor(2);

		mu_pT_pred[i]->SetMarkerStyle(21);
		mu_pT_pred[i]->SetMarkerColor(4);

		mu_pT_par[i]->Draw("E1");
		mu_pT_pred[i]->Draw("SAME");

		TString can_name_mu = Form("imgs/pred_mu_pT_%d.png", i);
		c1->SaveAs(can_name_mu.Data());


		nu_pT_par[i]->SetMarkerStyle(8);
		nu_pT_par[i]->SetMarkerColor(2);

		nu_pT_pred[i]->SetMarkerStyle(21);
		nu_pT_pred[i]->SetMarkerColor(4);

		nu_pT_par[i]->Draw("E1");
		nu_pT_pred[i]->Draw("SAME");

		TString can_name_nu = Form("imgs/pred_nu_pT_%d.png", i);
		c1->SaveAs(can_name_nu.Data());
	}
}