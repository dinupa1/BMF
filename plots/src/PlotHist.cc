/*
* dinupa3@gmail.com
*/

#include<TFile.h>
#include<TTree.h>
#include<TCanvas.h>
#include<TH1D.h>
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
	tree->SetBranchAddress("X_preds",	X_preds);
}

void PlotHist::DeltaHist(int ii)
{
	TString lambda_par_name = Form("lambda_par_%d", ii);
	TH3D* lambda_par = new TH3D(lambda_par_name.Data(), "", 3, mass_edges, 3, pT_edges, 3, xF_edges);

	TString mu_par_name = Form("mu_par_%d", ii);
	TH3D* mu_par = new TH3D(mu_par_name.Data(), "", 3, mass_edges, 3, pT_edges, 3, xF_edges);

	TString nu_par_name = Form("nu_par_%d", ii);
	TH3D* nu_par = new TH3D(nu_par_name.Data(), "", 3, mass_edges, 3, pT_edges, 3, xF_edges);

	TString lambda_preds_name = Form("lambda_preds_%d", ii);
	TH3D* lambda_preds = new TH3D(lambda_preds_name.Data(), "", 3, mass_edges, 3, pT_edges, 3, xF_edges);

	TString mu_preds_name = Form("mu_preds_%d", ii);
	TH3D* mu_preds = new TH3D(mu_preds_name.Data(), "", 3, mass_edges, 3, pT_edges, 3, xF_edges);

	TString nu_preds_name = Form("nu_preds_%d", ii);
	TH3D* nu_preds = new TH3D(nu_preds_name.Data(), "", 3, mass_edges, 3, pT_edges, 3, xF_edges);

	for(int i = 0; i < 3; i++)
	{
		for(int j = 0; j < 3; j++)
		{
			for(int k = 0; k < 3; k++)
			{
				lambda_par->SetBinContent(i+1, j+1, k+1, X_par[i][j][k]);
				mu_par->SetBinContent(i+1, j+1, k+1, X_par[3+i][j][k]);
				nu_par->SetBinContent(i+1, j+1, k+1, X_par[6+i][j][k]);

				lambda_preds->SetBinContent(i+1, j+1, k+1, X_preds[i][j][k]);
				mu_preds->SetBinContent(i+1, j+1, k+1, X_preds[3+i][j][k]);
				nu_preds->SetBinContent(i+1, j+1, k+1, X_preds[6+i][j][k]);
			}
		}
	}

	/*
	*/

	TH3D* delta_lambda = (TH3D*)lambda_par->Clone();
	TH1D* delta_lambda_mass = (TH1D*)delta_lambda->ProjectionX();
	TH1D* delta_lambda_pT = (TH1D*)delta_lambda->ProjectionY();
	TH1D* delta_lambda_xF = (TH1D*)delta_lambda->ProjectionZ();

	TH3D* delta_lambda_preds = (TH3D*)lambda_preds->Clone();
	TH1D* delta_lambda_mass_preds = (TH1D*)delta_lambda_preds->ProjectionX();
	TH1D* delta_lambda_pT_preds = (TH1D*)delta_lambda_preds->ProjectionY();
	TH1D* delta_lambda_xF_preds = (TH1D*)delta_lambda_preds->ProjectionZ();

	delta_lambda_mass->Add(delta_lambda_mass_preds, -1);
	delta_lambda_pT->Add(delta_lambda_pT_preds, -1);
	delta_lambda_xF->Add(delta_lambda_xF_preds, -1);

	/*
	*/

	TH3D* delta_mu = (TH3D*)mu_par->Clone();
	TH1D* delta_mu_mass = (TH1D*)delta_mu->ProjectionX();
	TH1D* delta_mu_pT = (TH1D*)delta_mu->ProjectionY();
	TH1D* delta_mu_xF = (TH1D*)delta_mu->ProjectionZ();

	TH3D* delta_mu_preds = (TH3D*)mu_preds->Clone();
	TH1D* delta_mu_mass_preds = (TH1D*)delta_mu_preds->ProjectionX();
	TH1D* delta_mu_pT_preds = (TH1D*)delta_mu_preds->ProjectionY();
	TH1D* delta_mu_xF_preds = (TH1D*)delta_mu_preds->ProjectionZ();

	delta_mu_mass->Add(delta_mu_mass_preds, -1);
	delta_mu_pT->Add(delta_mu_pT_preds, -1);
	delta_mu_xF->Add(delta_mu_xF_preds, -1);

	/*
	*/

	TH3D* delta_nu = (TH3D*)nu_par->Clone();
	TH1D* delta_nu_mass = (TH1D*)delta_nu->ProjectionX();
	TH1D* delta_nu_pT = (TH1D*)delta_nu->ProjectionY();
	TH1D* delta_nu_xF = (TH1D*)delta_nu->ProjectionZ();

	TH3D* delta_nu_preds = (TH3D*)nu_preds->Clone();
	TH1D* delta_nu_mass_preds = (TH1D*)delta_nu_preds->ProjectionX();
	TH1D* delta_nu_pT_preds = (TH1D*)delta_nu_preds->ProjectionY();
	TH1D* delta_nu_xF_preds = (TH1D*)delta_nu_preds->ProjectionZ();

	delta_nu_mass->Add(delta_nu_mass_preds, -1);
	delta_nu_pT->Add(delta_nu_pT_preds, -1);
	delta_nu_xF->Add(delta_nu_xF_preds, -1);

	for(int i = 0; i < 3; i++)
	{
		lambda_array[0][i] = delta_lambda_mass->GetBinContent(i+1);
		lambda_array[1][i] = delta_lambda_pT->GetBinContent(i+1);
		lambda_array[2][i] = delta_lambda_xF->GetBinContent(i+1);

		mu_array[0][i] = delta_mu_mass->GetBinContent(i+1);
		mu_array[1][i] = delta_mu_pT->GetBinContent(i+1);
		mu_array[2][i] = delta_mu_xF->GetBinContent(i+1);

		nu_array[0][i] = delta_nu_mass->GetBinContent(i+1);
		nu_array[1][i] = delta_nu_pT->GetBinContent(i+1);
		nu_array[2][i] = delta_nu_xF->GetBinContent(i+1);
	}
}

void PlotHist::DrawResolution()
{
	for(int i = 0; i < 3; i++)
	{
		TString lambda_mass_name = Form("lambda_mass_%d", i);
		lambda_mass[i] = new TH1D(lambda_mass_name.Data(), "mass ; #Delta #lambda [a.u.]; counts [a.u.]", 20, -8., 8.);

		TString lambda_pT_name = Form("lambda_pT_%d", i);
		lambda_pT[i] = new TH1D(lambda_pT_name.Data(), "p_{T}; #Delta #lambda [a.u.]; counts [a.u.]", 20, -8., 8.);

		TString lambda_xF_name = Form("lambda_xF_%d", i);
		lambda_xF[i] = new TH1D(lambda_xF_name.Data(), "x_{F}; #Delta #lambda [a.u.]; counts [a.u.]", 20, -8., 8.);

		TString mu_mass_name = Form("mu_mass_%d", i);
		mu_mass[i] = new TH1D(mu_mass_name.Data(), "mass; #Delta #mu [a.u.]; counts [a.u.]", 20, -4., 4.);

		TString mu_pT_name = Form("mu_pT_%d", i);
		mu_pT[i] = new TH1D(mu_pT_name.Data(), "pT; #Delta #mu [a.u.]; counts [a.u.]", 20, -4., 4.);

		TString mu_xF_name = Form("mu_xF_%d", i);
		mu_xF[i] = new TH1D(mu_xF_name.Data(), "xF; #Delta #mu [a.u.]; counts [a.u.]", 20, -4., 4.);

		TString nu_mass_name = Form("nu_mass_%d", i);
		nu_mass[i] = new TH1D(nu_mass_name.Data(), "mass; #Delta #nu [a.u.]; counts [a.u.]", 20, -4., 4.);

		TString nu_pT_name = Form("nu_pT_%d", i);
		nu_pT[i] = new TH1D(nu_pT_name.Data(), "pT; #Delta #nu [a.u.]; counts [a.u.]", 20, -4., 4.);

		TString nu_xF_name = Form("nu_xF_%d", i);
		nu_xF[i] = new TH1D(nu_xF_name.Data(), "xF; #Delta #nu [a.u.]; counts [a.u.]", 20, -4., 4.);
	}

	for(int i = 0; i < nevents; i++)
	{
		tree->GetEntry(i);
		DeltaHist(i);

		for(int j = 0; j < 3; j++)
		{
			lambda_mass[j]->Fill(lambda_array[0][j]);
			lambda_pT[j]->Fill(lambda_array[1][j]);
			lambda_xF[j]->Fill(lambda_array[2][j]);

			mu_mass[j]->Fill(mu_array[0][j]);
			mu_pT[j]->Fill(mu_array[1][j]);
			mu_xF[j]->Fill(mu_array[2][j]);

			nu_mass[j]->Fill(nu_array[0][j]);
			nu_pT[j]->Fill(nu_array[1][j]);
			nu_xF[j]->Fill(nu_array[2][j]);
		}
	}

	TCanvas* can = new TCanvas();

	for(int i = 0; i < 3; i++)
	{
		TString lambda_mass_can = Form("imgs/lambda_mass_%d.png", i);
		lambda_mass[i]->Draw("E1");
		can->SaveAs(lambda_mass_can.Data());

		TString lambda_pT_can = Form("imgs/lambda_pT_%d.png", i);
		lambda_pT[i]->Draw("E1");
		can->SaveAs(lambda_pT_can.Data());

		TString lambda_xF_can = Form("imgs/lambda_xF_%d.png", i);
		lambda_xF[i]->Draw("E1");
		can->SaveAs(lambda_xF_can.Data());

		/*
		*/

		TString mu_mass_can = Form("imgs/mu_mass_%d.png", i);
		mu_mass[i]->Draw("E1");
		can->SaveAs(mu_mass_can.Data());

		TString mu_pT_can = Form("imgs/mu_pT_%d.png", i);
		mu_pT[i]->Draw("E1");
		can->SaveAs(mu_pT_can.Data());

		TString mu_xF_can = Form("imgs/mu_xF_%d.png", i);
		mu_xF[i]->Draw("E1");
		can->SaveAs(mu_xF_can.Data());

		/*
		*/
		TString nu_mass_can = Form("imgs/nu_mass_%d.png", i);
		nu_mass[i]->Draw("E1");
		can->SaveAs(nu_mass_can.Data());

		TString nu_pT_can = Form("imgs/nu_pT_%d.png", i);
		nu_pT[i]->Draw("E1");
		can->SaveAs(nu_pT_can.Data());

		TString nu_xF_can = Form("imgs/nu_xF_%d.png", i);
		nu_xF[i]->Draw("E1");
		can->SaveAs(nu_xF_can.Data());
	}
}