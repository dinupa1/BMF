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

void PlotHist::DrawResolution(int theta_dim, TString hist_title, double x_min, double x_max)
{
	TCanvas* can = new TCanvas();

	for(int i = 0; i < 3; i++)
	{
		for(int j = 0; j < 3; j++)
		{
			TString hist_name = Form("hist_%d_%d_%d", theta_dim, i, j);
			TH1D* hist = new TH1D(hist_name.Data(), hist_title.Data(), 20, x_max, x_min);

			for(int ii = 0; ii < nevents; ii++)
			{
				tree->GetEntry(ii);
				hist->Fill(X_par[theta_dim][i][j] - X_preds[theta_dim][i][j]);
			}

			TString can_name = Form("imgs/hist_%d_%d_%d.png", theta_dim, i, j);
			hist->Draw("E1");
			can->SaveAs(can_name.Data());
		}
	}
}

void PlotHist::DrawHist()
{
	const int nhist = 5;

	/*
	* injected
	*/

	TH1D* lambda_mass_par[nhist];
	TH1D* mu_mass_par[nhist];
	TH1D* nu_mass_par[nhist];

	TH1D* lambda_pT_par[nhist];
	TH1D* mu_pT_par[nhist];
	TH1D* nu_pT_par[nhist];

	TH1D* lambda_xF_par[nhist];
	TH1D* mu_xF_par[nhist];
	TH1D* nu_xF_par[nhist];

	/*
	* prediction
	*/

	TH1D* lambda_mass_preds[nhist];
	TH1D* mu_mass_preds[nhist];
	TH1D* nu_mass_preds[nhist];

	TH1D* lambda_pT_preds[nhist];
	TH1D* mu_pT_preds[nhist];
	TH1D* nu_pT_preds[nhist];

	TH1D* lambda_xF_preds[nhist];
	TH1D* mu_xF_preds[nhist];
	TH1D* nu_xF_preds[nhist];

	double mass_edges[4] = {4., 5.5, 6.5, 9.};
	double pT_edges[4] = {0., 0.5, 1., 2.5};
	double xF_edges[4] = {-0.1, 0.3, 0.5, 1.0};

	/*
	* fill histogram
	*/

	for(int i = 0; i < nhist; i++)
	{
		tree->GetEntry(i);

		/*
		* injected
		*/

		/*
		* mass
		*/

		TString lambda_mass_par_name = Form("lambda_mass_par_%d", i);
		lambda_mass_par[i] = new TH1D(lambda_mass_par_name.Data(), "; mass [GeV]; #lambda", 3, mass_edges);

		TString mu_mass_par_name = Form("mu_mass_par_%d", i);
		mu_mass_par[i] = new TH1D(mu_mass_par_name.Data(), "; mass [GeV]; #mu", 3, mass_edges);

		TString nu_mass_par_name = Form("nu_mass_par_%d", i);
		nu_mass_par[i] = new TH1D(nu_mass_par_name.Data(), "; mass [GeV]; #nu", 3, mass_edges);

		/*
		* pT
		*/

		TString lambda_pT_par_name = Form("lambda_pT_par_%d", i);
		lambda_pT_par[i] = new TH1D(lambda_pT_par_name.Data(), "; pT [GeV]; #lambda", 3, pT_edges);

		TString mu_pT_par_name = Form("mu_pT_par_%d", i);
		mu_pT_par[i] = new TH1D(mu_pT_par_name.Data(), "; pT [GeV]; #mu", 3, pT_edges);

		TString nu_pT_par_name = Form("nu_pT_par_%d", i);
		nu_pT_par[i] = new TH1D(nu_pT_par_name.Data(), "; pT [GeV]; #nu", 3, pT_edges);

		/*
		* xF
		*/

		TString lambda_xF_par_name = Form("lambda_xF_par_%d", i);
		lambda_xF_par[i] = new TH1D(lambda_xF_par_name.Data(), "; xF; #lambda", 3, xF_edges);

		TString mu_xF_par_name = Form("mu_xF_par_%d", i);
		mu_xF_par[i] = new TH1D(mu_xF_par_name.Data(), "; xF; #mu", 3, xF_edges);

		TString nu_xF_par_name = Form("nu_xF_par_%d", i);
		nu_xF_par[i] = new TH1D(nu_xF_par_name.Data(), "; xF; #nu", 3, xF_edges);


		/*
		* predicted
		*/

		/*
		* mass
		*/

		TString lambda_mass_preds_name = Form("lambda_mass_preds_%d", i);
		lambda_mass_preds[i] = new TH1D(lambda_mass_preds_name.Data(), "; mass [GeV]; #lambda", 3, mass_edges);

		TString mu_mass_preds_name = Form("mu_mass_preds_%d", i);
		mu_mass_preds[i] = new TH1D(mu_mass_preds_name.Data(), "; mass [GeV]; #mu", 3, mass_edges);

		TString nu_mass_preds_name = Form("nu_mass_prreds_%d", i);
		nu_mass_preds[i] = new TH1D(nu_mass_preds_name.Data(), "; mass [GeV]; #nu", 3, mass_edges);

		/*
		* pT
		*/

		TString lambda_pT_preds_name = Form("lambda_pT_preds_%d", i);
		lambda_pT_preds[i] = new TH1D(lambda_pT_preds_name.Data(), "; pT [GeV]; #lambda", 3, pT_edges);

		TString mu_pT_preds_name = Form("mu_pT_preds_%d", i);
		mu_pT_preds[i] = new TH1D(mu_pT_preds_name.Data(), "; pT [GeV]; #mu", 3, pT_edges);

		TString nu_pT_preds_name = Form("nu_pT_preds_%d", i);
		nu_pT_preds[i] = new TH1D(nu_pT_preds_name.Data(), "; pT [GeV]; #nu", 3, pT_edges);

		/*
		* xF
		*/

		TString lambda_xF_preds_name = Form("lambda_xF_preds_%d", i);
		lambda_xF_preds[i] = new TH1D(lambda_xF_preds_name.Data(), "; xF; #lambda", 3, xF_edges);

		TString mu_xF_preds_name = Form("mu_xF_preds_%d", i);
		mu_xF_preds[i] = new TH1D(mu_xF_preds_name.Data(), "; xF; #mu", 3, xF_edges);

		TString nu_xF_preds_name = Form("nu_xF_preds_%d", i);
		nu_xF_preds[i] = new TH1D(nu_xF_preds_name.Data(), "; xF; #nu", 3, xF_edges);

		for(int j = 0; j < 3; j++)
		{
			lambda_mass_par[i]->SetBinContent(j+1, X_par[0][0][j]);
			lambda_mass_par[i]->SetBinError(j+1, X_par[1][0][j]);

			lambda_pT_par[i]->SetBinContent(j+1, X_par[0][1][j]);
			lambda_pT_par[i]->SetBinError(j+1, X_par[1][1][j]);

			lambda_xF_par[i]->SetBinContent(j+1, X_par[0][2][j]);
			lambda_xF_par[i]->SetBinError(j+1, X_par[1][2][j]);

			mu_mass_par[i]->SetBinContent(j+1, X_par[2][0][j]);
			mu_mass_par[i]->SetBinError(j+1, X_par[3][0][j]);

			mu_pT_par[i]->SetBinContent(j+1, X_par[2][1][j]);
			mu_pT_par[i]->SetBinError(j+1, X_par[3][1][j]);

			mu_xF_par[i]->SetBinContent(j+1, X_par[2][2][j]);
			mu_xF_par[i]->SetBinError(j+1, X_par[3][2][j]);


			nu_mass_par[i]->SetBinContent(j+1, X_par[4][0][j]);
			nu_mass_par[i]->SetBinError(j+1, X_par[5][0][j]);

			nu_pT_par[i]->SetBinContent(j+1, X_par[4][1][j]);
			nu_pT_par[i]->SetBinError(j+1, X_par[5][1][j]);

			nu_xF_par[i]->SetBinContent(j+1, X_par[4][2][j]);
			nu_xF_par[i]->SetBinError(j+1, X_par[5][2][j]);


			lambda_mass_preds[i]->SetBinContent(j+1, X_preds[0][0][j]);
			lambda_mass_preds[i]->SetBinError(j+1, X_preds[1][0][j]);

			lambda_pT_preds[i]->SetBinContent(j+1, X_preds[0][1][j]);
			lambda_pT_preds[i]->SetBinError(j+1, X_preds[1][1][j]);

			lambda_xF_preds[i]->SetBinContent(j+1, X_preds[0][2][j]);
			lambda_xF_preds[i]->SetBinError(j+1, X_preds[1][2][j]);

			mu_mass_preds[i]->SetBinContent(j+1, X_preds[2][0][j]);
			mu_mass_preds[i]->SetBinError(j+1, X_preds[3][0][j]);

			mu_pT_preds[i]->SetBinContent(j+1, X_preds[2][1][j]);
			mu_pT_preds[i]->SetBinError(j+1, X_preds[3][1][j]);

			mu_xF_preds[i]->SetBinContent(j+1, X_preds[2][2][j]);
			mu_xF_preds[i]->SetBinError(j+1, X_preds[3][2][j]);


			nu_mass_preds[i]->SetBinContent(j+1, X_preds[4][0][j]);
			nu_mass_preds[i]->SetBinError(j+1, X_preds[5][0][j]);

			nu_pT_preds[i]->SetBinContent(j+1, X_preds[4][1][j]);
			nu_pT_preds[i]->SetBinError(j+1, X_preds[5][1][j]);

			nu_xF_preds[i]->SetBinContent(j+1, X_preds[4][2][j]);
			nu_xF_preds[i]->SetBinError(j+1, X_preds[5][2][j]);
		}
	}

	/*
	* draw the histograms
	*/

	TCanvas* can = new TCanvas();

	for(int i = 0; i < nhist; i++)
	{
		lambda_mass_par[i]->SetMaximum(5.);
		lambda_mass_par[i]->SetMinimum(-5.);

		lambda_mass_par[i]->SetMarkerStyle(8);
		lambda_mass_par[i]->SetMarkerColor(2);

		lambda_mass_preds[i]->SetMaximum(5.);
		lambda_mass_preds[i]->SetMinimum(-5.);

		lambda_mass_preds[i]->SetMarkerStyle(21);
		lambda_mass_preds[i]->SetMarkerColor(4);

		lambda_mass_par[i]->Draw("E1");
		lambda_mass_preds[i]->Draw("SAME");

		TString can_lambda_mass = Form("imgs/lambda_mass_%d.png", i);
		can->SaveAs(can_lambda_mass.Data());

		lambda_pT_par[i]->SetMaximum(5.);
		lambda_pT_par[i]->SetMinimum(-5.);

		lambda_pT_par[i]->SetMarkerStyle(8);
		lambda_pT_par[i]->SetMarkerColor(2); // red

		lambda_pT_preds[i]->SetMaximum(5.);
		lambda_pT_preds[i]->SetMinimum(-5.);

		lambda_pT_preds[i]->SetMarkerStyle(21);
		lambda_pT_preds[i]->SetMarkerColor(4); // blue

		lambda_pT_par[i]->Draw("E1");
		lambda_pT_preds[i]->Draw("SAME");

		TString can_lambda_pT = Form("imgs/lambda_pT_%d.png", i);
		can->SaveAs(can_lambda_pT.Data());


		lambda_xF_par[i]->SetMaximum(5.);
		lambda_xF_par[i]->SetMinimum(-5.);

		lambda_xF_par[i]->SetMarkerStyle(8);
		lambda_xF_par[i]->SetMarkerColor(2);

		lambda_xF_preds[i]->SetMaximum(5.);
		lambda_xF_preds[i]->SetMinimum(-5.);

		lambda_xF_preds[i]->SetMarkerStyle(21);
		lambda_xF_preds[i]->SetMarkerColor(4);

		lambda_xF_par[i]->Draw("E1");
		lambda_xF_preds[i]->Draw("SAME");

		TString can_lambda_xF = Form("imgs/lambda_xF_%d.png", i);
		can->SaveAs(can_lambda_xF.Data());

		/*
		*/

		mu_mass_par[i]->SetMaximum(5.);
		mu_mass_par[i]->SetMinimum(-5.);

		mu_mass_par[i]->SetMarkerStyle(8);
		mu_mass_par[i]->SetMarkerColor(2);

		mu_mass_preds[i]->SetMaximum(5.);
		mu_mass_preds[i]->SetMinimum(-5.);

		mu_mass_preds[i]->SetMarkerStyle(21);
		mu_mass_preds[i]->SetMarkerColor(4);

		mu_mass_par[i]->Draw("E1");
		mu_mass_preds[i]->Draw("SAME");

		TString can_mu_mass = Form("imgs/mu_mass_%d.png", i);
		can->SaveAs(can_mu_mass.Data());

		mu_pT_par[i]->SetMaximum(5.);
		mu_pT_par[i]->SetMinimum(-5.);

		mu_pT_par[i]->SetMarkerStyle(8);
		mu_pT_par[i]->SetMarkerColor(2);

		mu_pT_preds[i]->SetMaximum(5.);
		mu_pT_preds[i]->SetMinimum(-5.);

		mu_pT_preds[i]->SetMarkerStyle(21);
		mu_pT_preds[i]->SetMarkerColor(4);

		mu_pT_par[i]->Draw("E1");
		mu_pT_preds[i]->Draw("SAME");

		TString can_mu_pT = Form("imgs/mu_pT_%d.png", i);
		can->SaveAs(can_mu_pT.Data());


		mu_xF_par[i]->SetMaximum(5.);
		mu_xF_par[i]->SetMinimum(-5.);

		mu_xF_par[i]->SetMarkerStyle(8);
		mu_xF_par[i]->SetMarkerColor(2);

		mu_xF_preds[i]->SetMaximum(5.);
		mu_xF_preds[i]->SetMinimum(-5.);

		mu_xF_preds[i]->SetMarkerStyle(21);
		mu_xF_preds[i]->SetMarkerColor(4);

		mu_xF_par[i]->Draw("E1");
		mu_xF_preds[i]->Draw("SAME");

		TString can_mu_xF = Form("imgs/mu_xF_%d.png", i);
		can->SaveAs(can_mu_xF.Data());

		/*
		*/

		nu_mass_par[i]->SetMaximum(5.);
		nu_mass_par[i]->SetMinimum(-5.);

		nu_mass_par[i]->SetMarkerStyle(8);
		nu_mass_par[i]->SetMarkerColor(2);

		nu_mass_preds[i]->SetMaximum(5.);
		nu_mass_preds[i]->SetMinimum(-5.);

		nu_mass_preds[i]->SetMarkerStyle(21);
		nu_mass_preds[i]->SetMarkerColor(4);

		nu_mass_par[i]->Draw("E1");
		nu_mass_preds[i]->Draw("SAME");

		TString can_nu_mass = Form("imgs/nu_mass_%d.png", i);
		can->SaveAs(can_nu_mass.Data());

		nu_pT_par[i]->SetMaximum(5.);
		nu_pT_par[i]->SetMinimum(-5.);

		nu_pT_par[i]->SetMarkerStyle(8);
		nu_pT_par[i]->SetMarkerColor(2);

		nu_pT_preds[i]->SetMaximum(5.);
		nu_pT_preds[i]->SetMinimum(-5.);

		nu_pT_preds[i]->SetMarkerStyle(21);
		nu_pT_preds[i]->SetMarkerColor(4);

		nu_pT_par[i]->Draw("E1");
		nu_pT_preds[i]->Draw("SAME");

		TString can_nu_pT = Form("imgs/nu_pT_%d.png", i);
		can->SaveAs(can_nu_pT.Data());


		nu_xF_par[i]->SetMaximum(5.);
		nu_xF_par[i]->SetMinimum(-5.);

		nu_xF_par[i]->SetMarkerStyle(8);
		nu_xF_par[i]->SetMarkerColor(2);

		nu_xF_preds[i]->SetMaximum(5.);
		nu_xF_preds[i]->SetMinimum(-5.);

		nu_xF_preds[i]->SetMarkerStyle(21);
		nu_xF_preds[i]->SetMarkerColor(4);

		nu_xF_par[i]->Draw("E1");
		nu_xF_preds[i]->Draw("SAME");

		TString can_nu_xF = Form("imgs/nu_xF_%d.png", i);
		can->SaveAs(can_nu_xF.Data());
	}

}