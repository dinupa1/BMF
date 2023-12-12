/*
* dinupa3@gmail.com
*/

#include<TTree.h>
#include<TH3D.h>
#include<TH2D.h>
#include<TMath.h>
#include<TRandom3.h>
#include<TCanvas.h>
#include<TString.h>
#include<iostream>

#include "../include/MakeHist.hh"

using namespace std;

MakeHist::MakeHist(TRandom3* event)
{
	/*
	* Fill theta for each bin
	*/
	for(int i = 0; i < N_BINS; i++)
	{
		lambda_par[i] = event->Uniform(-1., 1.);
		mu_par[i] = event->Uniform(-0.5, 0.5);
		nu_par[i] = event->Uniform(-0.5, 0.5);

		TString hist_name = Form("phi_costheta_%d", i);
		phi_costheta[i] = new TH2D(hist_name.Data(), "; #phi [rad]; cos#theta [a.u.]", 10, -PI, PI, 10, -0.6, 0.6);
	}
}

MakeHist::~MakeHist()
{
	for(int i = 0; i < N_BINS; i++)
	{
		delete phi_costheta[i];
	}
}

void MakeHist::FillHist(TTree* data, TRandom3* event)
{
	int num_events = data->GetEntries();

	double mass, pT, xF, phi, costh, true_phi, true_costh;

	data->SetBranchAddress("mass",			&mass);
	data->SetBranchAddress("pT",			&pT);
	data->SetBranchAddress("xF",			&xF);
	data->SetBranchAddress("phi",			&phi);
	data->SetBranchAddress("costh",			&costh);
	data->SetBranchAddress("true_phi",		&true_phi);
	data->SetBranchAddress("true_costh",	&true_costh);

	/*
	* Fill detector level distributions
	*/

	int n_reco = 10000;
	int n_fill = 0;

	for(int i = 0; i < num_events; i++)
	{
		data->GetEntry(i);

		double acc1 = event->Uniform(5.0, 9.0);
		double acc2 = event->Uniform(0.0, 1.0);

		if(acc1 <= mass && acc2 <= xF)
		{
			for(int j = 0; j < N_BINS; j++)
			{
				if(pT_edges[j] < pT && pT <= pT_edges[j+1])
				{
					double event_weight = weight_fn(lambda_par[j], mu_par[j], nu_par[j], true_phi, true_costh);
					phi_costheta[j]->Fill(phi, costh, event_weight);
				}
			}
			n_fill += 1;
		}
		if(n_fill==n_reco){break;}
	}

	for(int i = 0; i < N_BINS; i++)
	{
		phi_costheta[i]->Scale(1./phi_costheta[i]->Integral());
	}
}

void MakeHist::DrawHist()
{
	TCanvas* can = new TCanvas();

	TH1D* pT_lambda = new TH1D("pT_lambda", "; p_{T}; #lambda [a.u.]", N_BINS, pT_edges);
	TH1D* pT_mu = new TH1D("pT_mu", "; p_{T}; #mu [a.u.]", N_BINS, pT_edges);
	TH1D* pT_nu = new TH1D("pT_nu", "; p_{T}; #nu [a.u.]", N_BINS, pT_edges);

	for(int i = 0; i < N_BINS; i++)
	{
		pT_lambda->SetBinContent(i+1, lambda_par[i]);
		pT_mu->SetBinContent(i+1, mu_par[i]);
		pT_nu->SetBinContent(i+1, nu_par[i]);
	}

	pT_lambda->Draw();
	can->SaveAs("imgs/pT_lambda.png");

	pT_mu->Draw();
	can->SaveAs("imgs/pT_mu.png");

	pT_nu->Draw();
	can->SaveAs("imgs/pT_nu.png");


	for(int i = 0; i < N_BINS; i++)
	{
		phi_costheta[i]->Draw("COLZ");

		TString pic_name = Form("imgs/phi_costheta_%d.png", i);
		can->SaveAs(pic_name.Data());
	}
}