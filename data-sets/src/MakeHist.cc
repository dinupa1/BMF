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

MakeHist::MakeHist()
{
    phi_costheta = new TH2D("phi_costh", "; #phi [rad]; cos#theta [a.u.]", 10, -PI, PI, 10, -0.6, 0.6);
}

MakeHist::~MakeHist()
{
    delete phi_costheta;
}

void MakeHist::FillHist(TTree* data, TRandom3* event, double lambda, double mu, double nu)
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
            if(pT_edges[0] < pT && pT <= pT_edges[1])
            {
                double event_weight = weight_fn(lambda, mu, nu, true_phi, true_costh);
                phi_costheta->Fill(phi, costh, event_weight);
            }
            n_fill += 1;
        }
        if(n_fill==n_reco){break;}
    }
    
    phi_costheta->Scale(1./phi_costheta->Integral());
}

/*
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
*/