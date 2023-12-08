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
	phi_costheta = new TH2D("phi_costh", "; #phi [rad]; cos#theta [a.u.]", 12, -PI, PI, 12, 0.6, 0.6);
	cosphi_costheta = new TH2D("cosphi_costh", "; cos#phi [a.u.]; cos#theta [a.u.]", 12, -1., 1., 12, 0.6, 0.6);
	cos2phi_costheta = new TH2D("cos2phi_costh", "; cos2#phi [a.u.]; cos#theta [a.u.]", 12, -1., 1., 12, 0.6, 0.6);

	/*
	* Fill theta for each bin
	*/

	for(int i = 0; i < N_BINS; i++)
	{
		for(int j = 0; j < N_BINS; j++)
		{
			for(int k = 0; k < N_BINS; k++)
			{
				lambda_par[i][j][k] = event->Uniform(-1., 1.);
				mu_par[i][j][k] = event->Uniform(-0.5, 0.5);
				nu_par[i][j][k] = event->Uniform(-0.5, 0.5);
			}
		}
	}
}

MakeHist::~MakeHist()
{
	delete phi_costheta;
	delete cosphi_costheta;
	delete cos2phi_costheta;
}

void MakeHist::FillHist(TTree* data, TRandom3* event)
{
	int num_events = data->GetEntries();

	//int fpga1;
	double true_mass, true_pT, true_xF, true_phi, true_costh, phi, costh;

	//data->SetBranchAddress("fpga1",			&fpga1);
	data->SetBranchAddress("true_mass",		&true_mass);
	data->SetBranchAddress("true_pT",		&true_pT);
	data->SetBranchAddress("true_xF",		&true_xF);
	data->SetBranchAddress("true_phi",		&true_phi);
	data->SetBranchAddress("true_costh",	&true_costh);
	data->SetBranchAddress("phi",			&phi);
	data->SetBranchAddress("costh",			&costh);

	/*
	* Fill detector level distributions
	*/

	int n_reco = 10000;
	int n_fill;
	for(int i = 0; i < num_events; i++)
	{
		data->GetEntry(i);

		/*
		* Random variables for event rejection
		*/
		float rand_mass = event->Uniform(4., 10.);
		float rand_pT = event->Uniform(0., 3.);
		float rand_xF = event->Uniform(-0.2, 1.);

		if(rand_mass <= true_mass && rand_pT <= true_pT && rand_xF <= true_xF)
		{
			for(int ii = 0; ii < N_BINS; ii++)
			{
				double mass_min = mass_edges[ii];
				double mass_max = mass_edges[ii+1];

				for(int jj = 0; jj < N_BINS; jj++)
				{
					double pT_min = pT_edges[jj];
					double pT_max = pT_edges[jj+1];

					for(int kk = 0; kk < N_BINS; kk++)
					{
						double xF_min = xF_edges[kk];
						double xF_max = xF_edges[kk+1];

						if(true_mass > mass_min && true_mass <= mass_max &&
							true_pT > pT_min && true_pT <= pT_max && 
							true_xF > xF_min && true_xF <= xF_max)
						{
							double theta_weight = weight_fn(lambda_par[ii][jj][kk], mu_par[ii][jj][kk], nu_par[ii][jj][kk], true_phi, true_costh);

							phi_costheta->Fill(phi, costh, theta_weight);
							cosphi_costheta->Fill(cos(phi), costh, theta_weight);
							cos2phi_costheta->Fill(cos(2.* phi), costh, theta_weight);
							n_fill +=1;
						}
					}
				}
			}
		} // filling is done
		if(n_fill==n_reco){break;}
	}

	/*
	* Normalize to unity
	*/

	phi_costheta->Scale(1./phi_costheta->Integral());
	cosphi_costheta->Scale(1./cosphi_costheta->Integral());
	cos2phi_costheta->Scale(1./cos2phi_costheta->Integral());
}

void MakeHist::DrawHist()
{
	TCanvas* can = new TCanvas();

	phi_costheta->Draw("COLZ");
	can->SaveAs("imgs/phi_costheta.png");

	cosphi_costheta->Draw("COLZ");
	can->SaveAs("imgs/cosphi_costheta.png");

	cos2phi_costheta->Draw("COLZ");
	can->SaveAs("imgs/cos2phi_costheta.png");

	TH3D* hist_lambda = new TH3D("hist_lambda", "", N_BINS, mass_edges, N_BINS, pT_edges, N_BINS, xF_edges);
	TH3D* hist_mu = new TH3D("hist_mu", "", N_BINS, mass_edges, N_BINS, pT_edges, N_BINS, xF_edges);
	TH3D* hist_nu = new TH3D("hist_nu", "", N_BINS, mass_edges, N_BINS, pT_edges, N_BINS, xF_edges);

	for(int i = 0; i < N_BINS; i++)
	{
		for(int j = 0; j < N_BINS; j++)
		{
			for(int k = 0; k < N_BINS; k++)
			{
				hist_lambda->SetBinContent(i+1, j+1, k+1, lambda_par[i][j][k]);
				hist_mu->SetBinContent(i+1, j+1, k+1, mu_par[i][j][k]);
				hist_nu->SetBinContent(i+1, j+1, k+1, nu_par[i][j][k]);
				TString par_out = Form("lambda = %.3f, mu = %.3f, nu = %.3f", lambda_par[i][j][k], mu_par[i][j][k], nu_par[i][j][k]);
				cout << par_out.Data() << endl;
			}
		}
	}

	TH1D* lambda_pT = (TH1D*)hist_lambda->ProjectionY();
	lambda_pT->SetNameTitle("lambda_pT", "; p_{T} [GeV]; #lambda [a.u.]");
	// lambda_pT->SetMarkerStyle(8);
	// lambda_pT->SetMarkerColor(4);

	lambda_pT->Draw("HIST");
	can->SaveAs("imgs/lambda_pT.png");

	TH1D* mu_pT = (TH1D*)hist_mu->ProjectionY();
	mu_pT->SetNameTitle("mu_pT", "; p_{T} [GeV]; #mu [a.u.]");
	// mu_pT->SetMarkerStyle(8);
	// mu_pT->SetMarkerColor(4);

	mu_pT->Draw("HIST");
	can->SaveAs("imgs/mu_pT.png");

	TH1D* nu_pT = (TH1D*)hist_nu->ProjectionY();
	nu_pT->SetNameTitle("nu_pT", "; p_{T} [GeV]; #nu [a.u.]");
	// nu_pT->SetMarkerStyle(8);
	// nu_pT->SetMarkerColor(4);

	nu_pT->Draw("HIST");
	can->SaveAs("imgs/nu_pT.png");
}