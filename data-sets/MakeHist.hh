/*
* dinupa3@gmail.com
*/

#ifndef _H_MakeHist_H_
#define _H_MakeHist_H_

#include<TTree.h>
#include<TH3D.h>
#include<TH2D.h>
#include<TMath.h>
#include<TRandom3.h>
#include<iostream>

using namespace std;

double weight_fn(float lambda, float mu, float nu, float phi, float costh)
{
	float weight = 1. + lambda* costh* costh + 2.* mu* costh* sqrt(1. - costh* costh) *cos(phi) + 0.5* nu* (1. - costh* costh)* cos(2.* phi);
	return weight/(1. + costh* costh);
}

class MakeHist
{
	double PI = TMath::Pi();
	int N_BINS = 3;

	double mass_edges[4] = {4., 5.5, 6.5, 9.};
	double pT_edges[4] = {0., 0.5, 1., 2.5};
	double xF_edges[4] = {-0.1, 0.3, 0.5, 1.0};

public:
	TH3D* lambda_par; // injected lambda, mu, nu values
	TH3D* mu_par;
	TH3D* nu_par;
	TH2D* phi_costheta; // phi vs. costh histogram in the detector level
	TH2D* sin2theta_cosphi; // sin2theta vs. cosphi in the detector level
	TH2D* sintheta2_cos2phi; // sinth2 vs. cos2phi in the detector level

	MakeHist(TRandom3* event);
	virtual ~MakeHist(){};
	void FillHist(TTree* data, TRandom3* event);
};


MakeHist::MakeHist(TRandom3* event)
{
	lambda_par = new TH3D("lambda_par", "theta_par", N_BINS, mass_edges, N_BINS, pT_edges, N_BINS, xF_edges);
	mu_par = new TH3D("mu_par", "theta_par", N_BINS, mass_edges, N_BINS, pT_edges, N_BINS, xF_edges);
	nu_par = new TH3D("nu_par", "theta_par", N_BINS, mass_edges, N_BINS, pT_edges, N_BINS, xF_edges);

	phi_costheta = new TH2D("phi_costh", "; #phi [rad]; cos#theta [a.u.]", 12, -PI, PI, 12, 0.6, 0.6);
	sin2theta_cosphi = new TH2D("sin2theta_cosphi", "; sin2#theta [a.u.]; cos#phi [a.u.]", 12, -1., 1., 12, -1., 1.);
	sintheta2_cos2phi = new TH2D("sintheta_cos2phi", "; sin#theta [a.u.]; cos2#phi [a.u.]", 12, -1., 1., 12, -1., 1.);

	/*
	* Fill theta for each bin
	*/

	for(int i = 0; i < 3; i++)
	{
		for(int j = 0; j < 3; j++)
		{
			for(int k = 0; k < 3; k++)
			{
				lambda_par->SetBinContent(i+1, j+1, k+1, event->Uniform(-1., 1.));
				lambda_par->SetBinError(i+1, j+1, k+1, event->Uniform(0., 1.));

				mu_par->SetBinContent(i+1, j+1, k+1, event->Uniform(-0.5, 0.5));
				mu_par->SetBinError(i+1, j+1, k+1, event->Uniform(0., 1.));

				nu_par->SetBinContent(i+1, j+1, k+1, event->Uniform(-0.5, 0.5));
				nu_par->SetBinError(i+1, j+1, k+1, event->Uniform(0., 1.));
			}
		}
	}
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
				int mass_bin = ii+1;

				for(int jj = 0; jj < N_BINS; jj++)
				{
					double pT_min = pT_edges[jj];
					double pT_max = pT_edges[jj+1];
					int pT_bin = jj+1;

					for(int kk = 0; kk < N_BINS; kk++)
					{
						double xF_min = xF_edges[kk];
						double xF_max = xF_edges[kk+1];
						int xF_bin = kk+1;

						if(true_mass > mass_min && true_mass <= mass_max &&
							true_pT > pT_min && true_pT <= pT_max && 
							true_xF > xF_min && true_xF <= xF_max)
						{
							double lambda = event->Gaus(lambda_par->GetBinContent(mass_bin, pT_bin, xF_bin), lambda_par->GetBinError(mass_bin, pT_bin, xF_bin));
							double mu = event->Gaus(mu_par->GetBinContent(mass_bin, pT_bin, xF_bin), mu_par->GetBinError(mass_bin, pT_bin, xF_bin));
							double nu = event->Gaus(nu_par->GetBinContent(mass_bin, pT_bin, xF_bin), nu_par->GetBinError(mass_bin, pT_bin, xF_bin));
							double theta_weight = weight_fn(lambda, mu, nu, true_phi, true_costh);

							phi_costheta->Fill(phi, costh, theta_weight);
							sin2theta_cosphi->Fill(2.* costh* sqrt(1. - costh* costh), cos(phi), theta_weight);
							sintheta2_cos2phi->Fill(1. - costh* costh, cos(2.* phi), theta_weight);
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
	sin2theta_cosphi->Scale(1./sin2theta_cosphi->Integral());
	sintheta2_cos2phi->Scale(1./sintheta2_cos2phi->Integral());
}
#endif /* _H_MakeHist_H_ */