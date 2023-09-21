//
// dinupa3@gmail.com
// 09-11-2023
//

#include <TFile.h>
#include <TTree.h>
#include <TH2D.h>
#include <TH1I.h>
#include <TCanvas.h>
#include <TRandom.h>
#include <TMath.h>
#include <iostream>




using namespace std;

double weight_fn(double lambda, double mu, double nu, double phi, double costh)
{
	double weight = 1. + lambda* costh * costh + 2.* mu* costh * sqrt(1. - costh * costh) * cos(phi) + 0.5* nu* (1. - costh * costh)* cos(2. * phi);
    return weight/(1. + costh * costh);
}

void MakeBinnedData()
{
	auto file = TFile::Open("data.root", "READ");
	auto tree = (TTree*)file->Get("save");
	int entries = tree->GetEntries();

	// cout << "===> number of entries = " << entries << endl;

	double mass, pT, xF, phi, costh, true_phi, true_costh;

	double pi = TMath::Pi();
	double lambda_min = -1.;
	double lambda_max = 1.;
	double mu_min = -0.5;
	double mu_max = 0.5;
	double nu_min = -0.5;
	double nu_max = 0.5;

	tree->SetBranchAddress("mass", &mass);
	tree->SetBranchAddress("pT", &pT);
	tree->SetBranchAddress("xF", &xF);
	tree->SetBranchAddress("phi", &phi);
	tree->SetBranchAddress("costh", &costh);
	tree->SetBranchAddress("true_phi", &true_phi);
	tree->SetBranchAddress("true_costh", &true_costh);

	cout << "===> creating training data " << endl;
	int hist_entries = 10000;
	int ntrain = 100000;

	auto outfile = new TFile("BinMCData.root", "RECREATE");

	double thetas[3];
	double bin_count[144];
	double label[2];

	tree->LoadBaskets(99999999999);

	//
	// make data for VAE training
	//
	auto vae_tree = new TTree("vae_tree", "binned MC data for VAE training");

	vae_tree->Branch("bin_count", 	bin_count, 	"bin_count[144]/D");
	vae_tree->Branch("thetas", 		thetas, 	"thetas[3]/D");
	vae_tree->Branch("label", 		label,		"label[2]/D");

	auto rand = new TRandom3(1);
	auto lamnda_rand = new TRandom3(2);
	auto mu_rand = new TRandom3(3);
	auto nu_rand = new TRandom3(4);

	// create train data with label = 0.
	for(int i = 0; i < ntrain; i++)
	{
		auto hist = new TH2D("hist", "; #phi [rad]; cos#theta [a.u.]", 12, -pi, pi, 12, -0.5, 0.5);

		double lambda_rnd = lamnda_rand->Rndm();
		double mu_rnd = mu_rand->Rndm();
		double nu_rnd = nu_rand->Rndm();

		double lambda = 1.0; //lambda_min + lambda_rnd* (lambda_max - lambda_min);
		double mu = 0.0; //mu_min + mu_rnd* (mu_max - mu_min);
		double nu = nu_min + nu_rnd* (nu_max - nu_min);

		for(int j = 0; j < hist_entries; j++)
		{
			double r = rand->Rndm(i+j);

			tree->GetEntry(TMath::Nint(0. + r* (entries/4. - 0.)));

			if(4.5 < mass){hist->Fill(phi, costh, weight_fn(lambda, mu, nu, true_phi, true_costh));}
		}

		hist->Scale(1./hist->Integral());

		for(int j = 0; j < 12; j++)
		{
			for(int k = 0; k < 12; k++)
			{
				bin_count[12*j + k] = hist->GetBinContent(j+1, k+1);
			}
		}

		thetas[0] = lambda;
		thetas[1] = mu;
		thetas[2] = nu;

		if(i%10000 == 0){cout << "===> creating " << i << " vae data " << endl;}

		vae_tree->Fill();
		delete hist;
	}

	//
	// make data for classifier training
	//
	auto classifier_tree = new TTree("classifier_tree", "binned MC data for VAE training");

	classifier_tree->Branch("bin_count", 	bin_count, 	"bin_count[144]/D");
	classifier_tree->Branch("thetas", 		thetas, 	"thetas[3]/D");
	classifier_tree->Branch("label", 		&label, 		"label/D");

	// auto lamnda_rand = new TRandom3();
	// auto mu_rand = new TRandom3();
	// auto nu_rand = new TRandom3();

	//
	// create train data with label = 0.
	//
	for(int i = 0; i < ntrain; i++)
	{
		auto hist = new TH2D("hist", "; #phi [rad]; cos#theta [a.u.]", 12, -pi, pi, 12, -0.5, 0.5);

		double lambda_rnd = lamnda_rand->Rndm();
		double mu_rnd = mu_rand->Rndm();
		double nu_rnd = nu_rand->Rndm();

		double lambda = 1.0; //lambda_min + lambda_rnd* (lambda_max - lambda_min);
		double mu = 0.0; //mu_min + mu_rnd* (mu_max - mu_min);
		double nu = nu_min + nu_rnd* (nu_max - nu_min);

		for(int j = 0; j < hist_entries; j++)
		{
			double r = rand->Rndm();

			tree->GetEntry(TMath::Nint(entries/4. + r* (2.* entries/4. - entries/4.)));

			if(4.5 < mass){hist->Fill(phi, costh, weight_fn(1.0, 0.0, 0.0, true_phi, true_costh));}
		}

		hist->Scale(1./hist->Integral());

		for(int j = 0; j < 12; j++)
		{
			for(int k = 0; k < 12; k++)
			{
				bin_count[12*j + k] = hist->GetBinContent(j+1, k+1);
			}
		}

		thetas[0] = lambda;
		thetas[1] = mu;
		thetas[2] = nu;
		label = 0.0;

		if(i%10000 == 0){cout << "===> creating " << i << " label 0 data " << endl;}

		classifier_tree->Fill();
		delete hist;
	}

	//
	// create train data with label = 1.
	//
	for(int i = 0; i < ntrain; i++)
	{
		auto hist = new TH2D("hist", "; #phi [rad]; cos#theta [a.u.]", 12, -pi, pi, 12, -0.5, 0.5);

		double lambda_rnd = lamnda_rand->Rndm();
		double mu_rnd = mu_rand->Rndm();
		double nu_rnd = nu_rand->Rndm();

		double lambda = 1.0; //lambda_min + lambda_rnd* (lambda_max - lambda_min);
		double mu = 0.0; //mu_min + mu_rnd* (mu_max - mu_min);
		double nu = nu_min + nu_rnd* (nu_max - nu_min);

		for(int j = 0; j < hist_entries; j++)
		{
			double r = rand->Rndm();

			tree->GetEntry(TMath::Nint(2.* entries/4. + r* (3.* entries/4. - 2.* entries/4.)));

			if(4.5 < mass){hist->Fill(phi, costh, weight_fn(lambda, mu, nu, true_phi, true_costh));}
		}

		hist->Scale(1./hist->Integral());

		for(int j = 0; j < 12; j++)
		{
			for(int k = 0; k < 12; k++)
			{
				bin_count[12*j + k] = hist->GetBinContent(j+1, k+1);
			}
		}

		thetas[0] = lambda;
		thetas[1] = mu;
		thetas[2] = nu;
		label = 1.0;

		if(i%10000 == 0){cout << "===> creating " << i << " label 1 data " << endl;}

		classifier_tree->Fill();
		delete hist;
	}

	int ntest = 30000;

	cout << "===> creating test data " << endl;

	//
	// make secret data
	//
	auto secret_tree = new TTree("secret_tree", "binned MC with secret data");

	secret_tree->Branch("bin_count", 		bin_count, 	"bin_count[144]/D");
	secret_tree->Branch("thetas", 			thetas, 	"thetas[3]/D");
	secret_tree->Branch("label",			&label,		"label/D");

	// double nu_array[11] = {-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5};

	// create test data with label = 1
	for(int i = 0; i < ntest; i++)
	{
		auto hist = new TH2D("hist", "; #phi [rad]; cos#theta [a.u.]", 12, -pi, pi, 12, -0.5, 0.5);

		// lets test with lambda = 0.8, mu = 0.1, nu = 0.2
		double lambda = 1.0;
		double mu = 0.0;
		double nu = 0.2; //nu_array[i];

		for(int j = 0; j < hist_entries; j++)
		{
			double r = rand->Rndm();

			tree->GetEntry(TMath::Nint(entries/2.+ r* (entries - entries/2.)));

			if(4.5 < mass){hist->Fill(phi, costh, weight_fn(lambda, mu, nu, true_phi, true_costh));}
		}

		hist->Scale(1./hist->Integral());

		for(int j = 0; j < 12; j++)
		{
			for(int k = 0; k < 12; k++)
			{
				bin_count[12*j + k] = hist->GetBinContent(j+1, k+1);
			}
		}

		thetas[0] = lambda;
		thetas[1] = mu;
		thetas[2] = nu;

		// cout << "===> nu = " << nu << endl;

		if(i%10000 == 0){cout << "===> creating " << i << " secret data " << endl;}

		secret_tree->Fill();
		delete hist;
	}

	vae_tree->Write();
	classifier_tree->Write();
	secret_tree->Write();
	outfile->Close();
}
