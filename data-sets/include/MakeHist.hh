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
#include<TCanvas.h>
#include<iostream>

using namespace std;

double weight_fn(float lambda, float mu, float nu, float phi, float costh)
{
	double weight = 1. + lambda* costh* costh + 2.* mu* costh* sqrt(1. - costh* costh) *cos(phi) + 0.5* nu* (1. - costh* costh)* cos(2.* phi);
	return weight/(1. + costh* costh);
}

class MakeHist
{
	double PI = TMath::Pi();
	int N_BINS = 4;

	//double mass_edges[4] = {4., 5.5, 6.2, 9.};
	double pT_edges[5] = {0.0, 0.4, 0.8, 1.2, 2.5};
	//double xF_edges[4] = {-0.1, 0.2, 0.5, 0.7, 1.0};

public:
	float lambda_par; // injected lambda, mu, nu values
	float mu_par;
	float nu_par;
	TH2D* phi_costheta; // phi vs. costh histogram in the detector level

	MakeHist();
	virtual ~MakeHist();
	void FillHist(TTree* data, TRandom3* event, double lambda, double mu, double nu);
	//void DrawHist();
};
#endif /* _H_MakeHist_H_ */