/*
* dinupa3@gmail.com
*/

#ifndef _H_PlotHist_H_
#define _H_PlotHist_H_

#include<TFile.h>
#include<TTree.h>
#include<TCanvas.h>
#include<TH1D.h>
#include<TH3D.h>
#include<TString.h>
#include<iostream>

using namespace std;

class PlotHist
{
	TTree* tree;
	int nevents;

	float X_par[6][3][3];
	float X_preds[6][3][3];

	double lambda_array[3][3];
	double mu_array[3][3];
	double nu_array[3][3];

	TH1D* lambda_mass[3];
	TH1D* lambda_pT[3];
	TH1D* lambda_xF[3];

	TH1D* mu_mass[3];
	TH1D* mu_pT[3];
	TH1D* mu_xF[3];

	TH1D* nu_mass[3];
	TH1D* nu_pT[3];
	TH1D* nu_xF[3];

	double mass_edges[4] = {4., 5.5, 6.5, 9.};
	double pT_edges[4] = {0., 0.5, 1., 2.5};
	double xF_edges[4] = {-0.1, 0.3, 0.5, 1.0};

public:
	PlotHist();
	virtual ~PlotHist(){};
	void DeltaHist(int i);
	void DrawResolution();
	//void DrawHist();
};
#endif /* _H_PlotHist_H_ */