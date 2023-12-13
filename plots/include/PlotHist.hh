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

	float X_par[4][3];
	float X_pred[4][3];

	// double mass_edges[4] = {4., 5.5, 6.5, 9.};
	double pT_edges[5] = {0.0, 0.4, 0.8, 1.2, 2.5};
	// double xF_edges[4] = {-0.1, 0.3, 0.5, 1.0};

public:
	PlotHist();
	virtual ~PlotHist(){};
	void DrawResolution();
	void DrawHist();
};
#endif /* _H_PlotHist_H_ */