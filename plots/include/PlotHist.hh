/*
* dinupa3@gmail.com
*/

#ifndef _H_PlotHist_H_
#define _H_PlotHist_H_

#include<TFile.h>
#include<TTree.h>
#include<TCanvas.h>
#include<TH1D.h>
#include<TString.h>
#include<iostream>

using namespace std;

class PlotHist
{
	TTree* tree;
	int nevents;
	float X_par[6][3][3];
	float X_preds[6][3][3];
public:
	PlotHist();
	virtual ~PlotHist(){};
	void DrawResolution(int theta_dim, TString hist_title, double x_min, double x_max);
	void DrawHist();
};
#endif /* _H_PlotHist_H_ */