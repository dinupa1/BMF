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

void PlotHist::DrawResolution()
{
	TCanvas* can = new TCanvas();

	for(int i = 0; i < 6; i++)
	{
		for(int j = 0; j < 3; j++)
		{
			for(int k = 0; k < 3; k++)
			{
				TString hist_name = Form("hist_%d_%d_%d", i, j, k);
				TH1D* hist = new TH1D(hist_name.Data(), "; #Delta [a.u.]; events [a.u.]", 20, -10., 10.);

				for(int ii = 0; ii < nevents; ii++)
				{
					tree->GetEntry(ii);
					hist->Fill(X_par[i][j][k] - X_preds[i][j][k]);
				}

				TString can_name = Form("imgs/hist_%d_%d_%d.png", i, j, k);
				hist->Draw("E1");
				can->SaveAs(can_name.Data());
			}
		}
	}
}