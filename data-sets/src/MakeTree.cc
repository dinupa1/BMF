/*
* dinupa3@gmail.com
*/

#include<TTree.h>
#include<TH3D.h>
#include<TH2D.h>
#include<TString.h>
#include<TRandom3.h>
#include<iostream>

#include "../include/MakeTree.hh"

using namespace std;

MakeTree::MakeTree(TString out_tree)
{
	cout << "===> creating tree with name " << out_tree.Data() << endl;

	tree = new TTree(out_tree.Data(), out_tree.Data());
	tree->Branch("X_det",		X_det,		"X_det[4][10][10]/F");
	tree->Branch("X_par",		X_par,		"X_par[4][3]/F");
}

void MakeTree::FillTree(int n_events, TTree* data, TRandom3* event)
{
	for(int i = 0; i < n_events; i++)
	{
		MakeHist* mh = new MakeHist(event);
		mh->FillHist(data, event);

		/*
		* Fill particle level distributions
		*/
		for(int ii = 0; ii < 4; ii++)
		{
			X_par[ii][0] = mh->lambda_par[ii];
			X_par[ii][1] = mh->mu_par[ii];
			X_par[ii][2] = mh->nu_par[ii];
		}

		/*
		* Fill detector level distributions
		*/

		for(int ii = 0; ii < 10; ii++)
		{
			for(int jj = 0; jj < 10; jj++)
			{
				X_det[0][ii][jj] = mh->phi_costheta[0]->GetBinContent(ii+1, jj+1);
				X_det[1][ii][jj] = mh->phi_costheta[1]->GetBinContent(ii+1, jj+1);
				X_det[2][ii][jj] = mh->phi_costheta[2]->GetBinContent(ii+1, jj+1);
				X_det[3][ii][jj] = mh->phi_costheta[3]->GetBinContent(ii+1, jj+1);
			}
		}

		tree->Fill();

		delete mh;

		if(i%1000==0){cout << "===> starting event " << i << " <===" << endl;}
	}
}