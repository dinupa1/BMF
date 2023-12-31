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
	tree->Branch("X_det",		X_det,		"X_det[100]/D");
	tree->Branch("X_par",		X_par,		"X_par[3]/D");
}

void MakeTree::FillTree(int n_events, TTree* data, TRandom3* event)
{
	for(int i = 0; i < n_events; i++)
	{
        
        X_par[0] = event->Uniform(-1., 1.);
        X_par[1] = event->Uniform(-0.5, 0.5);
        X_par[2] = event->Uniform(-0.5, 0.5);

        double nhist = event->Uniform(10., 20);

        for(int j = 0; j < TMath::Nint(nhist); j++)
        {
            MakeHist* mh = new MakeHist();
            mh->FillHist(data, event, X_par[0], X_par[1], X_par[2]);
            
            /*
            * Fill detector level distributions
            */
            
            for(int ii = 0; ii < 10; ii++)
            {
                for(int jj = 0; jj < 10; jj++)
                {
                    X_det[10* ii + jj] = mh->phi_costheta->GetBinContent(ii+1, jj+1);
                }
            }
            
            tree->Fill();
            delete mh;
        }
        
        if(i%1000==0){cout << "===> starting event " << i << " <===" << endl;}
	}
}