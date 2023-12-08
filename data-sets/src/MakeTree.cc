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
	tree->Branch("X_det",		X_det,		"X_det[3][12][12]/F");
	tree->Branch("X_par",		X_par,		"X_par[9][3][3]/F");
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

		for(int ii = 0; ii < 3; ii++)
		{
			for(int jj = 0; jj < 3; jj++)
			{
				for(int kk = 0; kk < 3; kk++)
				{
					X_par[ii][jj][kk] = mh->lambda_par[ii][jj][kk];
					X_par[3+ii][jj][kk] = mh->mu_par[ii][jj][kk];
					X_par[6+ii][jj][kk] = mh->nu_par[ii][jj][kk];
				}
			}
		}

		/*
		* Fill detector level distributions
		*/

		for(int ii = 0; ii < 12; ii += 4)
		{
			for(int jj = 0; jj < 12; jj += 4)
			{

				/*
				* phi vs. costh
				*/

				X_det[0][ii+0][jj+0] = mh->phi_costheta->GetBinContent(ii+1+0, jj+1+0);
				X_det[0][ii+0][jj+1] = mh->phi_costheta->GetBinContent(ii+1+0, jj+1+1);
				X_det[0][ii+0][jj+2] = mh->phi_costheta->GetBinContent(ii+1+0, jj+1+2);
				X_det[0][ii+0][jj+3] = mh->phi_costheta->GetBinContent(ii+1+0, jj+1+3);

				X_det[0][ii+1][jj+0] = mh->phi_costheta->GetBinContent(ii+1+1, jj+1+0);
				X_det[0][ii+1][jj+1] = mh->phi_costheta->GetBinContent(ii+1+1, jj+1+1);
				X_det[0][ii+1][jj+2] = mh->phi_costheta->GetBinContent(ii+1+1, jj+1+2);
				X_det[0][ii+1][jj+3] = mh->phi_costheta->GetBinContent(ii+1+1, jj+1+3);

				X_det[0][ii+2][jj+0] = mh->phi_costheta->GetBinContent(ii+1+2, jj+1+0);
				X_det[0][ii+2][jj+1] = mh->phi_costheta->GetBinContent(ii+1+2, jj+1+1);
				X_det[0][ii+2][jj+2] = mh->phi_costheta->GetBinContent(ii+1+2, jj+1+2);
				X_det[0][ii+2][jj+3] = mh->phi_costheta->GetBinContent(ii+1+2, jj+1+3);

				X_det[0][ii+3][jj+0] = mh->phi_costheta->GetBinContent(ii+1+3, jj+1+0);
				X_det[0][ii+3][jj+1] = mh->phi_costheta->GetBinContent(ii+1+3, jj+1+1);
				X_det[0][ii+3][jj+2] = mh->phi_costheta->GetBinContent(ii+1+3, jj+1+2);
				X_det[0][ii+3][jj+3] = mh->phi_costheta->GetBinContent(ii+1+3, jj+1+3);

				/*
				* cosphi vs. costh
				*/

				X_det[1][ii+0][jj+0] = mh->cosphi_costheta->GetBinContent(ii+1+0, jj+1+0);
				X_det[1][ii+0][jj+1] = mh->cosphi_costheta->GetBinContent(ii+1+0, jj+1+1);
				X_det[1][ii+0][jj+2] = mh->cosphi_costheta->GetBinContent(ii+1+0, jj+1+2);
				X_det[1][ii+0][jj+3] = mh->cosphi_costheta->GetBinContent(ii+1+0, jj+1+3);

				X_det[1][ii+1][jj+0] = mh->cosphi_costheta->GetBinContent(ii+1+1, jj+1+0);
				X_det[1][ii+1][jj+1] = mh->cosphi_costheta->GetBinContent(ii+1+1, jj+1+1);
				X_det[1][ii+1][jj+2] = mh->cosphi_costheta->GetBinContent(ii+1+1, jj+1+2);
				X_det[1][ii+1][jj+3] = mh->cosphi_costheta->GetBinContent(ii+1+1, jj+1+3);

				X_det[1][ii+2][jj+0] = mh->cosphi_costheta->GetBinContent(ii+1+2, jj+1+0);
				X_det[1][ii+2][jj+1] = mh->cosphi_costheta->GetBinContent(ii+1+2, jj+1+1);
				X_det[1][ii+2][jj+2] = mh->cosphi_costheta->GetBinContent(ii+1+2, jj+1+2);
				X_det[1][ii+2][jj+3] = mh->cosphi_costheta->GetBinContent(ii+1+2, jj+1+3);

				X_det[1][ii+3][jj+0] = mh->cosphi_costheta->GetBinContent(ii+1+3, jj+1+0);
				X_det[1][ii+3][jj+1] = mh->cosphi_costheta->GetBinContent(ii+1+3, jj+1+1);
				X_det[1][ii+3][jj+2] = mh->cosphi_costheta->GetBinContent(ii+1+3, jj+1+2);
				X_det[1][ii+3][jj+3] = mh->cosphi_costheta->GetBinContent(ii+1+3, jj+1+3);

				/*
				* cos2phi vs. costh
				*/

				X_det[2][ii+0][jj+0] = mh->cos2phi_costheta->GetBinContent(ii+1+0, jj+1+0);
				X_det[2][ii+0][jj+1] = mh->cos2phi_costheta->GetBinContent(ii+1+0, jj+1+1);
				X_det[2][ii+0][jj+2] = mh->cos2phi_costheta->GetBinContent(ii+1+0, jj+1+2);
				X_det[2][ii+0][jj+3] = mh->cos2phi_costheta->GetBinContent(ii+1+0, jj+1+3);

				X_det[2][ii+1][jj+0] = mh->cos2phi_costheta->GetBinContent(ii+1+1, jj+1+0);
				X_det[2][ii+1][jj+1] = mh->cos2phi_costheta->GetBinContent(ii+1+1, jj+1+1);
				X_det[2][ii+1][jj+2] = mh->cos2phi_costheta->GetBinContent(ii+1+1, jj+1+2);
				X_det[2][ii+1][jj+3] = mh->cos2phi_costheta->GetBinContent(ii+1+1, jj+1+3);

				X_det[2][ii+2][jj+0] = mh->cos2phi_costheta->GetBinContent(ii+1+2, jj+1+0);
				X_det[2][ii+2][jj+1] = mh->cos2phi_costheta->GetBinContent(ii+1+2, jj+1+1);
				X_det[2][ii+2][jj+2] = mh->cos2phi_costheta->GetBinContent(ii+1+2, jj+1+2);
				X_det[2][ii+2][jj+3] = mh->cos2phi_costheta->GetBinContent(ii+1+2, jj+1+3);

				X_det[2][ii+3][jj+0] = mh->cos2phi_costheta->GetBinContent(ii+1+3, jj+1+0);
				X_det[2][ii+3][jj+1] = mh->cos2phi_costheta->GetBinContent(ii+1+3, jj+1+1);
				X_det[2][ii+3][jj+2] = mh->cos2phi_costheta->GetBinContent(ii+1+3, jj+1+2);
				X_det[2][ii+3][jj+3] = mh->cos2phi_costheta->GetBinContent(ii+1+3, jj+1+3);
			}
		}

		tree->Fill();

		delete mh;
		/*delete mh->phi_costheta;
		delete mh->sin2theta_cosphi;
		delete mh->sintheta2_cos2phi;
		delete mh->lambda_par;
		delete mh->mu_par;
		delete mh->nu_par;*/

		if(i%1000==0){cout << "===> starting event " << i << " <===" << endl;}
	}
}