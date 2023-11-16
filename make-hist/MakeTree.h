/*
 * dinupa3@gmail.com
 */

#include <TFile.h>
#include <TTree.h>
#include <TMath.h>
#include <TCanvas.h>
#include <TRandom3.h>
#include <iostream>

using namespace std;

#ifndef _H_MakeTree_H_
#define _H_MakeTree_H_

class MakeTree
{
    float true_hist[2][72][72];
    float reco_hist[2][72][72];
    float lambda, mu, nu;

public:
    TTree* tree;
    MakeTree();
    virtual ~MakeTree(){};
    void Init(TString tree_name);
    void FillTree(MakeHist* mh, int n_events, TRandom3* event);
};

MakeTree::MakeTree()
{;}

void MakeTree::Init(TString tree_name)
{
    tree = new TTree(tree_name.Data(), tree_name.Data());
    tree->Branch("true_hist",       true_hist,      "true_hist[2][72][72]/F");
    tree->Branch("reco_hist",       reco_hist,      "reco_hist[2][72][72]/F");
    tree->Branch("lambda",          &lambda,        "lambda/F");
    tree->Branch("mu",              &mu,            "mu/F");
    tree->Branch("nu",              &nu,            "nu/F");
}


void MakeTree::FillTree(MakeHist* mh, int n_events, TRandom3* event)
{
    for(int i = 0; i < n_events; i++)
    {
        lambda = event->Uniform(-1., 1.);
        mu = event->Uniform(-0.5, 0.5);
        nu = event->Uniform(-0.5, 0.5);

        mh->FillHist(lambda, mu, nu, event);

        for(int ii = 0; ii < 72; ii += 4)
        {
            for(int jj = 0; jj < 72; jj += 4)
            {
                /*
                 * fill particle level info.
                 */

                // ii = 0
                true_hist[0][ii+0][jj+0] = mh->true_count[72*(ii+0) + (jj+0)];
                true_hist[1][ii+0][jj+0] = mh->true_error2[72*(ii+0) + (jj+0)];

                true_hist[0][ii+0][jj+1] = mh->true_count[72*(ii+0) + (jj+1)];
                true_hist[1][ii+0][jj+1] = mh->true_error2[72*(ii+0) + (jj+1)];

                true_hist[0][ii+0][ii+2] = mh->true_count[72*(ii+0) + (jj+2)];
                true_hist[1][ii+0][ii+2] = mh->true_error2[72*(ii+0) + (jj+2)];

                true_hist[0][ii+0][ii+3] = mh->true_count[72*(ii+0) + (jj+3)];
                true_hist[1][ii+0][ii+3] = mh->true_error2[72*(ii+0) + (jj+3)];

                // ii = 1
                true_hist[0][ii+1][jj+0] = mh->true_count[72*(ii+1) + (jj+0)];
                true_hist[1][ii+1][jj+0] = mh->true_error2[72*(ii+1) + (jj+0)];

                true_hist[0][ii+1][jj+1] = mh->true_count[72*(ii+1) + (jj+1)];
                true_hist[1][ii+1][jj+1] = mh->true_error2[72*(ii+1) + (jj+1)];

                true_hist[0][ii+1][ii+2] = mh->true_count[72*(ii+1) + (jj+2)];
                true_hist[1][ii+1][ii+2] = mh->true_error2[72*(ii+1) + (jj+2)];

                true_hist[0][ii+1][ii+3] = mh->true_count[72*(ii+1) + (jj+3)];
                true_hist[1][ii+1][ii+3] = mh->true_error2[72*(ii+1) + (jj+3)];

                // ii = 2
                true_hist[0][ii+2][jj+0] = mh->true_count[72*(ii+2) + (jj+0)];
                true_hist[1][ii+2][jj+0] = mh->true_error2[72*(ii+2) + (jj+0)];

                true_hist[0][ii+2][jj+1] = mh->true_count[72*(ii+2) + (jj+1)];
                true_hist[1][ii+2][jj+1] = mh->true_error2[72*(ii+2) + (jj+1)];

                true_hist[0][ii+2][ii+2] = mh->true_count[72*(ii+2) + (jj+2)];
                true_hist[1][ii+2][ii+2] = mh->true_error2[72*(ii+2) + (jj+2)];

                true_hist[0][ii+2][ii+3] = mh->true_count[72*(ii+2) + (jj+3)];
                true_hist[1][ii+2][ii+3] = mh->true_error2[72*(ii+2) + (jj+3)];

                // ii = 3
                true_hist[0][ii+3][jj+0] = mh->true_count[72*(ii+3) + (jj+0)];
                true_hist[1][ii+3][jj+0] = mh->true_error2[72*(ii+3) + (jj+0)];

                true_hist[0][ii+3][jj+1] = mh->true_count[72*(ii+3) + (jj+1)];
                true_hist[1][ii+3][jj+1] = mh->true_error2[72*(ii+3) + (jj+1)];

                true_hist[0][ii+3][ii+2] = mh->true_count[72*(ii+3) + (jj+2)];
                true_hist[1][ii+3][ii+2] = mh->true_error2[72*(ii+3) + (jj+2)];

                true_hist[0][ii+3][ii+3] = mh->true_count[72*(ii+3) + (jj+3)];
                true_hist[1][ii+3][ii+3] = mh->true_error2[72*(ii+3) + (jj+3)];

                /*
                 * fill detector level
                 */

                // ii = 0
                reco_hist[0][ii+0][jj+0] = mh->reco_count[72*(ii+0) + (jj+0)];
                reco_hist[1][ii+0][jj+0] = mh->reco_error2[72*(ii+0) + (jj+0)];

                reco_hist[0][ii+0][jj+1] = mh->reco_count[72*(ii+0) + (jj+1)];
                reco_hist[1][ii+0][jj+1] = mh->reco_error2[72*(ii+0) + (jj+1)];

                reco_hist[0][ii+0][ii+2] = mh->reco_count[72*(ii+0) + (jj+2)];
                reco_hist[1][ii+0][ii+2] = mh->reco_error2[72*(ii+0) + (jj+2)];

                reco_hist[0][ii+0][ii+3] = mh->reco_count[72*(ii+0) + (jj+3)];
                reco_hist[1][ii+0][ii+3] = mh->reco_error2[72*(ii+0) + (jj+3)];

                // ii = 1
                reco_hist[0][ii+1][jj+0] = mh->reco_count[72*(ii+1) + (jj+0)];
                reco_hist[1][ii+1][jj+0] = mh->reco_error2[72*(ii+1) + (jj+0)];

                reco_hist[0][ii+1][jj+1] = mh->reco_count[72*(ii+1) + (jj+1)];
                reco_hist[1][ii+1][jj+1] = mh->reco_error2[72*(ii+1) + (jj+1)];

                reco_hist[0][ii+1][ii+2] = mh->reco_count[72*(ii+1) + (jj+2)];
                reco_hist[1][ii+1][ii+2] = mh->reco_error2[72*(ii+1) + (jj+2)];

                reco_hist[0][ii+1][ii+3] = mh->reco_count[72*(ii+1) + (jj+3)];
                reco_hist[1][ii+1][ii+3] = mh->reco_error2[72*(ii+1) + (jj+3)];

                // ii = 2
                reco_hist[0][ii+2][jj+0] = mh->reco_count[72*(ii+2) + (jj+0)];
                reco_hist[1][ii+2][jj+0] = mh->reco_error2[72*(ii+2) + (jj+0)];

                reco_hist[0][ii+2][jj+1] = mh->reco_count[72*(ii+2) + (jj+1)];
                reco_hist[1][ii+2][jj+1] = mh->reco_error2[72*(ii+2) + (jj+1)];

                reco_hist[0][ii+2][ii+2] = mh->reco_count[72*(ii+2) + (jj+2)];
                reco_hist[1][ii+2][ii+2] = mh->reco_error2[72*(ii+2) + (jj+2)];

                reco_hist[0][ii+2][ii+3] = mh->reco_count[72*(ii+2) + (jj+3)];
                reco_hist[1][ii+2][ii+3] = mh->reco_error2[72*(ii+2) + (jj+3)];

                // ii = 3
                reco_hist[0][ii+3][jj+0] = mh->reco_count[72*(ii+3) + (jj+0)];
                reco_hist[1][ii+3][jj+0] = mh->reco_error2[72*(ii+3) + (jj+0)];

                reco_hist[0][ii+3][jj+1] = mh->reco_count[72*(ii+3) + (jj+1)];
                reco_hist[1][ii+3][jj+1] = mh->reco_error2[72*(ii+3) + (jj+1)];

                reco_hist[0][ii+3][ii+2] = mh->reco_count[72*(ii+3) + (jj+2)];
                reco_hist[1][ii+3][ii+2] = mh->reco_error2[72*(ii+3) + (jj+2)];

                reco_hist[0][ii+3][ii+3] = mh->reco_count[72*(ii+3) + (jj+3)];
                reco_hist[1][ii+3][ii+3] = mh->reco_error2[72*(ii+3) + (jj+3)];

            }
        }

        if (i+1%100==0) {cout << "===> event : " << i+1 << " lambda : " << lambda << " mu : " << mu << " nu : " << nu << endl;}
        // cout << "===> event : " << i << " lambda : " << lambda << " mu : " << mu << " nu : " << nu << endl;

        tree->Fill();
    }
}
#endif /* _H_MakeTree_H_ */
