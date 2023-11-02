//
// dinupa3@gmail.com
//

#ifndef _H_MakeTree_H_
#define _H_MakeTree_H_


// ROOT
#include <TFile.h>
#include <TTree.h>
#include <TH2D.h>
#include <THnSparse.h>
#include <TMath.h>
#include <TSystem.h>
#include <TCanvas.h>
#include <TRandom.h>
#include <TCanvas.h>

class MakeHist
{
    TTree* data;
    int fpga1;
    float mass, pT, xF, phi, costh, true_mass, true_pT, true_xF, true_phi, true_costh;
    double pi = TMath::Pi();
public:
    int events;

    double true_count[3][3][4][12][12];
    double true_error[3][3][4][12][12];
    double reco_count[3][3][4][12][12];
    double reco_error[3][3][4][12][12];

    MakeHist();
    virtual ~MakeHist(){};
    void Init(TString tree_name);
    void FillHist(double lambda, double mu, double nu, int seed);
};


class MakeTree
{
    double true_hist[2][72][72];
    double reco_hist[2][72][72];
    double lambda, mu, nu;
public:
    TTree* tree;
    MakeTree();
    virtual ~MakeTree(){};
    void Init(TString tree_name);
    void FillTree(MakeHist* mh, int n_events, int seed);
};
#endif /* _H_MakeTree_H_ */
