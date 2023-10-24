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
    TH2D* true_hist;
    TH2D* reco_hist;
    int phi_bins = 12;
    int costh_bins = 12;
    MakeHist();
    virtual ~MakeHist(){};
    void Init(TString tree_name);
    void FillHist(int ev1, int ev2, double lambda, double mu, double nu);
};


class MakeTree
{
    double true_hist[2][12][12];
    double reco_hist[2][12][12];
    double lambda, mu, nu;
    int h_events = 1000000;
    TRandom* rn;
public:
    TTree* tree;
    MakeTree();
    virtual ~MakeTree(){};
    void Init(TString tree_name, int n);
    void FillTree(MakeHist* mh, int n_events);
};
#endif /* _H_MakeTree_H_ */
