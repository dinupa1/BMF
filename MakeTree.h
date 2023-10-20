//
// dinupa3@gmail.com
//

#ifndef _H_MakeTree_H_
#define _H_MakeTree_H_


// ROOT
#include <TFile.h>
#include <TTree.h>
#include <TH2D.h>
#include <TMath.h>
#include <TSystem.h>
#include <TCanvas.h>
#include <TRandom.h>
#include <TCanvas.h>

class MakeHist
{
    TTree* data;
    int fpga1;
    float mass, phi, costh, true_phi, true_costh;
public:
    int events;
    TH2D* true_hist;
    TH2D* reco_hist;
    MakeHist();
    virtual ~MakeHist(){};
    void Init(TString tree_name);
    void FillHist(int ev1, int ev2, double lambda, double mu, double nu);
};


class MakeTree
{
    double true_hist[12][12];
    double true_error[12][12];
    double reco_hist[12][12];
    double reco_error[12][12];
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
