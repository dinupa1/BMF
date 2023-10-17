//
// dinupa3@gmail.com
//


#ifndef _H_FitHist_H_
#define _H_FitHist_H_


// ROOT
#include <TFile.h>
#include <TTree.h>
#include <TH2D.h>
#include <TH1D.h>
#include <TCanvas.h>
#include <TF2.h>
#include <TFitResultPtr.h>
#include <TF1.h>

class FitHist
{
    TTree* tree;
    int n_events;
    double true_hist[144];
    double true_error[144];
    double reco_hist[144];
    double reco_error[144];
    float pred_hist[144];
    double lambda, mu, nu;

public:
    FitHist();
    virtual ~FitHist(){};
    void Init();
    void DrawFits();
    void DrawResults();
};


#endif /* _H_FitHist_H_ */
