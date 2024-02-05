/*
 * dinupa3@gmail.com
 *
 */
#ifndef _TREE_DATA__HH_
#define _TREE_DATA__HH_

#include <TFile.h>
#include <TTree.h>
#include <TRandom3.h>
#include <TCanvas.h>
#include <TMath.h>
#include <TString.h>
#include <TH2D.h>
#include <vector>

using namespace std;

double PI = TMath::Pi();
int NBINS = 15;

int num_reco = 10000;
const int num_samples = 1000000;

double thetas[3];
double X_par[2];
double X_det[2];
double W_par[2];
double W_det[2];
double label;

double true_phi;
double true_costh;
double phi;
double costh;

TTree* save;

double lambda[num_samples];
double mu[num_samples];
double nu[num_samples];

TH2D* hPar;
TH2D* hDet;

TRandom3* event = new TRandom3();

double weight_fn(double lambda, double mu, double nu, double phi, double costh);

void make_tree(TString tname);

void fill_histo(TFile* inputs, TString tname, double lambda, double mu, double nu);

void fill_train_tree();

void fill_test_tree();

#endif /*_TREE_DATA__HH_*/
