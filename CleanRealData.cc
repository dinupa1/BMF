//
// clean real data
// 04-11-2023
// dinupa3@gmail.com
//
// a simple script to clean the real data
//
#include <TString.h>
#include <iostream>

using namespace std;

void CleanRealData(TString tree, TString file, TString output)
{
  ROOT::RDataFrame df(tree.Data(), file.Data());

//  apply physics cuts
  auto df1 = df.Filter("mass > 4.5 && mass < 8.8 && xF < 0.95 && xF > -0.1 && xT > 0.05 && xT < 0.55 && abs(costh) < 0.5 && D1 < 200.");
//  apply positive track cuts
  auto df2 = df1.Filter("chisq1_target < 15. && pz1_st1 > 9. && pz1_st1 < 75. && nHits1 > 13 && chisq1_target < 1.5* chisq1_upstream && chisq1_target < 1.5*chisq1_dump && z1_v < -5. && z1_v > -320. && chisq1/(nHits1-5) < 12. && y1_st1/y1_st3 < 1. && abs( abs(px1_st1-px1_st3)-0.416) < 0.008 && abs(py1_st1-py1_st3) < 0.008 && abs(pz1_st1-pz1_st3) < 0.08 && y1_st1*y1_st3 > 0. && abs(py1_st1)>0.02");
//  apply negative track cuts
  auto df3 = df2.Filter("chisq2_target < 15. && pz2_st1 > 9. && pz2_st1 < 75. && nHits2 > 13 && chisq2_target < 1.5* chisq2_upstream && chisq2_target < 1.5*chisq2_dump && z2_v < -5. && z2_v > -320. && chisq2/(nHits2-5) < 12 && y2_st1/y2_st3 < 1. && abs(abs(px2_st1-px2_st3)-0.416) < 0.008 && abs(py2_st1-py2_st3) < 0.008 && abs(pz2_st1-pz2_st3) < 0.08 && y2_st1*y2_st3 > 0 && abs(py2_st1)>0.02");
//  apply dimuon cuts
  auto df4 = df3.Filter("abs(dx) < 0.25 && dz > -280. && dz < -5. && abs(dpx) < 1.8 && abs(dpy) < 2. && dpx*dpx + dpy*dpy < 5. && dpz > 38. && dpz < 116. && abs(trackSeparation) < 270. && chisq_dimuon < 18. && abs(chisq1_target+chisq2_target-chisq_dimuon) < 2 && y1_st3*y2_st3 < 0. && nHits1 + nHits2 > 29 && nHits1St1 + nHits2St1 > 8. && abs(x1_st1+x2_st1)<42.");

  df4.Snapshot("tree", output.Data(), {"mass", "pT", "xF", "phi", "costh"});

}

