//
// plot the generated angular coef.
// 25March2023
// dinupa3@gmail.com
//

#include <TH1D.h>
#include <TCanvas.h>
#include <iostream>

using ROOT::RDataFrame;

using namespace std;

void PlotResult()
{
//  custom style
  gStyle->SetHistLineStyle(2);
  gStyle->SetHistLineWidth(2);
  gStyle->SetStatY(0.85);
  gStyle->SetStatX(0.85);
  gStyle->SetStatW(0.3);
  gStyle->SetStatH(0.3);
  gStyle->SetOptStat("emr");
  gStyle->SetStatBorderSize(0);
  gStyle->SetStatFont(42);
  gStyle->SetCanvasDefH(800);
  gStyle->SetCanvasDefW(800);
  gStyle->SetLegendBorderSize(0);
  gStyle->SetLegendBorderSize(0);

  auto can = new TCanvas();

//
// plot test results
//

  RDataFrame df("result", "result.root");

  auto Hlambda = df.Fill<float>(TH1D("Hlambda", "; lambda [a.u.]; count [a.u.]", 10, -1.2, -0.9), {"lambda"});
  Hlambda->Draw();
  can->SaveAs("imgs/lambda_xF2.png");

  auto Hmu = df.Fill<float>(TH1D("Hmu", "; mu [a.u.]; count [a.u.]", 10, -0.2, 0.2), {"mu"});
  Hmu->Draw();
  can->SaveAs("imgs/mu_xF2.png");

  auto Hnu = df.Fill<float>(TH1D("Hnu", "; nu [a.u.]; count [a.u.]", 10, 0.0, 0.3), {"nu"});
  Hnu->Draw();
  can->SaveAs("imgs/nu_xF2.png");

}