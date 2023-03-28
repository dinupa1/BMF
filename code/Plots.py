#
# plot the generated angular coef.
# 25March2023
# dinupa3@gmail.com
#

import uproot
import numpy as np
import awkward as ak

import ROOT

# custom style
# ROOT.gStyle.SetHistFillColor(422)
ROOT.gStyle.SetHistLineStyle(2)
ROOT.gStyle.SetHistLineWidth(2)
# ROOT.gStyle.SetStatFontSize(0.1)
ROOT.gStyle.SetStatY(0.85)
ROOT.gStyle.SetStatX(0.85)
ROOT.gStyle.SetStatW(0.3)
ROOT.gStyle.SetStatH(0.3)
ROOT.gStyle.SetOptStat("emr")
ROOT.gStyle.SetStatBorderSize(0)
ROOT.gStyle.SetStatFont(42)
ROOT.gStyle.SetCanvasDefH(800)
ROOT.gStyle.SetCanvasDefW(800)
ROOT.gStyle.SetLegendBorderSize(0)
ROOT.gStyle.SetLegendBorderSize(0)
# ROOT.gStyle.SetHistMinimumZero("kTrue")

can = ROOT.TCanvas()

# plot test results
# --------------------------------------------------------------------
tree1 = uproot.open("result.root:test1")
test1 = tree1.arrays(["lambda", "mu", "nu"], library="np")

lambda1 = ROOT.TH1D("lambda1", "; lambda [a.u.]; counts [a.u.]", 11, -0.3, 0.9)
mu1 = ROOT.TH1D("mu1", "; mu [a.u.]; counts [a.u.]", 11, -0.1, 0.6)
nu1 = ROOT.TH1D("nu1", "; nu [a.u.]; counts [a.u.]", 11, -0.1, 0.6)

[lambda1.Fill(m) for m in test1["lambda"]]
[mu1.Fill(m) for m in test1["mu"]]
[nu1.Fill(m) for m in test1["nu"]]

lambda1.Draw()
can.SaveAs("imgs/lambda1.png")

mu1.Draw()
can.SaveAs("imgs/mu1.png")

nu1.Draw()
can.SaveAs("imgs/nu1.png")


# # --------------------------------------------------------------------------
# tree2 = uproot.open("result.root:test2")
# test2 = tree2.arrays(["lambda", "mu", "nu"], library="np")
#
# lambda2 = ROOT.TH1D("lambda2", "; lambda [a.u.]; counts [a.u.]", 11, -0.6, 0.2)
# mu2 = ROOT.TH1D("mu2", "; mu [a.u.]; counts [a.u.]", 11, 0.2, 0.6)
# nu2 = ROOT.TH1D("nu2", "; nu [a.u.]; counts [a.u.]", 11, 0.3, 0.6)
#
# [lambda2.Fill(m) for m in test2["lambda"]]
# [mu2.Fill(m) for m in test2["mu"]]
# [nu2.Fill(m) for m in test2["nu"]]
#
# lambda2.Draw()
# can.SaveAs("imgs/lambda2.png")
#
# mu2.Draw()
# can.SaveAs("imgs/mu2.png")
#
# nu2.Draw()
# can.SaveAs("imgs/nu2.png")
#
#
# # --------------------------------------------------------------------------
# tree3 = uproot.open("result.root:test3")
# test3 = tree3.arrays(["lambda", "mu", "nu"], library="np")
#
# lambda3 = ROOT.TH1D("lambda3", "; lambda [a.u.]; counts [a.u.]", 11, 0.0, 1.0)
# mu3 = ROOT.TH1D("mu3", "; mu [a.u.]; counts [a.u.]", 11, -0.2, 0.4)
# nu3 = ROOT.TH1D("nu3", "; nu [a.u.]; counts [a.u.]", 11, -0.5, -0.2)
#
# [lambda3.Fill(m) for m in test3["lambda"]]
# [mu3.Fill(m) for m in test3["mu"]]
# [nu3.Fill(m) for m in test3["nu"]]
#
# lambda3.Draw()
# can.SaveAs("imgs/lambda3.png")
#
# mu3.Draw()
# can.SaveAs("imgs/mu3.png")
#
# nu3.Draw()
# can.SaveAs("imgs/nu3.png")
#
#
# # --------------------------------------------------------------------------
# tree4 = uproot.open("result.root:test4")
# test4 = tree4.arrays(["lambda", "mu", "nu"], library="np")
#
# lambda4 = ROOT.TH1D("lambda4", "; lambda [a.u.]; counts [a.u.]", 11, -0.9, 0.0)
# mu4 = ROOT.TH1D("mu4", "; mu [a.u.]; counts [a.u.]", 11, -0.1, 0.4)
# nu4 = ROOT.TH1D("nu4", "; nu [a.u.]; counts [a.u.]", 11, 0.1, 0.4)
#
# [lambda4.Fill(m) for m in test4["lambda"]]
# [mu4.Fill(m) for m in test4["mu"]]
# [nu4.Fill(m) for m in test4["nu"]]
#
# lambda4.Draw()
# can.SaveAs("imgs/lambda4.png")
#
# mu4.Draw()
# can.SaveAs("imgs/mu4.png")
#
# nu4.Draw()
# can.SaveAs("imgs/nu4.png")