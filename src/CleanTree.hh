/*
 * Standard cuts used in the analysis
 *
 */
#ifndef _CLEAN_TREE__HH_
#define _CLEAN_TREE__HH_

#include <TString.h>

double beamOffset = 1.6; // set the beam-offset value for rs

TString dimuonIsValid_2111_v42 = Form("fabs(dx) < 0.25 && fabs(dy - %f) < 0.22 && dz < -5. && dz > -280. && fabs(dpx) < 1.8 && fabs(dpy) < 2.0 && fabs(costh) < 0.5 && dpz < 116. && dpz > 38. && (dpx* dpx + dpy* dpy) < 5. && (dx* dx + (dy - %f)* (dy - %f)) < 0.06 && xF < 0.95 && xF > -0.10 && xT > 0.05 && xT < 0.55 && fabs(trackSeparation) < 270. && chisq_dimuon < 18.", beamOffset, beamOffset, beamOffset);

TString posTrackIsValid_2111_v42 = Form("chisq1_target < 15. && pz1_st1 > 9. && pz1_st1 < 75. && nHits1 > 13 && (x1_t* x1_t + (y1_t - %f)* (y1_t - %f)) < 320. && (x1_d* x1_d + (y1_d - %f)* (y1_d - %f)) < 1100. && (x1_d* x1_d + (y1_d - %f)* (y1_d - %f)) > 16. && chisq1_target < 1.5* chisq1_upstream && chisq1_target < 1.5* chisq1_dump && z1_v < -5. && z1_v > -320. && chisq1/(nHits1 - 5) < 12. && (y1_st1 - %f )/(y1_st3 - %f) < 1. && fabs(fabs(px1_st1 - px1_st3) - 0.416) < 0.008 && fabs(py1_st1 - py1_st3) < 0.008 && fabs(pz1_st1 - pz1_st3) < 0.08 && (y1_st1 - %f)* (y1_st3 - %f) > 0. && fabs(py1_st1) > 0.02", beamOffset, beamOffset, beamOffset, beamOffset, beamOffset, beamOffset, beamOffset, beamOffset, beamOffset, beamOffset);

TString negTrackIsValid_2111_v42 = Form("chisq2_target < 15. && pz2_st1 > 9. && pz2_st1 < 75. && nHits2 > 13 && (x2_t* x2_t + (y2_t - %f)* (y2_t - %f)) < 320. && (x2_d* x2_d + (y2_d - %f)* (y2_d - %f)) < 1100. && (x2_d* x2_d + (y2_d - %f)* (y2_d - %f)) > 16. && chisq2_target < 1.5* chisq2_upstream && chisq2_target < 1.5* chisq2_dump && z2_v < -5. && z2_v > -320. && chisq2/(nHits2 - 5) < 12. && (y2_st1 - %f )/(y2_st3 - %f) < 1. && fabs(fabs(px2_st1 - px2_st3) - 0.416) < 0.008 && fabs(py2_st1 - py2_st3) < 0.008 && fabs(pz2_st1 - pz2_st3) < 0.08 && (y2_st1 - %f)* (y2_st3 - %f) > 0. && fabs(py2_st1) > 0.02", beamOffset, beamOffset, beamOffset, beamOffset, beamOffset, beamOffset, beamOffset, beamOffset, beamOffset, beamOffset);

TString tracksAreValid_2111_v42 = Form("fabs(chisq1_target + chisq2_target - chisq_dimuon) < 2. && (y1_st3 - %f)* (y2_st3 - %f) < 0. && (nHits1 + nHits2) > 29 && (nHits1St1 + nHits2St1) > 8 && fabs(x1_st1 + x2_st1) < 42.", beamOffset, beamOffset);

TString tightMode_2111_v42 = "mass > 4.5 && dz < -60. && chisq1_target < chisq1_dump && chisq2_target < chisq2_dump && fabs(chisq1_target + chisq2_target - chisq_dimuon) < 2. && (x1_st1 + x2_st1) < 32.";

TString dinupa_cuts = "mass > 5.0 && xF > 0.";

TString all_cuts = Form("%s && %s && %s && %s && %s", dimuonIsValid_2111_v42.Data(), posTrackIsValid_2111_v42.Data(), negTrackIsValid_2111_v42.Data(), tracksAreValid_2111_v42.Data(), tightMode_2111_v42.Data());

#endif /*_CLEAN_TREE__HH_*/
