//////////////
//// CUTS ////
//////////////
double getBeamOffset(int rs){
   return ( rs == 57 || rs == 59 ) ? 0.4 : 1.6;
}

double getPedestal(int spillID){
   if( spillID < 450000 )
      return 36.2;
   return 32.6;
}

bool dimuonIsValid_2111_v42(SRecDimuon dimuon, SRecEvent* recEvent, int rs, bool looseMode = false){
   double beamOffset = getBeamOffset(rs);

   SRecTrack track1 = recEvent->getTrack(dimuon.trackID_pos);
   SRecTrack track2 = recEvent->getTrack(dimuon.trackID_neg);

   TLorentzVector vtx_mom = dimuon.p_pos + dimuon.p_neg;

   if( ! ( fabs(dimuon.vtx.X()    ) < .25 ) ) return false;
   if( ! ( fabs(dimuon.vtx.Y()-beamOffset) < .22 ) ) return false;


   if( ! (      dimuon.vtx.Z()      < -  5) ) return false;
   if( ! (      dimuon.vtx.Z()      > -280) ) return false;
   // if( ! (      dimuon.vtx.Z()      > 0) ) return false; /// dump cut
   

   if( ! ( fabs(       vtx_mom.X()    ) < 1.8 ) ) return false;
   if( ! ( fabs(       vtx_mom.Y()    ) < 2.0 ) ) return false;

   if( ! ( fabs(dimuon.costh          ) < 0.5 ) ) return false;
   if( ! (             vtx_mom.Z()      < 116 ) ) return false;
   if( ! (             vtx_mom.Z()      >  38 ) ) return false;

   if( ! ( pow(   vtx_mom.X(), 2) + pow(       vtx_mom.Y()       , 2) < 5   ) ) return false;
   if( ! ( pow(dimuon.vtx.X(), 2) + pow(dimuon.vtx.Y()-beamOffset, 2) <  .06) ) return false;

   // if( ! ( dimuon.mass < 8.8 ) ) return false;
   // if( !looseMode ){

// temporary out
   // if( ! ( dimuon.mass > 4.5 ) ) return false;


   // }else{
   //    if( ! ( dimuon.mass > 4.1 ) ) return false;
   // }


   if( ! ( dimuon.xF <  .95 ) ) return false;
   if( !looseMode ){
      if( ! ( dimuon.xF > -.10 ) ) return false;
   }else{
      if( ! ( dimuon.xF > -.10 ) ) return false;
   }
   if( ! ( dimuon.x2 >  .05 ) ) return false; /// drellyan cut
   // if( ! ( dimuon.x2 >  .005 ) ) return false; /// jpsi cut
   if( ! ( dimuon.x2 <   .55 ) ) return false;

   double trackSeparation = track1.getZVertex() - track2.getZVertex();
   if( ! ( fabs(dimuon.costh          ) <  .5 ) ) return false;
   if( ! ( fabs(       trackSeparation) < 270 ) ) return false;

   if( ! ( dimuon.chisq_kf < 18 ) ) return false;

   return true;
}

bool trackIsValid_2111_v42(SRecTrack track, int rs){
   double beamOffset = getBeamOffset(rs);
   // double beamOffset = 0; //getBeamOffset(rs);
   // if( ! ( track.chisq_target < 15 ) ) return false;
   if( ! ( track.getChisqTarget() < 15 ) ) return false;
  
   if( ! ( track.getMomentumVecSt1().Z() >  9 ) ) return false;
   if( ! ( track.getMomentumVecSt1().Z() < 75 ) ) return false;

   if( ! ( track.getNHits() > 13 ) ) return false;

   if( ! ( pow(track.getTargetPos().X(), 2) + pow(track.getTargetPos().Y()-beamOffset, 2) <  320 ) ) return false;
   if( ! ( pow(track.getDumpPos()  .X(), 2) + pow(track.getDumpPos  ().Y()-beamOffset, 2) < 1100 ) ) return false;
   if( ! ( pow(track.getDumpPos()  .X(), 2) + pow(track.getDumpPos  ().Y()-beamOffset, 2) >   16 ) ) return false; // drellyan cut
   // if( ! ( pow(track.getDumpPos()  .X(), 2) + pow(track.getDumpPos  ().Y()-beamOffset, 2) >    8 ) ) return false;   //jpsi cut

   /// target cuts
   if( ! ( track.getChisqTarget() < 1.5 * track.getChisqUpstream() ) ) return false;
   if( ! ( track.getChisqTarget() < 1.5 * track.getChisqDump()     ) ) return false;

   /// target cuts
   if( ! ( track.getVertex().Z() < -  5 ) ) return false;
   if( ! ( track.getVertex().Z() > -320 ) ) return false;
   // if( ! ( track.getVertex().Z() > 0 ) ) return false;

   if( ! ( track.getChisq() / ( track.getNHits() - 5 ) < 12 ) ) return false;

   // if( ! ( track.getPositionVecSt1().Y() / track.getPositionVecSt3().Y() < 1. ) ) return false;
   if( ! ( (track.getPositionVecSt1().Y()-beamOffset) / (track.getPositionVecSt3().Y()-beamOffset) < 1. ) ) return false;
   // if( ! ( (track.posSt[0].Y()-beamOffset) / (track.posSt[2].Y()-beamOffset) < 1. ) ) return false;

   if( ! fabs(( fabs( track.getMomentumVecSt1().X() - track.getMomentumVecSt3().X() ) - .416) < .008 ) ) return false;
   if( ! ( fabs( track.getMomentumVecSt1().Y() - track.getMomentumVecSt3().Y() )        < .008 ) ) return false;
   if( ! ( fabs( track.getMomentumVecSt1().Z() - track.getMomentumVecSt3().Z() )        < .08  ) ) return false;

   // if( ! ( track.getPositionVecSt1().Y() * track.getPositionVecSt3().Y() > 0. ) ) return false;
   if( ! ( (track.getPositionVecSt1().Y()-beamOffset) * (track.getPositionVecSt3().Y()-beamOffset) > 0. ) ) return false;
   // if( ! ( (track.posSt[0].Y()-beamOffset) * (track.posSt[2].Y()-beamOffset) > 0. ) ) return false;

   if( ! ( fabs( track.getMomentumVecSt1().Y() ) > .02 ) ) return false;

   return true;
}

bool tracksAreValid_2111_v42(SRecTrack trackP, SRecTrack trackN, SRecDimuon dimuon, int rs){
   double beamOffset = getBeamOffset(rs);
   // beamOffset = 0;
   if( ! ( fabs( trackP.getChisqTarget() + trackN.getChisqTarget() - dimuon.chisq_kf ) < 2 ) ) return false;

   // if( ! ( trackP.getPositionVecSt3().Y() * trackN.getPositionVecSt3().Y() < 0 ) ) return false;
   if( ! ( (trackP.getPositionVecSt3().Y()-beamOffset) * (trackN.getPositionVecSt3().Y()-beamOffset) < 0 ) ) return false;

   if( ! ( trackP.getNHits()           + trackN.getNHits()           > 29 ) ) return false;
   if( ! ( trackP.getNHitsInStation(1) + trackN.getNHitsInStation(1) >  8 ) ) return false;
   if( ! ( fabs( trackP.getPositionVecSt1().X() + trackN.getPositionVecSt1().X() ) < 42 ) ) return false;
   // this should not be in the loose mode

   return true;
}

bool tightMode_2111_v42(SRecTrack trackP, SRecTrack trackN, SRecDimuon dimuon){
   if( dimuon.mass        <= 4.3 ) return false;
   if( dimuon.vtx_pos.Z() >= -60 ) return false;
   if( trackP.getChisqTarget() >= trackP.getChisqDump() ) return false;   
   if( trackN.getChisqTarget() >= trackN.getChisqDump() ) return false;   
   if( fabs(trackP.getChisqTarget() + trackN.getChisqTarget() - dimuon.chisq_kf) >= 2 ) return false;
   if( ! ( trackP.getPositionVecSt1().X() + trackN.getPositionVecSt1().X() < 32 ) ) return false;
   return true;
}

