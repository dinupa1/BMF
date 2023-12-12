/*
* dinupa3@gmail.com
*/

#ifndef _H_MakeTree_H_
#define _H_MakeTree_H_

#include<TTree.h>
#include<TH3D.h>
#include<TH2D.h>
#include<TString.h>
#include<TRandom3.h>
#include<iostream>

using namespace std;

class MakeTree
{
	float X_par[4][3]; // lambda, mu, nu, (and errors) in the particle level
	float X_det[4][10][10]; // phi vs. costheta, sin2theta vs. cosphi and sintheta2 vs. cos2phi

public:
	TTree* tree;
	MakeTree(TString outTree);
	virtual ~MakeTree(){};
	void FillTree(int n_events, TTree* data, TRandom3* event);
};
#endif /* _H_MakeTree_H_ */