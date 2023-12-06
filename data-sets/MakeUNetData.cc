/*
* dinupa3@gmail.com
*/

#include<TFile.h>
#include<TTree.h>
#include<TH3D.h>
#include<TH2D.h>
#include<TRandom3.h>
#include<TString.h>
#include<iostream>

#include "src/MakeHist.cc"
#include "src/MakeTree.cc"

using namespace std;

void MakeUNetData(int train_events, int val_events, int test_events)
{
	TRandom3* event = new TRandom3();

	TFile* split_file = TFile::Open("split.root", "read");
	TTree* train_data = (TTree*)split_file->Get("train_data");
	TTree* val_data = (TTree*)split_file->Get("val_data");
	TTree* test_data = (TTree*)split_file->Get("test_data");

	TFile* unet_file = new TFile("unet.root", "recreate");

	MakeTree* train_tree = new MakeTree("train_tree");
	train_tree->FillTree(train_events, train_data, event);

	MakeTree* val_tree = new MakeTree("val_tree");
	val_tree->FillTree(val_events, val_data, event);

	MakeTree* test_tree = new MakeTree("test_tree");
	test_tree->FillTree(test_events, test_data, event);

	train_tree->tree->Write();
	val_tree->tree->Write();
	test_tree->tree->Write();

	unet_file->Close();
}