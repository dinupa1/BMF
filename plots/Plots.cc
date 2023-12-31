/*
* dinupa3@gmail.com
*/

#include "src/PlotHist.cc"

using namespace std;

void Plots()
{
	auto ph = new PlotHist();
	ph->DrawResolution();
	ph->Print();
}