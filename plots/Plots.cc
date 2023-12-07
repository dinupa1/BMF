/*
* dinupa3@gmail.com
*/

#include "src/PlotHist.cc"

using namespace std;

void Plots()
{
	auto ph = new PlotHist();
	/*
	* plot lambda
	*/
	ph->DrawResolution(0, "; #Delta #lambda [a.u.]; counts [a.u.]", -10., 10.);
	ph->DrawResolution(1, "; #Delta #lambda_{error} [a.u.]; counts [a.u.]", -10., 10.);

	/*
	* plot mu
	*/
	ph->DrawResolution(2, "; #Delta #mu [a.u.]; counts [a.u.]", -4., 4.);
	ph->DrawResolution(3, "; #Delta #mu_{error} [a.u.]; counts [a.u.]", -4., 4.);

	/*
	* plot nu
	*/
	ph->DrawResolution(4, "; #Delta #nu [a.u.]; counts [a.u.]", -2., 2.);
	ph->DrawResolution(5, "; #Delta #nu_{error} [a.u.]; counts [a.u.]", -2., 2.);

	ph->DrawHist();
}