

#include <boost/math/special_functions/bessel.hpp>
#include <ctime>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <limits>
#include <cassert>
#include <cstdio>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <string>
#include <vector>
#include <cctype>
#include <complex>
#include <functional>
#include <random>


//You should choose NX the power of 2.
#define NX  256
//you cannot use LATTICE_SIZE =/ 8, because the matrix V are generated in this size
#define LATTICE_SIZE  8

#define M_PI  3.141592653589793238462643383
#define Nc 3
#define ADJNc 8
#define ALPHA_S	0.3
#define P_UPPER 5.0

#define Round_Proton

extern const double Rp = 1.0;
extern const double R_CQ = Rp / 3.0;
extern const double g2_mu_Rp = 30.0;
extern const double m_Rp = 2.0;
extern const double mass = m_Rp / Rp;


double modified_bessel1(const double x)
{
	double x_times_k_1 = 0;
	if (abs(x) > 900.0) {
		return sqrt(M_PI / x / 2.0)*exp(-x);
	}
	else if (abs(x) < 1.0e-30)
	{
		return 1.0 / x;
	}
	else {
		return boost::math::cyl_bessel_k(1, x);
	}
}


double modified_bessel0(const double x)
{
	double x_times_k_1 = 0;
	if (abs(x) > 900.0) {
		return sqrt(M_PI / x / 2.0)*exp(-x);
	}
	else if (abs(x) < 1.0e-30)
	{
		return log(x / 2.0);
	}
	else {
		return boost::math::cyl_bessel_k(0, x);
	}
}

void MV_model_calculation_of_T_matrix()
{
	double impact_parameter = 1.0*NX / LATTICE_SIZE;

	std::vector<double> D_matrix(NX / 2, 0);

	double h = 1.0*LATTICE_SIZE / NX;
	std::vector<double> relative_distance(NX / 2, 0);
	for (int re = 0; re < NX / 2; ++re) {
		relative_distance[re] = 2.0*re*h + h;
	}

	std::vector<double> x(NX*NX, 0), y(NX*NX, 0);
	double   xmax = h *NX / 2.0, xmin = -h*NX / 2.0, ymin = -h*NX / 2.0;
	for (int j = 0; j < NX; j++) {
		for (int i = 0; i < NX; i++)
		{
			x[NX*j + i] = xmin + i*h;
			y[NX*j + i] = ymin + j*h;
		}
	}

#pragma omp parallel for num_threads(6)
	for (int re = 0; re < NX / 2; re++) {

		for (int j = 0; j < NX; j++) {
			for (int i = 0; i < NX; i++) {
				double simpson1 = 1.0;
				double simpson2 = 1.0;
				if (j == 0 || j == NX - 1) {
					simpson1 = 1.0 / 3.0;
				}
				else if (j % 2 == 0) {
					simpson1 = 2.0 / 3.0;
				}
				else {
					simpson1 = 4.0 / 3.0;
				}

				if (i == 0 || i == NX - 1) {
					simpson2 = 1.0 / 3.0;
				}
				else if (i % 2 == 0) {
					simpson2 = 2.0 / 3.0;
				}
				else {
					simpson2 = 4.0 / 3.0;
				}
				if (abs((x[NX*j + i] - impact_parameter*h - relative_distance[re] / 2.0)*(x[NX*j + i] - impact_parameter*h - relative_distance[re] / 2.0)
					+ y[NX*j + i] * y[NX*j + i]) < 1.0e-16
					|| abs((x[NX*j + i] - impact_parameter*h + relative_distance[re] / 2.0)*(x[NX*j + i] - impact_parameter*h + relative_distance[re] / 2.0)
						+ y[NX*j + i] * y[NX*j + i]) < 1.0e-16) {

					D_matrix[re] += 0;
				}
				else {
					D_matrix[re] += simpson1*simpson2*exp(-(x[NX*j + i] * x[NX*j + i] + y[NX*j + i] * y[NX*j + i]) / 1.0 / Rp / Rp)
						*(modified_bessel0(mass
							*sqrt((x[NX*j + i] - impact_parameter*h - relative_distance[re] / 2.0)*(x[NX*j + i] - impact_parameter*h - relative_distance[re] / 2.0)
								+ y[NX*j + i] * y[NX*j + i]))
							- modified_bessel0(mass
								*sqrt((x[NX*j + i] - impact_parameter*h + relative_distance[re] / 2.0)*(x[NX*j + i] - impact_parameter*h + relative_distance[re] / 2.0)
									+ y[NX*j + i] * y[NX*j + i])))
						*(modified_bessel0(mass
							*sqrt((x[NX*j + i] - impact_parameter*h - relative_distance[re] / 2.0)*(x[NX*j + i] - impact_parameter*h - relative_distance[re] / 2.0)
								+ y[NX*j + i] * y[NX*j + i]))
							- modified_bessel0(mass
								*sqrt((x[NX*j + i] - impact_parameter*h + relative_distance[re] / 2.0)*(x[NX*j + i] - impact_parameter*h + relative_distance[re] / 2.0)
									+ y[NX*j + i] * y[NX*j + i])));

				}
			}
		}
	}

	std::ostringstream ofilename_i2;
#ifdef Round_Proton

#ifdef v_Parameter
	ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\output\\D_matrix_MV_position_RP_" << impact_parameter << "_NX_" << NX
		<< "_vPara_" << v_Parameter << ".txt";
#else
	ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\output\\D_matrix_MV_position_RP_" << impact_parameter << "_NX_" << NX << ".txt";
#endif

#else

#ifdef v_Parameter
	ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\output\\D_matrix_MV_position_" << impact_parameter << "_NX_" << NX
		<< "_vPara_" << v_Parameter << ".txt";
#else
	ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\output\\D_matrix_MV_position_" << impact_parameter << "_NX_" << NX << ".txt";
#endif

#endif

	std::ofstream ofs_res_i2(ofilename_i2.str().c_str());

	ofs_res_i2 << "#relative distance \t D_matrix \t position " << impact_parameter << "\n";

	double exp_coeff = -g2_mu_Rp*g2_mu_Rp*(Nc*Nc - 1.0) / 4.0 / Nc / (2.0*M_PI) / (2.0*M_PI) / (1.0*M_PI);

	for (int re = 0; re < NX / 2; ++re) {
		ofs_res_i2 << relative_distance[re] << "\t" << 1.0 - exp(exp_coeff*h*h*D_matrix[re]) << "\t" << D_matrix[re] << "\n";
	}


}

void g4times_Gamma_functions(double position_x1, double position_x2, double position_y1, double position_y2, double* gamma_functions)
{
	double dd_gamma = 0;
	double d_gamma_d_gamma_x1 = 0;
	double d_gamma_d_gamma_x2 = 0;
	double d_gamma_d_gamma_y1 = 0;
	double d_gamma_d_gamma_y2 = 0;
	double Gamma = 0;


	double h = 1.0*LATTICE_SIZE / NX;


	std::vector<double> x(NX*NX, 0), y(NX*NX, 0);
	double   xmax = h *NX / 2.0, xmin = -h*NX / 2.0, ymin = -h*NX / 2.0;
	for (int j = 0; j < NX; j++) {
		for (int i = 0; i < NX; i++)
		{
			x[NX*j + i] = xmin + i*h;
			y[NX*j + i] = ymin + j*h;
		}
	}


	for (int j = 0; j < NX; j++) {
		for (int i = 0; i < NX; i++) {
			double simpson1 = 1.0;
			double simpson2 = 1.0;
			if (j == 0 || j == NX - 1) {
				simpson1 = 1.0 / 3.0;
			}
			else if (j % 2 == 0) {
				simpson1 = 2.0 / 3.0;
			}
			else {
				simpson1 = 4.0 / 3.0;
			}

			if (i == 0 || i == NX - 1) {
				simpson2 = 1.0 / 3.0;
			}
			else if (i % 2 == 0) {
				simpson2 = 2.0 / 3.0;
			}
			else {
				simpson2 = 4.0 / 3.0;
			}
			if (abs((position_x1 - x[NX*j + i])*(position_x1 - x[NX*j + i])
				+ (position_x2 - y[NX*j + i])*(position_x2 - y[NX*j + i])) < 1.0e-16
				|| abs((position_y1 - x[NX*j + i])*(position_y1 - x[NX*j + i])
					+ (position_y2 - y[NX*j + i])*(position_y2 - y[NX*j + i])) < 1.0e-16) {

				dd_gamma += 0;
				d_gamma_d_gamma_x1 += 0;
				d_gamma_d_gamma_x2 += 0;
				d_gamma_d_gamma_y1 += 0;
				d_gamma_d_gamma_y2 += 0;
				Gamma += 0;
			}
			else {
				dd_gamma += simpson1*simpson2*exp(-(x[NX*j + i] * x[NX*j + i] + y[NX*j + i] * y[NX*j + i]) / 1.0 / Rp / Rp)
					*((position_x1 - x[NX*j + i])*(position_y1 - x[NX*j + i])
						+ (position_x2 - y[NX*j + i])*(position_y2 - y[NX*j + i]))
					/ sqrt((position_x1 - x[NX*j + i])*(position_x1 - x[NX*j + i])
						+ (position_x2 - y[NX*j + i])*(position_x2 - y[NX*j + i]))
					/ sqrt((position_y1 - x[NX*j + i])*(position_y1 - x[NX*j + i])
						+ (position_y2 - y[NX*j + i])*(position_y2 - y[NX*j + i]))
					*modified_bessel1(mass
						*sqrt((position_x1 - x[NX*j + i])*(position_x1 - x[NX*j + i])
							+ (position_x2 - y[NX*j + i])*(position_x2 - y[NX*j + i])))
					*modified_bessel1(mass
						*sqrt((position_y1 - x[NX*j + i])*(position_y1 - x[NX*j + i])
							+ (position_y2 - y[NX*j + i])*(position_y2 - y[NX*j + i])));


				d_gamma_d_gamma_x1 += simpson1*simpson2*exp(-(x[NX*j + i] * x[NX*j + i] + y[NX*j + i] * y[NX*j + i]) / 1.0 / Rp / Rp)
					*(position_x1 - x[NX*j + i])
					/ sqrt((position_x1 - x[NX*j + i])*(position_x1 - x[NX*j + i])
						+ (position_x2 - y[NX*j + i])*(position_x2 - y[NX*j + i]))
					*modified_bessel1(mass
						*sqrt((position_x1 - x[NX*j + i])*(position_x1 - x[NX*j + i])
							+ (position_x2 - y[NX*j + i])*(position_x2 - y[NX*j + i])))
					*(-modified_bessel0(mass
						*sqrt((position_x1 - x[NX*j + i])*(position_x1 - x[NX*j + i])
							+ (position_x2 - y[NX*j + i])*(position_x2 - y[NX*j + i])))
						+ modified_bessel0(mass
							*sqrt((position_y1 - x[NX*j + i])*(position_y1 - x[NX*j + i])
								+ (position_y2 - y[NX*j + i])*(position_y2 - y[NX*j + i]))));
				d_gamma_d_gamma_x2 += simpson1*simpson2*exp(-(x[NX*j + i] * x[NX*j + i] + y[NX*j + i] * y[NX*j + i]) / 1.0 / Rp / Rp)
					*(position_x2 - y[NX*j + i])
					/ sqrt((position_x1 - x[NX*j + i])*(position_x1 - x[NX*j + i])
						+ (position_x2 - y[NX*j + i])*(position_x2 - y[NX*j + i]))
					*modified_bessel1(mass
						*sqrt((position_x1 - x[NX*j + i])*(position_x1 - x[NX*j + i])
							+ (position_x2 - y[NX*j + i])*(position_x2 - y[NX*j + i])))
					*(-modified_bessel0(mass
						*sqrt((position_x1 - x[NX*j + i])*(position_x1 - x[NX*j + i])
							+ (position_x2 - y[NX*j + i])*(position_x2 - y[NX*j + i])))
						+ modified_bessel0(mass
							*sqrt((position_y1 - x[NX*j + i])*(position_y1 - x[NX*j + i])
								+ (position_y2 - y[NX*j + i])*(position_y2 - y[NX*j + i]))));
				d_gamma_d_gamma_y1 += simpson1*simpson2*exp(-(x[NX*j + i] * x[NX*j + i] + y[NX*j + i] * y[NX*j + i]) / 1.0 / Rp / Rp)
					*(position_y1 - x[NX*j + i])
					/ sqrt((position_y1 - x[NX*j + i])*(position_y1 - x[NX*j + i])
						+ (position_y2 - y[NX*j + i])*(position_y2 - y[NX*j + i]))
					*modified_bessel1(mass
						*sqrt((position_y1 - x[NX*j + i])*(position_y1 - x[NX*j + i])
							+ (position_y2 - y[NX*j + i])*(position_y2 - y[NX*j + i])))
					*(-modified_bessel0(mass
						*sqrt((position_y1 - x[NX*j + i])*(position_y1 - x[NX*j + i])
							+ (position_y2 - y[NX*j + i])*(position_y2 - y[NX*j + i])))
						+ modified_bessel0(mass
							*sqrt((position_x1 - x[NX*j + i])*(position_x1 - x[NX*j + i])
								+ (position_x2 - y[NX*j + i])*(position_x2 - y[NX*j + i]))));
				d_gamma_d_gamma_y2 += simpson1*simpson2*exp(-(x[NX*j + i] * x[NX*j + i] + y[NX*j + i] * y[NX*j + i]) / 1.0 / Rp / Rp)
					*(position_y2 - y[NX*j + i])
					/ sqrt((position_y1 - x[NX*j + i])*(position_y1 - x[NX*j + i])
						+ (position_y2 - y[NX*j + i])*(position_y2 - y[NX*j + i]))
					*modified_bessel1(mass
						*sqrt((position_y1 - x[NX*j + i])*(position_y1 - x[NX*j + i])
							+ (position_y2 - y[NX*j + i])*(position_y2 - y[NX*j + i])))
					*(-modified_bessel0(mass
						*sqrt((position_y1 - x[NX*j + i])*(position_y1 - x[NX*j + i])
							+ (position_y2 - y[NX*j + i])*(position_y2 - y[NX*j + i])))
						+ modified_bessel0(mass
							*sqrt((position_x1 - x[NX*j + i])*(position_x1 - x[NX*j + i])
								+ (position_x2 - y[NX*j + i])*(position_x2 - y[NX*j + i]))));

				Gamma += -simpson1*simpson2*exp(-(x[NX*j + i] * x[NX*j + i] + y[NX*j + i] * y[NX*j + i]) / 1.0 / Rp / Rp)
					*(modified_bessel0(mass
						*sqrt((position_x1 - x[NX*j + i])*(position_x1 - x[NX*j + i])
							+ (position_x2 - y[NX*j + i])*(position_x2 - y[NX*j + i])))
						- modified_bessel0(mass
							*sqrt((position_y1 - x[NX*j + i])*(position_y1 - x[NX*j + i])
								+ (position_y2 - y[NX*j + i])*(position_y2 - y[NX*j + i]))))
					*(modified_bessel0(mass
						*sqrt((position_x1 - x[NX*j + i])*(position_x1 - x[NX*j + i])
							+ (position_x2 - y[NX*j + i])*(position_x2 - y[NX*j + i])))
						- modified_bessel0(mass
							*sqrt((position_y1 - x[NX*j + i])*(position_y1 - x[NX*j + i])
								+ (position_y2 - y[NX*j + i])*(position_y2 - y[NX*j + i]))));

			}

		}
	}

	gamma_functions[0] = g2_mu_Rp*g2_mu_Rp*dd_gamma*mass*mass / 2.0 / M_PI / 2.0 / M_PI*h*h;
	gamma_functions[1] = g2_mu_Rp*g2_mu_Rp*d_gamma_d_gamma_x1*mass / M_PI*h*h*g2_mu_Rp*g2_mu_Rp*d_gamma_d_gamma_y1*mass / M_PI*h*h
		+ g2_mu_Rp*g2_mu_Rp*d_gamma_d_gamma_x2*mass / M_PI*h*h*g2_mu_Rp*g2_mu_Rp*d_gamma_d_gamma_y2*mass / M_PI*h*h;
	gamma_functions[2] = g2_mu_Rp*g2_mu_Rp*Gamma / 2.0 / M_PI / 2.0 / M_PI*h*h;

}



void MV_model_calculation()
{
	double impact_parameter = 1.0;

	std::vector<double> D_matrix(NX / 2, 0), D_matrix2(NX / 2, 0), D_matrix3(NX / 2, 0);

	double h = 1.0*LATTICE_SIZE / NX;
	std::vector<double> relative_distance(NX / 2, 0);
	for (int re = 1; re <= NX / 2; ++re) {
		relative_distance[re - 1] = 2.0*re*h;
	}

	std::vector<double> x(NX*NX, 0), y(NX*NX, 0);
	double   xmax = h *NX / 2.0, xmin = -h*NX / 2.0, ymin = -h*NX / 2.0;
	for (int j = 0; j < NX; j++) {
		for (int i = 0; i < NX; i++)
		{
			x[NX*j + i] = xmin + i*h;
			y[NX*j + i] = ymin + j*h;
		}
	}

#pragma omp parallel for num_threads(6)
	for (int re = 1; re <= NX / 2; re++) {

		double impact_param = re*h;

		std::vector<double> temp_gamma(3, 0);

		g4times_Gamma_functions(impact_parameter+impact_param,0,
			impact_parameter - impact_param,0, temp_gamma.data());
		D_matrix[re-1] = temp_gamma[0];
		D_matrix2[re - 1] = temp_gamma[1];
		D_matrix3[re - 1] = temp_gamma[2];
	}

	std::ostringstream ofilename_i2;
#ifdef Round_Proton

#ifdef v_Parameter
	ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\output\\MV_position_RP_" << impact_parameter << "_NX_" << NX
		<< "_vPara_" << v_Parameter << ".txt";
#else
	ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\output\\MV_position_RP_" << impact_parameter << "_NX_" << NX << ".txt";
#endif

#else

#ifdef v_Parameter
	ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\output\\MV_position_" << impact_parameter << "_NX_" << NX
		<< "_vPara_" << v_Parameter << ".txt";
#else
	ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\output\\MV_position_" << impact_parameter << "_NX_" << NX << ".txt";
#endif

#endif

	std::ofstream ofs_res_i2(ofilename_i2.str().c_str());

	ofs_res_i2 << "#relative distance \t dd_gamma \t dgamma*dgamma \t Gamma " << impact_parameter << "\n";

	for (int re = 0; re < NX / 2; ++re) {
		ofs_res_i2 << relative_distance[re] << "\t" << (Nc*Nc-1.0)/2.0/Nc* D_matrix[re]
			<< "\t" << (Nc*Nc - 1.0) / 2.0 / Nc*(Nc*Nc - 1.0) / 2.0 / Nc/4.0*D_matrix2[re] 
			<< "\t" << (Nc*Nc - 1.0) / 2.0 / Nc/2.0* D_matrix3[re] << "\n";
	}


}



void MV_Wigner(int maxmom)
{
	std::vector<double> g4gammas(3, 0);
	std::vector<double> b_space, Wigner, EWigner, b_spaceS(NX / 2, 0), WignerS(maxmom*NX / 2, 0), EWignerS(maxmom*NX / 2, 0);

	double h = 1.0*LATTICE_SIZE / NX;

	std::vector<double> x(NX*NX, 0), y(NX*NX, 0);
	double   xmax = h *NX / 2.0, xmin = -h*NX / 2.0, ymin = -h*NX / 2.0;
	for (int j = 0; j < NX; j++) {
		for (int i = 0; i < NX; i++)
		{
			x[NX*j + i] = xmin + i*h;
			y[NX*j + i] = ymin + j*h;
		}
	}

	for (int mom = 0; mom < maxmom; mom++) {
		double momk = P_UPPER / maxmom*mom;


#pragma omp parallel for num_threads(6)
		for (int ip = 1; ip <= NX / 2; ip++) {
			double impact_parameter = ip*h;

			for (int j = 0; j < NX; j++) {
				for (int i = 0; i < NX; i++)
				{
					double simpson1 = 1.0;
					double simpson2 = 1.0;
					if (j == 0 || j == NX - 1) {
						simpson1 = 1.0 / 3.0;
					}
					else if (j % 2 == 0) {
						simpson1 = 2.0 / 3.0;
					}
					else {
						simpson1 = 4.0 / 3.0;
					}

					if (i == 0 || i == NX - 1) {
						simpson2 = 1.0 / 3.0;
					}
					else if (i % 2 == 0) {
						simpson2 = 2.0 / 3.0;
					}
					else {
						simpson2 = 4.0 / 3.0;
					}

					if (abs(x[NX*j + i] * x[NX*j + i] + y[NX*j + i] * y[NX*j + i]) < 1.0e-16) {
						WignerS[mom*NX / 2 + ip - 1] += 0;
						EWignerS[mom*NX / 2 + ip - 1] += 0;
					}
					else {
						g4times_Gamma_functions(impact_parameter + x[NX*j + i], y[NX*j + i],
							impact_parameter - x[NX*j + i], -y[NX*j + i], g4gammas.data());
						WignerS[mom*NX / 2 + ip - 1] += simpson1*simpson2
							*_j0(2.0*momk*sqrt(x[NX*j + i] * x[NX*j + i] + y[NX*j + i] * y[NX*j + i]))
							*((Nc*Nc - 1.0) / 2.0 / Nc * g4gammas[0]
								+ (Nc*Nc - 1.0) / 2.0 / Nc*(Nc*Nc - 1.0) / 2.0 / Nc / 4.0*g4gammas[1]
								)
							*exp(-x[NX*j + i] * x[NX*j + i] - y[NX*j + i] * y[NX*j + i])
							*exp((Nc*Nc - 1.0) / 2.0 / Nc / 2.0*g4gammas[2])
							;
						EWignerS[mom*NX / 2 + ip - 1] += simpson1*simpson2
							*_j0(2.0*momk*sqrt(x[NX*j + i] * x[NX*j + i] + y[NX*j + i] * y[NX*j + i]))
							*exp(-x[NX*j + i] * x[NX*j + i] - y[NX*j + i] * y[NX*j + i])
							*g4gammas[0] / g4gammas[2]
							* (exp(Nc / 2.0*g4gammas[2]) - 1.0);
					}

				}


			}

		}

		std::cout << "mom" << mom << "\n";
	}

	std::ostringstream ofilename_Wigner, ofilename_Wigner_all;
	ofilename_Wigner << "DP_WW_Wigner0_MV_diag_direct_NX_" << NX << "_size_" << LATTICE_SIZE
		<< "_real.txt";
	std::ofstream ofs_res_Wigner(ofilename_Wigner.str().c_str());

	ofs_res_Wigner << "#b \t momk \t DP \t WW \n";

	for (int mom = 0; mom < maxmom; mom++) {
		double momk = P_UPPER / maxmom*mom;
		for (int j = 0; j < NX / 2; j++) {

			ofs_res_Wigner << (j + 1)*h << "\t" << momk << "\t" << 2.0*Nc / ALPHA_S / M_PI / M_PI* WignerS[NX / 2 * mom + j] * h*h
				<< "\t" << 4.0*(Nc*Nc - 1) / 2.0 / Nc / ALPHA_S / M_PI / M_PI*EWignerS[NX / 2 * mom + j] * h*h << "\n";
		}
		ofs_res_Wigner << "\n";
	}


}



int main(void)
{
	int maxmom = 2;
	MV_model_calculation();

	return 0;
}