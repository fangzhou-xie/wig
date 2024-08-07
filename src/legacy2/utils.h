
#ifndef WIG_PACKAGE_CPP_UTILS_HEADER
#define WIG_PACKAGE_CPP_UTILS_HEADER

#include <RcppArmadillo.h>

arma::cube update_S(arma::mat &C, arma::mat &f, arma::mat &g, const double &eps);
arma::vec update_r(arma::mat &C, arma::vec &log_d, const double &eps);
arma::mat softmin_col(arma::cube &S, const double &eps);
arma::mat softmin_row(arma::cube &S, const double &eps);


#endif
