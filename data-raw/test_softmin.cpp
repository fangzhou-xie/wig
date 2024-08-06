

#include <RcppArmadillo.h>

// [[Rcpp::plugins(cpp17)]]

// #include "utils.h" // softmin_col, softmin_row
// [x] checked
double softmin1(arma::vec z, double eps) {
  return z.min() - eps * log(accu(exp(-(z-z.min())/eps)));
}

double softmin2(arma::vec &z, double eps) {
  return z.min() - eps * log(accu(exp(-(z-z.min())/eps)));
}

// [[Rcpp::export]]
arma::vec softmin1_col(arma::mat &S, const double eps) {
  arma::vec S_col(S.n_cols, arma::fill::zeros);
  for (size_t j = 0; j < S.n_cols; ++j) {
    S_col(j) = softmin1(S.col(j), eps);
  }
  return S_col;
}

// [[Rcpp::export]]
arma::vec softmin2_col(arma::mat &S, const double eps) {
  arma::vec S_col(S.n_cols, arma::fill::zeros);
  for (size_t j = 0; j < S.n_cols; ++j) {
    arma::vec S_j = S.col(j);
    S_col(j) = softmin2(S_j, eps);
  }
  return S_col;
}


// [[Rcpp::export]]
int test_var1(arma::mat &m) {
  arma::mat 
}
