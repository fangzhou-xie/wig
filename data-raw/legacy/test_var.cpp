
#include <RcppArmadillo.h>

// [[Rcpp::export]]
int test_var1(arma::mat &m) {
  arma::vec S(m.n_rows, arma::fill::zeros);
  for (size_t j = 0; j < m.n_cols; ++j) {
    S = m.col(j);
  }
  return 0;
}

// [[Rcpp::export]]
int test_var2(arma::mat &m) {
  for (size_t j = 0; j < m.n_cols; ++j) {
    arma::vec Sj = m.col(j);
  }
  return 0;
}

// [[Rcpp::export]]
int test_var3(arma::mat &m) {
  arma::vec S;
  for (size_t j = 0; j < m.n_cols; ++j) {
    S = m.col(j);
  }
  return 0;
}
