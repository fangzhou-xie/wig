
#include <RcppArmadillo.h>

// [[Rcpp::export]]
arma::vec softmin_row(arma::mat &m, const double eps) {
  arma::vec v(m.n_rows, arma::fill::zeros);
  arma::vec mi;
  for (size_t i = 0; i < m.n_rows; ++i) {
    mi = m.row(i).t();
    v(i) = mi.min() - eps * log(accu(exp(-(mi-mi.min())/eps)));
  }
  return v;
}

// [[Rcpp::export]]
arma::vec softmin_col(arma::mat &m, const double eps) {
  arma::vec v(m.n_cols, arma::fill::zeros);
  arma::vec mj;
  for (size_t j = 0; j < m.n_rows; ++j) {
    mj = m.col(j);
    v(j) = mj.min() - eps * log(accu(exp(-(mj-mj.min())/eps)));
  }
  return v;
}
