
#include <RcppArmadillo.h>

double softmin(arma::vec z, double eps) {
  return z.min() - eps * log(accu(exp(-(z-z.min())/eps)));
}

arma::mat softmin_col(arma::mat &C, arma::mat &f, arma::mat &g, const double eps) {
  // compute S first: S = C - f * 1^T - 1 * g^T
  arma::vec ones_N = arma::ones(C.n_rows);
  arma::mat S_col(C.n_cols, f.n_cols, arma::fill::zeros);
  
  for (size_t k = 0; k < f.n_cols; ++k) {
    arma::mat S = C - f.col(k) * ones_N.t() - ones_N * g.col(k).t();
    for (size_t j = 0; j < C.n_cols; ++j) {
      S_col(j,k) = softmin(S.col(j), eps);
    }
  }
  return S_col;
}

arma::mat softmin_row(arma::mat &C, arma::mat &f, arma::mat &g, const double eps) {
  arma::vec ones_N = arma::ones(C.n_cols);
  arma::mat S_row(C.n_rows, f.n_cols, arma::fill::zeros);
  
  for (size_t k = 0; k < f.n_cols; ++k) {
    arma::mat S = C - f.col(k) * ones_N.t() - ones_N * g.col(k).t();
    for (size_t i = 0; i < C.n_rows; ++i) {
      S_row(i,k) = softmin(S.row(i).t(), eps);
    }
  }
  return S_row;
}


// [[Rcpp::export]]
int test_logKT(arma::mat &A, arma::mat &C, arma::vec &w, double eps) {
  arma::mat K = exp(-C/eps);
  arma::mat f(arma::size(A), arma::fill::zeros);
  // arma::mat g(arma::size(A), arma::fill::zeros);
  arma::mat v(arma::size(A), arma::fill::ones);
  
  arma::mat u = A / (K*v);
  f = softmin_row(C, f, f, eps) + f + eps*log(A);
  
  // Rcpp::Rcout << u << std::endl;
  // Rcpp::Rcout << exp(f/eps) << std::endl;
  
  arma::mat KTu = K.t()*u;
  arma::mat PiKTu = prod(pow(KTu.each_row(), w.t()), 1);
  
  arma::vec logPi = -(softmin_col(C, f, f, eps) + f) * w / eps;
  
  Rcpp::Rcout << PiKTu << std::endl;
  Rcpp::Rcout << exp(logPi) << std::endl;
  Rcpp::Rcout << accu(PiKTu) << std::endl;
  
  return 0;
}
