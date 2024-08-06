
// [[Rcpp::plugins(cpp17)]]

#include <RcppArmadillo.h>

// double softmin(arma::vec z, double eps) {
//   return z.min() - eps * log(accu(exp(-(z-z.min())/eps)));
// }

arma::mat softmin_col(arma::mat &C, arma::mat &f, arma::mat &g, const double eps) {
  // compute S first: S = C - f * 1^T - 1 * g^T
  arma::vec ones_N = arma::ones(C.n_rows);
  arma::mat S_col(C.n_cols, f.n_cols, arma::fill::zeros);
  arma::vec S_j(C.n_rows, arma::fill::zeros);
  for (size_t k = 0; k < f.n_cols; ++k) {
    arma::mat S = C - f.col(k) * ones_N.t() - ones_N * g.col(k).t();
    for (size_t j = 0; j < C.n_cols; ++j) {
      S_j = S.col(j);
      S_col(j,k) = S_j.min() - eps * log(accu(exp(-(S_j-S_j.min())/eps)));
      // S_col(j,k) = softmin(S.col(j), eps);
    }
  }
  return S_col;
}

arma::mat softmin_row(arma::mat &C, arma::mat &f, arma::mat &g, const double eps) {
  arma::vec ones_N = arma::ones(C.n_cols);
  arma::mat S_row(C.n_rows, f.n_cols, arma::fill::zeros);
  arma::vec S_i(C.n_rows, arma::fill::zeros);
  for (size_t k = 0; k < f.n_cols; ++k) {
    arma::mat S = C - f.col(k) * ones_N.t() - ones_N * g.col(k).t();
    for (size_t i = 0; i < C.n_rows; ++i) {
      S_i = S.row(i).t();
      S_row(i,k) = S_i.min() - eps * log(accu(exp(-(S_i-S_i.min())/eps)));
      // S_row(i,k) = softmin(S.row(i).t(), eps);
    }
  }
  return S_row;
}

arma::vec softmin_row(arma::mat &C, arma::vec log_d, const double eps) {
  arma::vec ones_N = arma::ones(C.n_cols);
  arma::mat S = C - ones_N * log_d.t();
  arma::vec r_row(C.n_rows, arma::fill::zeros);
  arma::vec S_i(C.n_rows, arma::fill::zeros);
  for (size_t i = 0; i < C.n_rows; ++i) {
    S_i = S.row(i).t();
    r_row(i) = S_i.min() - eps * log(accu(exp(-(S_i-S_i.min())/eps)));
    // r_row(i) = softmin(S.row(i).t(), eps);
  }
  return r_row;
}


// [[Rcpp::export]]
int test_dsink(arma::mat &A, arma::mat &C, arma::vec &w, const double reg) {
  arma::mat F(arma::size(A), arma::fill::zeros);
  arma::mat G(arma::size(A), arma::fill::zeros);
  arma::mat log_A = log(A);
  arma::vec log_b(A.n_rows, arma::fill::value(log(1/(double)A.n_rows)));
  arma::vec log_d(A.n_rows, arma::fill::zeros);
  arma::mat S(arma::size(A), arma::fill::zeros);
  arma::vec r(A.n_rows, arma::fill::zeros);
  
  arma::mat K = exp(-C/reg);
  arma::mat u(arma::size(A), arma::fill::ones);
  arma::mat v(arma::size(A), arma::fill::ones);
  arma::vec b = exp(log_b);
  arma::vec d = exp(log_d);
  
  // iter 1
  Rcpp::Rcout << "iter 1" << std::endl;
  F += softmin_row(C, F, G, reg) + reg*log_A;
  S = softmin_col(C, F, G, reg);
  u = A / (K * v);
  
  Rcpp::Rcout << exp(F/reg) << std::endl << u << std::endl;
  // Rcpp::Rcout << "f:" << std::endl << F << std::endl;
  
  log_b = log_d - (S + G)*w / reg;
  arma::mat KTu = K.t() * u;
  arma::vec Pi = prod(pow(KTu.each_row(), w.t()), 1);
  b = d % Pi;
  
  Rcpp::Rcout << exp(log_b) << std::endl << b << std::endl;
  // Rcpp::Rcout << "log_b:" << std::endl << log_b << std::endl;
  
  G += S.each_col() + reg * log_b;
  v = b / KTu.each_col();
  
  Rcpp::Rcout << exp(G/reg) << std::endl << v << std::endl;
  // Rcpp::Rcout << "g:" << std::endl << G << std::endl;
  
  r = softmin_row(C, reg*log_d, reg);
  log_d = (log_d + log_b + r/reg)/2;
  d = sqrt((d % b) / (K*d));
  
  Rcpp::Rcout << exp(log_d) << std::endl << d << std::endl;
  // Rcpp::Rcout << "r:" << std::endl << r << std::endl;
  Rcpp::Rcout << "log_d:" << std::endl << log_d << std::endl;
  
  // // iter 2
  // Rcpp::Rcout << "iter 2" << std::endl;
  // F += softmin_row(C, F, G, reg) + reg*log_A;
  // S = softmin_col(C, F, G, reg);
  // u = A / (K * v);
  // 
  // Rcpp::Rcout << exp(F/reg) << std::endl << u << std::endl;
  // 
  // log_b = log_d - (S + G)*w / reg;
  // KTu = K.t() * u;
  // Pi = prod(pow(KTu.each_row(), w.t()), 1);
  // b = d % Pi;
  // 
  // Rcpp::Rcout << exp(log_b) << std::endl << b << std::endl;
  // 
  // G += S.each_col() + reg * log_b;
  // v = b / KTu.each_col();
  // 
  // Rcpp::Rcout << exp(G/reg) << std::endl << v << std::endl;
  // 
  // r = softmin_row(C, reg*log_d, reg);
  // log_d = (log_d + log_b + r/reg)/2;
  // d = sqrt((d % b) / (K*d));
  // 
  // Rcpp::Rcout << exp(log_d) << std::endl << d << std::endl;
  // 
  // // iter 3
  // Rcpp::Rcout << "iter 3" << std::endl;
  // F += softmin_row(C, F, G, reg) + reg*log_A;
  // S = softmin_col(C, F, G, reg);
  // u = A / (K * v);
  // 
  // Rcpp::Rcout << exp(F/reg) << std::endl << u << std::endl;
  // 
  // log_b = log_d - (S + G)*w / reg;
  // KTu = K.t() * u;
  // Pi = prod(pow(KTu.each_row(), w.t()), 1);
  // b = d % Pi;
  // 
  // Rcpp::Rcout << exp(log_b) << std::endl << b << std::endl;
  // 
  // G += S.each_col() + reg * log_b;
  // v = b / KTu.each_col();
  // 
  // Rcpp::Rcout << exp(G/reg) << std::endl << v << std::endl;
  // 
  // r = softmin_row(C, reg*log_d, reg);
  // log_d = (log_d + log_b + r/reg)/2;
  // d = sqrt((d % b) / (K*d));
  // 
  // Rcpp::Rcout << exp(log_d) << std::endl << d << std::endl;
  
  return 0;
}
