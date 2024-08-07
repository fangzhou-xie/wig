
// #include <Rcpp.h>z
#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
arma::mat tt(SEXP m, SEXP v) {
  arma::mat m2 = Rcpp::as<arma::mat>(m);
  arma::vec v2 = Rcpp::as<arma::vec>(v);
  return m2.each_col() / v2;
}

// [[Rcpp::export]]
arma::mat tt2(arma::mat m, arma::vec v) {
  // arma::mat m2 = Rcpp::as<arma::mat>(m);
  // arma::vec v2 = Rcpp::as<arma::vec>(v);
  return m.each_col() / v;
}

// [[Rcpp::export]]
arma::mat tt3(int n, double m = 0, double sd = 1) {
  Rcpp::Function f("rnorm");
  auto v = f(n, Rcpp::Named("mean") = m, Rcpp::Named("sd") = sd);
  std::vector<double> v_vec = Rcpp::as<std::vector<double>>(v);
  
  arma::mat am(v_vec);
  // return am(v_vec);
  return am;
}
