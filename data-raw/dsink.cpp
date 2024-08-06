
#include <RcppArmadillo.h>

// debiased sinkhorn 

// [[Rcpp::export]]
Rcpp::List dsink(arma::mat D, arma::mat C, arma::vec lambda, double eps) {
  arma::mat K = exp(-C/eps);
  
  arma::mat u(size(D), arma::fill::ones);
  arma::mat v(size(D), arma::fill::ones);
  
  arma::vec alpha(D.n_rows);
  arma::vec d(D.n_rows, arma::fill::ones);
  
  arma::mat u_prev, v_prev;
  
  // init
  int iter = 0;
  double err = 1000.;
  
  const int maxIter = 1000;
  const double zeroTol = 1e-8;
  
  while ((iter < maxIter) & (err > zeroTol)) {
    u_prev = u;
    v_prev = v;
    
    u = D / (K * v);
    arma::mat KTu = K.t() * u;
    alpha = d % prod(pow(KTu.each_col(), lambda), 1);
    // pow(KTu.each_col(), lambda); // N * K
    v = alpha / KTu.each_col();
    arma::mat Kd = K * d;
    d = sqrt(d % (alpha / Kd.each_col()));
    
    ++iter;
    err = norm(D - u % (K * v));
  }
  
  Rcpp::List res = Rcpp::List::create(
    Rcpp::Named("u") = u,
    Rcpp::Named("v") = v,
    Rcpp::Named("K") = K,
    Rcpp::Named("alpha") = alpha,
    Rcpp::Named("d") = d,
    Rcpp::Named("iter") = iter,
    Rcpp::Named("err") = err
  );
  return res;
}
