
#include <RcppArmadillo.h>

// implement sinkhorn (matrix scaling) and log-domain stable version

// [[Rcpp::export]]
Rcpp::List sinkhorn_scaling(arma::vec a,arma::mat M,arma::vec b,double reg = .1,
              int maxIter=1000,double zeroTol=1e-8) {
  // init 
  int iter = 0;
  double err = 1000.;
  
  arma::mat K = exp(-M/reg);
  
  arma::vec u(b.n_rows, arma::fill::ones);
  arma::vec v(b.n_rows, arma::fill::ones);
  u = u / b.n_rows;
  v = v / b.n_rows;
  arma::vec u_prev, v_prev;
  
  while ((iter < maxIter) & (err > zeroTol)) {
    u_prev = u;
    v_prev = v;
    
    u = a / (K * v);
    v = b / (K.t() * u);
    
    if (
      !u.is_finite() | !v.is_finite()
    ) {
      u = u_prev;
      v = v_prev;
    }
    
    ++iter;
    err = norm(a - u % (K*v)) + norm(b - v % (K.t() * u));
  }
  
  Rcpp::List res = Rcpp::List::create(
    Rcpp::Named("u") = u,
    Rcpp::Named("v") = v,
    Rcpp::Named("K") = K,
    Rcpp::Named("iter") = iter,
    Rcpp::Named("err") = err
  );
  return res;
}


// void sinkhorn_log(arma::vec a,arma::mat M,arma::vec b,double reg = .1,
//                   int maxIter=1000,double zeroTol=1e-8) {
//   // log domain sinkhorn
//   
//   
// }

// [[Rcpp::export]]
Rcpp::List sinkhorn_barycenter(arma::mat D, arma::mat C, double reg = .1) {
  arma::mat u(D.n_rows, D.n_cols, arma::fill::ones);
  arma::mat v(D.n_rows, D.n_cols, arma::fill::ones);
  arma::mat u_prev, v_prev;
  
  arma::mat Phi(D.n_rows, D.n_cols, arma::fill::zeros);
  
  arma::mat K = exp(-C/reg);
  arma::vec p;
  
  arma::vec lbd(D.n_cols, arma::fill::ones);
  lbd /= D.n_cols;
  
  // init
  int iter = 0;
  double err = 1000.;
  int maxIter = 1000;
  double zeroTol = 1e-8;
  
  while ((iter < maxIter) & (err > zeroTol)) {
    u_prev = u;
    v_prev = v;
    
    v = D / (K * u);
    Phi = K.t() * v;
    
    p = sum(pow(Phi.each_col(), lbd), 1);
    u = p / Phi.each_col();
    
    if (!v.is_finite() | !u.is_finite()) {
      u = u_prev;
      v = v_prev;
    }
    
    ++iter;
    // err = norm(a - u % (K*v)) + norm(b - v % (K.t() * u));
    err = norm(D - v % (K * u));
  }
  
  Rcpp::List res = Rcpp::List::create(
    Rcpp::Named("u") = u,
    Rcpp::Named("v") = v,
    Rcpp::Named("K") = K,
    Rcpp::Named("p") = p,
    Rcpp::Named("iter") = iter,
    Rcpp::Named("err") = err
  );
  return res;
}
