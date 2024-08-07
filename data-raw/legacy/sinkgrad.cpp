
#include <RcppArmadillo.h>


// [[Rcpp::export]]
Rcpp::List sinkgrad(
  arma::vec doc,                  // input doc distribution
  arma::mat D,                  // topics
  arma::mat C,                  // cost
  arma::vec lambda,             // weight
  const double eps,             // epsilon: regularization const
  const bool with_grad = true,  // calculate gradient or not (pure barycenter)
  const int maxIter = 1000,
  const double zeroTol = 1e-8
) {
  
  const arma::mat K = exp(-C/eps);
  
  arma::mat u(D.n_rows, D.n_cols, arma::fill::ones);
  arma::mat v(D.n_rows, D.n_cols, arma::fill::ones);
  arma::mat u_prev, v_prev;
  
  arma::mat Phi(D.n_rows, D.n_cols, arma::fill::zeros);
  arma::vec p(D.n_rows); // estimated barycenter
  
  // vector to track the intermediate results
  std::vector<arma::mat> Phi_vec;
  std::vector<arma::mat> v_vec;
  
  // init 
  int iter = 0;
  double err = 1000.;
  
  while ((iter < maxIter) & (err > zeroTol)) {
    u_prev = u;
    v_prev = v;
    
    u = (D.each_col() / (K * v));
    Phi = K.t() * u;
    p = sum(pow(Phi.each_col(), lambda), 1);
    v = p / Phi.each_col();
    
    // append Phi and v into the vector: tracking gradient
    Phi_vec.push_back(Phi);
    v_vec.push_back(v);
    
    if (!v.is_finite() | !u.is_finite()) {
      u = u_prev;
      v = v_prev;
    }
    
    ++iter;
    err = norm(D - v % (K * u));
  }
  
  // calculate gradient if desired 
  // otherwise skip for pure barycenter calculation
  arma::vec w(D.n_cols);
  arma::mat r(D.n_cols, D.n_rows);
  arma::vec g = 2 * (p - doc) * p;
  // TODO: add other loss function gradients
  
  arma::mat y(D.n_cols, D.n_rows);
  arma::mat z(D.n_cols, D.n_rows);
  arma::vec n = 2 * (p - doc);
  
  if (with_grad) {
    
    // Backward: weights
    for (int i = iter; i > 0; --i) {
      
      for (int k = 0; k < D.n_cols; ++k) {
        // update w
        w[k] += sum(log(Phi_vec[i].col(k) * g));
        // update r
        // prepare v_vec[-1]
        arma::vec bs(D.n_rows, arma::fill::zeros);
        if (i > 0) {
          bs = v_vec[i-1].col(k);
        }
        
        arma::mat r1 = K * ((lambda[k] * g - r.row(k)) / Phi_vec[i].col(k));
        arma::mat r2 = r1 % (D.col(k) / pow(K * bs, 2));
        r.row(k) = - K.t() * r2 % bs;
      }
      
      // update g
      g = sum(r, 0); // N * 1
    }
    
    arma::mat c(D.n_cols, D.n_rows);
    arma::vec ones_N(D.n_rows, arma::fill::ones);
    // Backward: dictionary
    for (int i = iter; i > 0; --i) {
      
      for (int k = 0; k < D.n_cols; ++k) {
        // prepare v_vec[-1]
        arma::vec bs(D.n_rows, arma::fill::zeros);
        if (i > 0) {
          bs = v_vec[i-1].col(k);
        }
        // update c
        arma::vec c = K * ((lambda[k] * n) % v_vec[i]);
        // update y
        y.row(k) += c / (K * bs);
        // update z
        z.row(k) = - (ones_N / Phi_vec[i]) % 
          (K.t() * ((D.col(k) % c) / pow(K * bs, 2)));
      }
      
      // update n
      n = sum(z, 0);
    }
  }
  
  // return values
  Rcpp::List res = Rcpp::List::create(
    Rcpp::Named("p") = p,
    Rcpp::Named("grad_D") = y,
    Rcpp::Named("grad_lambda") = w,
    Rcpp::Named("u") = u,
    Rcpp::Named("v") = v,
    Rcpp::Named("K") = K
  );
  return res;
}
