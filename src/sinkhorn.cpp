
// implement the Sinkhorn algorithms with jacobians

// #include <Rcpp.h>
// using namespace Rcpp;


#include <RcppArmadillo.h>


// identity matrix
arma::mat I(const int n) {
  arma::mat In = arma::mat(n, n, arma::fill::eye);
  return In;
}

// ones vector
arma::mat ones(const int n) {
  arma::vec onesn = arma::vec(n, arma::fill::ones);
  return onesn;
}

// sinkhorn-parallel
void sinkhorn(arma::mat &A, arma::mat &B, arma::mat &C, const double eps,
              const int maxIter = 1000, const double zeroTol = 1e-8) {
  // init iter, err
  int iter = 0;
  double err = 1000.;
  
  // K
  arma::mat K = exp(-C/eps);
  
  // M,N,S
  int M = A.n_rows;
  int N = B.n_rows;
  int S = A.n_cols;
  
  // init U and V
  arma::mat U(arma::size(A), arma::fill::ones);
  arma::mat V(arma::size(B), arma::fill::ones);
  // Jacobian of U and V
  arma::mat JU(M*S, M*S, arma::fill::zeros);
  arma::mat JV(N*S, M*S, arma::fill::zeros);
  
  // temp mats
  arma::mat KV, KTU;
  
  while ((iter <= maxIter) & (err >= zeroTol)) {
    KV = K * V;
    JU = diagmat(vectorise(1 / KV)) 
      - diagmat(vectorise(A / pow(KV, 2))) * kron(I(S), K) * JV;
    U = A / KV;
    
    KTU = K.t() * U;
    JV = - diagmat(vectorise(B / pow(KTU, 2))) * kron(I(S), K).t() * JU;
    V = B / KTU;
    
    // increment
    iter += 1;
    err = norm(U % (K * V), 2) + norm(V % (K.t() * U), 2);
  }
}
