
// implement the Wasserstein barycenter algorithms (parallel + log)

#include <RcppArmadillo.h>
using namespace arma;

// TODO: add Jacobians and gradients

// [[Rcpp::export]]
Rcpp::List barycenter(mat &A, mat &C, vec &lbd, const double eps, 
                      const bool withgrad, 
                      const int maxIter = 1000, const double zeroTol = 1e-8) {
  // init iter, err
  int iter = 0;
  double err = 1000.;
  
  // K
  mat K = exp(-C/eps);
  
  // N,S
  int N = A.n_rows;
  int S = A.n_cols;
  
  // init U, V
  mat U(N, S, fill::ones);
  mat V(N, S, fill::ones);
  vec b(N, fill::zeros);
  
  // temp mats
  vec onesN = vec(N, fill::ones);
  vec onesS = vec(S, fill::ones);
  mat KV(N, S), KTU(N, S);
  
  while ((iter <= maxIter) & (err >= zeroTol)) {
    Rcpp::checkUserInterrupt();
    KV = K * V;
    U = A / KV;
    
    KTU = K.t() * U;
    b = prod(pow(KTU, kron(lbd.t(), onesN)), 1);
    V = kron(onesS.t(), b) / KTU;
    
    // check stop criterion
    iter++;
    err = norm(A - U % (K * V), 2);
  }
  
  Rcpp::List res;
  res = Rcpp::List::create(
    Rcpp::Named("b") = b,
    Rcpp::Named("iter") = iter,
    Rcpp::Named("err") = err
  );
  return res;
}


// [[Rcpp::export]]
vec barycenter_log(mat &A, mat &C, vec &lbd, const double eps, 
                    const bool withgrad,
                    const int maxIter = 1000, const double zeroTol = 1e-8) {
  // init iter, err
  int iter = 0;
  double err = 1000.;
  
  // N,S
  int N = A.n_rows;
  int S = A.n_cols;
  
  // init F, G
  mat F(N, S, fill::zeros), G(N, S, fill::zeros);
  vec b(N, fill::zeros), logb(N, fill::zeros);
  
  // init temp mats
  vec onesS(S, fill::ones), onesN(N, fill::ones);
  mat IS(S, S, fill::eye);
  mat onesSTkronC = kron(onesS.t(), C);
  mat onesSTkronCT = kron(onesS, C).t();
  mat ISkrononesNT = kron(IS, onesN.t());
  mat ISkrononesN = kron(IS, onesN);
  mat epslogA = eps * log(A);
  
  mat logB(N, S, fill::zeros);
  mat R(N, N*S, fill::zeros);
  mat Q(N, N*S, fill::zeros);
  mat logQISkrononesN;
  double c;
  
  // TODO: convert to a for-loop and report termination reason
  
  while ((iter <= maxIter) & (err >= zeroTol)) {
    
    // Rcpp::Rcout << G << std::endl;
    
    R = onesSTkronC - F * ISkrononesNT - kron(onesN, vectorise(G).t());
    c = R.min();
    Q = exp(-(R-c)/eps);
    F = F + epslogA + c - eps * log(Q * ISkrononesN);
    
    R = onesSTkronCT - kron(onesN, vectorise(F).t()) - G * ISkrononesNT;
    c = R.min();
    Q = exp(-(R-c)/eps);
    logQISkrononesN = log(Q * ISkrononesN);
    // Rcpp::Rcout << logQISkrononesN * lbd << std::endl;
    // Rcpp::Rcout << G * lbd / eps << std::endl;
    logb = logQISkrononesN * lbd - G * lbd / eps - c / eps;
    logB = kron(onesS.t(), logb);
    G = G + eps * logB + c - eps * logQISkrononesN;
    
    // check termination
    iter++;
    // TODO: how to check convergence of the algorithm?
    // TODO: 
    R = onesSTkronC - F * ISkrononesNT - kron(onesN, vectorise(G).t());
    c = R.min();
    Q = exp(-(R-c)/eps);
    err = norm((Q*c/eps) * ISkrononesN - A, 2);
    // Rcpp::Rcout << ((Q - c/eps) * ISkrononesN - A)(0,0) << std::endl;
    Rcpp::Rcout << iter << ": " << err << std::endl;
    Rcpp::Rcout << exp(-(
        onesSTkronC - F * ISkrononesNT - kron(onesN, vectorise(G).t())
    )/eps) << std::endl;
  }
  b = exp(logb);
  return b;
}
