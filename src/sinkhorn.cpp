
// implement the Sinkhorn algorithms with jacobians

#include <RcppArmadillo.h>
using namespace arma;

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

// commutation matrix
arma::mat commutation(int m, int n) {
  arma::uvec onetomtimesn = arma::linspace<arma::uvec>(0, m*n-1, m*n);
  arma::mat A = arma::reshape(arma::conv_to<arma::vec>::from(onetomtimesn), m, n);
  // arma::mat A = Avec.reshape(m, n);
  arma::uvec v = arma::conv_to<arma::uvec>::from(arma::vectorise(A.t()));
  arma::mat P(m*n, m*n, arma::fill::eye);
  P = P.submat(v, onetomtimesn);
  return P;
}

// sinkhorn-parallel
Rcpp::List sinkhorn(arma::mat &A, arma::mat &B, arma::mat &C, const double eps,
              const bool withgrad,
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
    if (withgrad) {
      JU = diagmat(vectorise(1 / KV)) 
      - diagmat(vectorise(A / pow(KV, 2))) * kron(I(S), K) * JV; 
    }
    U = A / KV;
    
    KTU = K.t() * U;
    if (withgrad) {
      JV = - diagmat(vectorise(B / pow(KTU, 2))) * kron(I(S), K).t() * JU; 
    }
    V = B / KTU;
    
    // increment
    iter += 1;
    err = norm(A - U % (K * V), 2) + norm(B - V % (K.t() * U), 2);
  }
  
  // result Rcpp list
  Rcpp::List res;
  if (withgrad) {
    res = Rcpp::List::create(
      Rcpp::Named("U") = U,
      Rcpp::Named("V") = V,
      Rcpp::Named("K") = K,
      Rcpp::Named("JU") = JU,
      Rcpp::Named("JV") = JV
    );
  } else {
    res = Rcpp::List::create(
      Rcpp::Named("U") = U,
      Rcpp::Named("V") = V,
      Rcpp::Named("K") = K
    );
  }
  return res;
}


// sinkhorn-log
Rcpp::List sinkhorn_log(arma::vec &a, arma::vec &b, arma::mat &C, const double eps,
                  const bool withgrad,
                  const int maxIter = 1000, const double zeroTol = 1e-8) {
  // init iter, err
  int iter = 0;
  double err = 1000.;
  
  // M,N
  int M = a.n_rows;
  int N = b.n_rows;
  
  // init f,g
  vec f(M, fill::zeros);
  vec g(N, fill::zeros);
  // init Jf, Jg
  mat Jf(M, M, fill::zeros);
  mat Jg(N, M, fill::zeros);
  // commutation matrix
  mat K = commutation(M, N);
  
  // update output matrix P (optimal coupling matrix)
  mat P(M, N, fill::zeros);
  
  // temp constant mats to speed up the computation
  vec epsloga = eps * log(a);
  vec epslogb = eps * log(b);
  mat epsdiag1overa = eps * diagmat(1 / a);
  
  vec onesN = vec(N, fill::ones);
  vec onesM = vec(M, fill::ones);
  vec onesMN = vec(M*N, fill::ones);
  mat IM = mat(M, M, fill::eye);
  mat IN = mat(N, N, fill::eye);
  
  vec onesNkronIM = kron(onesN, IM);
  vec INkrononesM = kron(IN, onesM);
  vec onesNkronIMT = kron(onesN.t(), IM);
  vec onesMkronINT = kron(onesM.t(), IN);
  
  mat R, Q;
  mat J, JP, Ja, nabla;
  double c;
  
  while ((iter <= maxIter) & (err >= zeroTol)) {
    // udpate f and Jf
    R = C - f * onesN.t() - onesM * g.t();
    c = R.min();
    Q = arma::exp(- (R - c) / eps);
    if (withgrad) {
      J = onesNkronIM * Jf + INkrononesM * Jg;
      Jf = Jf + epsdiag1overa - diagmat(vectorise(1 / (Q * onesN))) *
        onesNkronIMT * diagmat(vectorise(Q)) * J;
    }
    f = f + epsloga + c - eps * log(Q * onesN);
    // update g and Jg
    R = C - f * onesN.t() - onesM * g.t();
    c = R.min();
    Q = arma::exp(- (R - c) / eps);
    if (withgrad) {
      J = onesNkronIM * Jf + INkrononesM * Jg;
      Jg = Jg - diagmat(vectorise(1 / (Q.t() * onesM))) * 
        onesMkronINT * K * J;
    }
    g = g + epslogb + c - eps * log(Q.t() * onesM);
  }
  P = exp(- (C - f * onesN.t() - onesM * g.t()) / eps);
  if (withgrad) {
    JP = onesMN.t() * diagmat(vectorise(C + eps * log(P)));
    Ja = (diagmat(vectorise(P)) * (onesNkronIM * Jf + INkrononesM * Jg)) / eps;
    // gradient
    vec nabla = (JP * Ja).t();
  }
  
  // result Rcpp list
  Rcpp::List res;
  if (withgrad) {
    res = Rcpp::List::create(
      Rcpp::Named("f") = f,
      Rcpp::Named("g") = g,
      Rcpp::Named("P") = P,
      Rcpp::Named("nabla") = nabla
    );
  } else {
    res = Rcpp::List::create(
      Rcpp::Named("f") = f,
      Rcpp::Named("g") = g,
      Rcpp::Named("P") = P
    );
  }
  return res;
}
