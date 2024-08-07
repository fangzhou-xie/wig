
// implement the WIG model in Armadillo

#include <RcppArmadillo.h>

#include "utils.h"

// compute Jacobian matrix of u wrt a
arma::mat jacobian_u_wrt_a(arma::vec a,arma::mat K,arma::vec v,arma::mat Jv) {
  arma::vec Kv = K * v;
  arma::mat j1 = diagmat(1/Kv);
  arma::mat j2 = diagmat(a / pow(Kv, 2));
  arma::mat j3 = K * Jv;
  return j1 - j2 * j3;
}

// compute Jacobian matrix of v wrt a
arma::mat jacobian_v_wrt_a(arma::vec b,arma::mat K,arma::vec u,arma::mat Ju){
  arma::vec KTu = K.t() * u;
  arma::mat j1 = diagmat(b / pow(KTu, 2));
  arma::mat j2 = K.t() * Ju;
  return - j1 * j2;
}

// compute the gradient of loss wrt a
arma::vec calc_grad(
  arma::mat KM,
  arma::vec u,
  arma::vec v,
  arma::mat Ju_a,
  arma::mat Jv_a
) {
  arma::vec g(Ju_a.n_rows);
  
  for (int k = 0; k < Ju_a.n_cols; ++k) {
    arma::mat g1 = diagmat(Ju_a.col(k)) * KM * diagmat(v);
    arma::mat g2 = diagmat(u) * KM * diagmat(Jv_a.col(k));
    g[k] = accu(g1) + accu(g2);
  }
  return g;
}


//////////////////////////////////////////////////////////////
// core computation: sinkhorn algorithm with its gradient
//////////////////////////////////////////////////////////////

// core computation function: for any single sample `b` and basis/topics `a`
// compute the loss and gradient

std::tuple<arma::mat, arma::mat> sinkhorn(
  const arma::mat a,
  const arma::mat M,
  const arma::vec b,
  const arma::vec alpha,
  const double reg,
  const int maxIter,
  const double zeroTol
) {
  
}
