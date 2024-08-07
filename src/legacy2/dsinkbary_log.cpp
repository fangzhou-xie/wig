
// debiased Sinkhorn barycenter and its gradient (jacobian?)
// in the log domain

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

// horizontally concat matrices (before softmin operation)
arma::mat hcat(arma::mat &C, arma::mat &F, arma::mat &G) {
  
  if (F.n_cols != G.n_cols) {
    Rcpp::stop("number of cols in F and G not equal!");
  }
  arma::vec ones = arma::vec(C.n_rows, arma::fill::ones);
  arma::mat out = std::move(C - F.col(0) * ones.t() - ones * G.col(0).t());
  
  if (F.n_cols > 1) {
    for (size_t d = 1; d < F.n_cols; ++d) {
      arma::mat out_d = C - F.col(d) * ones.t() - ones * G.col(d).t();
      out = std::move(arma::join_rows(out, out_d));
    }
  }
  return out;
}

// softmin function after stacking the input into single matrix
arma::mat softmin(arma::mat &M, const double eps) {
  int N = M.n_rows;
  int D = M.n_cols / N;
  // const double Mmin = M.min();
  arma::mat I_D = arma::mat(D, D, arma::fill::eye);
  arma::vec ones_N = arma::vec(N, arma::fill::ones);
  return M.min() - eps*log(exp(-(M-M.min())/eps) * kron(I_D, ones_N));
}

// jacobian of softmin function wrt its argument matrix
arma::mat Dsoftmin(arma::mat &M, const double eps) {
  int N = M.n_rows;
  int D = M.n_cols / N;
  arma::mat M2 = exp(-vectorise(M)/eps);
  arma::mat M1 = kron(I(D), kron(ones(N).t(), I(N)));
  return diagmat(1 / (M1*M2)) * M1 * diagmat(M2);
}

// main computation function: debiased sinkhorn (with jacobian calculation)
// output is wrt b 
// [[Rcpp::export()]]
std::tuple<arma::mat,arma::mat> 
  dsink_log(arma::mat &A, arma::mat &C, arma::vec &w, 
            const double reg, const bool debias = true,
              const int maxIter = 2000, const double zeroTol = 1e-8) {
    
    // TODO: add option to do regular WDL (without debiasing)
    // that is to remove any log_d
  arma::mat F = arma::mat(arma::size(A), arma::fill::zeros);
  arma::mat G = arma::mat(arma::size(A), arma::fill::zeros);
  int N = A.n_rows;
  int D = A.n_cols;
  
  arma::mat log_A = log(A);
  arma::vec log_b = arma::vec(N, arma::fill::zeros);
  arma::vec log_d = arma::vec(N, arma::fill::zeros);
  arma::mat CT = C.t();
  
  // init intermediate vars
  arma::mat R, softmin_R;
  arma::mat S, softmin_S;
  arma::mat Q, softmin_Q;
  
  // init jacobian for eps*logA wrt A
  arma::mat eps_D_logA_A = reg * diagmat(1 / vectorise(A));
  // init jacobians for F, log_b, G, log_d: wrt w
  arma::mat D_F_w = arma::mat(N*D, D, arma::fill::zeros);
  arma::mat D_G_w = arma::mat(N*D, D, arma::fill::zeros);
  arma::mat D_logb_w = arma::mat(N, D, arma::fill::zeros);
  arma::mat D_logd_w = arma::mat(N, D, arma::fill::zeros);
  // init jacobians for F, log_b, G, log_d: wrt A
  arma::mat D_F_A = arma::mat(N*D, N*D, arma::fill::zeros);
  arma::mat D_G_A = arma::mat(N*D, N*D, arma::fill::zeros);
  arma::mat D_logb_A = arma::mat(N, N*D, arma::fill::zeros);
  arma::mat D_logd_A = arma::mat(N, N*D, arma::fill::zeros);
  
  // init other helper vec/mat
  // ones vector
  arma::vec ones_D = arma::vec(D, arma::fill::ones);
  arma::vec ones_N = arma::vec(N, arma::fill::ones);
  // Identity matrix
  arma::mat I_N = arma::mat(N,N,arma::fill::eye);
  arma::mat I_D = arma::mat(D,D,arma::fill::eye);
  // w^T * I(N)
  arma::mat kron_wT_IN = kron(w.t(), I_N);
  // kron(ones_N, I_N)
  arma::mat kron_onesN_IN = kron(ones_N, I_N);
  // kron(I_N, ones_N)
  arma::mat kron_IN_onesN = kron(I_N, ones_N);
  // kron(ones_D, I_N)
  arma::mat kron_onesD_IN = kron(ones_D, I_N);
  
  // init intermediate jacobian mats
  arma::mat D_softmin_R, D_R_A, D_R_w;
  arma::mat D_softmin_S, D_softmin_A, D_S_A, D_S_w, D_Gw_w, D_softminw_w;
  arma::mat D_softmin_Q, D_Q_A, D_Q_w;
  
  // terminal condition
  int iter = 0;
  double err = 1000.;
  arma::mat SS; // for computing the difference between the estimated A vs real
  
  while ((iter < maxIter) & (err > zeroTol)) {
    // 1. update F
    R = hcat(C, F, G);
    softmin_R = softmin(R, reg);
    F = reg * log_A + F + softmin_R;
    // 1.0 jacobian of intermediate vars
    D_softmin_R = Dsoftmin(R, reg);
    // 1.1 jacobian F wrt A
    D_R_A = -(
      kron(I_D, kron_onesN_IN) * D_F_A +
        kron(I_D, kron_IN_onesN) * D_G_A
    );
    D_F_A = D_F_A + eps_D_logA_A + D_softmin_R * D_R_A;
    // 1.2 jacobian F wrt w
    D_R_w = -(
      kron(I_D, kron_onesN_IN) * D_F_w +
        kron(I_D, kron_IN_onesN) * D_G_w
    );
    D_F_w = D_F_w + D_softmin_R * D_R_w;
    
    // 2. update log_b
    S = hcat(CT, G, F);
    softmin_S = softmin(S, reg);
    log_b = (debias ? log_d : 0) - (softmin_S + G) * w / reg;
    // log_b = log_d - (softmin_S + G) * w / reg;
    // Rcpp::Rcout << log_b << std::endl;
    // 2.0 jacobian of intermediate vars
    D_softmin_S = Dsoftmin(S, reg);
    // Rcpp::Rcout << "D_softmin_S" << "\n" << D_softmin_S << std::endl;
    // 2.1 jacobian log_b wrt A
    D_S_A = -(
      kron(I_D, kron_IN_onesN) * D_F_A +
        kron(I_D, kron_onesN_IN) * D_G_A
    );
    D_softmin_A = D_softmin_S * D_S_A;
    D_logb_A = D_logd_A - kron_wT_IN * (D_softmin_A + D_G_A) / reg;
    // 2.2 jacobian log_b wrt w
    D_S_w = -(
      kron(I_D, kron_onesN_IN) * D_G_w +
        kron(I_D, kron_IN_onesN) * D_F_w
    );
    D_Gw_w = kron_wT_IN * D_G_w + G;
    D_softminw_w = kron_wT_IN * D_softmin_S * D_S_w + softmin_S;
    D_logb_w = D_logd_w - D_softminw_w / reg - D_Gw_w / reg;
    
    // 3. update G
    G = reg * kron(ones_D.t(), log_b) + G + softmin_S;
    // 3.1 jacobian of G wrt A
    D_G_A = reg * kron_onesD_IN * D_logb_A + D_G_A + D_softmin_A;
    // 3.2 jacobian of G wrt w
    D_G_w = reg * kron_onesD_IN * D_logb_w + D_G_w + D_softmin_S * D_S_w;
    
    // 4. update log_d: only do this when using debiased sinkhorn barycenter
    if (debias) {
      Q = C - reg * (log_d * ones_N.t() + ones_N * log_d.t());
      // 4.0 jacobian of intermediate vars
      D_softmin_Q = Dsoftmin(Q, reg);
      // 4.1 jacobian of log_d wrt A
      D_Q_A = - reg * (kron_onesN_IN + kron_IN_onesN) * D_logd_A;
      D_logd_A = D_logd_A + (D_logb_A + D_softmin_Q * D_Q_A / reg) / 2;
      // 4.2 jacobian of log_d wrt w
      D_Q_w = - reg * (kron_onesN_IN + kron_IN_onesN) * D_logd_w;
      D_logd_w = D_logd_w + (D_logb_w + D_softmin_Q * D_Q_w / reg) / 2;
      // 4.3 update log_d at the end!!!
      softmin_Q = softmin(Q, reg);
      log_d = log_d + (log_b + softmin_Q / reg)/2;
    }
    
    // update counter and err
    ++iter;
    SS = hcat(C, F, G);
    err = accu(pow(log_A + softmin(SS, reg) / reg, 2));
  }
  
  // compute jacobian of b (instead of log_b) wrt A and w
  arma::mat D_b_A = diagmat(exp(log_b)) * D_logb_A;
  arma::mat D_b_w = diagmat(exp(log_b)) * D_logb_w;
  
  // return log_b;
  return {D_b_A, D_b_w};
}

