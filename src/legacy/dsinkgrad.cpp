
//[[Rcpp::depends(RcppClock)]]
// [[Rcpp::plugins(cpp17)]]

#include <RcppArmadillo.h>

#include <RcppClock.h>
#include <thread>


/////////////////////////////////////////////////////////////////////////
// utility functions
/////////////////////////////////////////////////////////////////////////

void softmax(arma::vec &v) {
  arma::vec vexp = exp(v - v.max());
  v = vexp / accu(vexp);
}

void softmax(arma::mat &m) {
  for (size_t i = 0; i < m.n_cols; ++i) {
    arma::vec vexp = exp(m.col(i) - m.col(i).max());
    m.col(i) = vexp / accu(vexp);
  }
}

arma::mat softmax_jac(arma::vec &y) {
  // y = softmax(x)
  // return jacobian of softmax wrt x
  return diagmat(y) - y * y.t();
}

// arma::mat softmax_jac(arma::vec &v) {
//   arma::vec vexp = exp(v - v.max());
//   arma::vec vsoftmax = vexp / accu(vexp);
//   arma::mat vjac = diagmat(vsoftmax) - vsoftmax * vsoftmax.t();
//   return vjac;
// }
// 
// arma::cube softmax_jac(arma::mat &m) {
//   arma::cube m_jac(m.n_rows, m.n_rows, m.n_cols, arma::fill::zeros);
//   for (size_t i = 0; i < m.n_cols; ++i) {
//     arma::vec vexp = exp(m.col(i) - m.col(i).max());
//     arma::vec vsoftmax = vexp / accu(vexp);
//     m_jac.slice(i) = diagmat(vsoftmax) - vsoftmax * vsoftmax.t();
//   }
//   return m_jac;
// }

/////////////////////////////////////////////////////////////////////////
// functions for calculating Jacobian
/////////////////////////////////////////////////////////////////////////

void compute_Ju_lbd(
    arma::cube &Ju_lbd,
    arma::mat &A, arma::mat &K, arma::mat &v, arma::cube &Jv_lbd) {
  arma::mat Kv = K*v;
  for (int k = 0; k < A.n_cols; ++k) {
    Ju_lbd.slice(k) = - diagmat(A.col(k) / pow(Kv.col(k),2)) * K * Jv_lbd.slice(k);
  }
}

void compute_Ju_A(
    std::vector<arma::cube> &Ju_A,  // ijks: row,col,slice,vec
    arma::mat &A, arma::mat &K, arma::mat &v, std::vector<arma::cube> &Jv_A
) {
  arma::mat Kv = K*v;
  arma::mat eyes(A.n_rows, A.n_rows, arma::fill::eye);
  arma::mat zeros(A.n_rows, A.n_rows, arma::fill::zeros);
  for (int s = 0; s < A.n_cols; ++s) {
    for (int k = 0; k < A.n_cols; ++k) {
      arma::mat JA_A = (k == s ? eyes : zeros);
      Ju_A[s].slice(k) = diagmat(1 / Kv.col(k)) * JA_A -
        diagmat(A.col(k) / pow(Kv.col(k), 2)) * K * Jv_A[s].slice(k);
    }
  }
}

// helper function to compute G: to be used for both lbd and A
arma::mat compute_G(arma::vec &lbd, arma::mat &KTu, arma::cube &KTJu) {
  arma::mat G(KTJu.n_rows, KTJu.n_cols, arma::fill::zeros);
  for (int i = 0; i < KTJu.n_rows; ++i) {
    for (int j = 0; j < KTJu.n_cols; ++j) {
      for (int k = 0; k < KTJu.n_slices; ++k) {
        G(i,j) += lbd(k) * KTJu(i,j,k) / KTu(i,k);
      }
    }
  }
  return G;
}

void compute_Jalpha_lbd(
  arma::mat &Jalpha_lbd,
  arma::mat &K, 
  arma::mat &u, arma::cube &Ju_lbd,
  arma::vec &d, arma::mat &Jd_lbd,
  arma::vec &Pi, arma::vec &lbd, arma::vec &alpha
) {
  arma::mat KTu = K.t()*u;
  arma::cube KTJu_lbd = K.t() * Ju_lbd.each_slice();
  Jalpha_lbd = 
    diagmat(Pi) * Jd_lbd + diagmat(alpha) * (log(KTu) + compute_G(lbd, KTu, KTJu_lbd));
}

void compute_Jalpha_A(
  arma::cube &Jalpha_A, 
  arma::mat &K,
  arma::mat &u, std::vector<arma::cube> &Ju_A, // ijks
  arma::vec &d, arma::cube &Jd_A,
  arma::vec &Pi, arma::vec &lbd
) {
  arma::mat KTu = K.t()*u;
  int nslices = Jalpha_A.n_slices;
  for (int s = 0; s < nslices; ++s) {
    arma::cube KTJu_A_s = K.t() * Ju_A[s].each_slice(); // ijk
    arma::mat G_s = compute_G(lbd, KTu, KTJu_A_s);
    Jalpha_A.slice(s) = diagmat(Pi)*(Jd_A.slice(s) + diagmat(d) * G_s);
  }
}

void compute_Jv_lbd(
  arma::cube &Jv_lbd,
  arma::mat &K, arma::mat &KTu, arma::cube &Ju_lbd, 
  arma::vec &alpha, arma::mat &Jalpha_lbd
) {
  for (int k = 0; k < KTu.n_cols; ++k) {
    Jv_lbd.slice(k) = diagmat(1 / KTu.col(k)) * Jalpha_lbd - 
      diagmat(alpha / pow(KTu.col(k), 2)) * K.t() * Ju_lbd.slice(k);
  }
}

void compute_Jv_A(
  std::vector<arma::cube> &Jv_A, // ijsk
  arma::mat &K, arma::mat &KTu,
  std::vector<arma::cube> &Ju_A,
  arma::vec &alpha, arma::cube &Jalpha_A
) {
  for (int k = 0; k < Jv_A.size(); ++k) {
    for (int s = 0; s < Jv_A.size(); ++s) {
      Jv_A[s].slice(k) = diagmat(1 / KTu.col(k)) * Jalpha_A.slice(s) -
        diagmat(alpha / pow(KTu.col(k), 2)) * K.t() * Ju_A[s].slice(k);
    }
  }
}

void compute_Jd_lbd(
  arma::mat &K, 
  arma::mat &Kd, arma::vec &dalpha,
  arma::vec &d, arma::mat &Jd_lbd,
  arma::vec &alpha, arma::mat &Jalpha_lbd
) {
  Jd_lbd = .5*diagmat(sqrt(Kd / dalpha)) * (
    diagmat(alpha / Kd) * Jd_lbd +
      diagmat(d / Kd) * Jalpha_lbd -
      diagmat(dalpha / pow(Kd, 2)) * K * Jd_lbd
  );
}

void compute_Jd_A(
  arma::mat &K,
  arma::mat &Kd, arma::vec &dalpha,
  arma::vec &d, arma::cube &Jd_A,
  arma::vec &alpha, arma::cube &Jalpha_A
) {
  for (int s = 0; s < Jd_A.n_slices; ++s) {
    Jd_A.slice(s) = .5*diagmat(sqrt(Kd / dalpha)) * (
      diagmat(alpha / Kd) * Jd_A.slice(s) + 
        diagmat(d / Kd) * Jalpha_A.slice(s) - 
        diagmat(dalpha / pow(Kd, 2)) * K * Jd_A.slice(s)
    );
  }
}

/////////////////////////////////////////////////////////////////////////
// functions for calculating Jacobian of Debiased Sinkhorn
/////////////////////////////////////////////////////////////////////////

// template<typename Tv, typename Tm, typename Tc>
std::tuple<arma::vec,arma::mat,arma::cube> 
dsinkjac(arma::mat &A, arma::mat &C, arma::vec &lbd, double &reg,
         const int maxIter=1000, const double zeroTol=1e-8) {
  Rcpp::Clock clock;
  
  arma::mat K = exp(-C/reg);
  
  // init v and d
  arma::vec alpha(A.n_rows, arma::fill::zeros);
  arma::mat u(A.n_rows, A.n_cols, arma::fill::ones);
  arma::mat v(A.n_rows, A.n_cols, arma::fill::ones);
  arma::vec d(A.n_rows, arma::fill::ones);
  
  arma::cube J_a_eg(A.n_rows, A.n_rows, A.n_cols, arma::fill::zeros);
  
  // init Jv (wrt A and lbd)
  arma::cube Jv_lbd(A.n_rows, A.n_cols, A.n_cols, arma::fill::zeros); // ijk
  std::vector<arma::cube> Jv_A(A.n_cols, J_a_eg); // ijks

  // init Ju (wrt A and lbd)
  arma::cube Ju_lbd(A.n_rows, A.n_cols, A.n_cols, arma::fill::zeros); // ijk
  std::vector<arma::cube> Ju_A(A.n_cols, J_a_eg); // ijks
  
  // init Jd (wrt A and lbd)
  arma::mat Jd_lbd(arma::size(A), arma::fill::zeros); // ij
  arma::cube Jd_A(A.n_rows, A.n_rows, A.n_cols, arma::fill::zeros); // ijs
  
  // output Jalpha (wrt A and lbd)
  arma::mat Jalpha_lbd(A.n_rows, A.n_cols, arma::fill::zeros);
  arma::cube Jalpha_A(A.n_rows, A.n_rows, A.n_cols, arma::fill::zeros); // ijk
  
  // init backup vars
  arma::mat u_prev = u;
  arma::mat v_prev = v;
  arma::cube Ju_lbd_prev = Ju_lbd;
  arma::cube Jv_lbd_prev = Jv_lbd;
  std::vector<arma::cube> Ju_A_prev = Ju_A;
  std::vector<arma::cube> Jv_A_prev = Jv_A;
  
  // init
  int iter = 0;
  double err = 1000.;
  
  while ((iter < maxIter) & (err > zeroTol)) {
    // backup the important vars
    u_prev = u;
    v_prev = v;
    Ju_lbd_prev = Ju_lbd;
    Jv_lbd_prev = Jv_lbd;
    Ju_A_prev = Ju_A;
    Jv_A_prev = Jv_A;
    
    clock.tick("update_u");
    // update u
    arma::mat Kv = K*v;
    u = A / Kv;
    // compute Ju wrt lbd
    compute_Ju_lbd(Ju_lbd, A, K, v, Jv_lbd);
    // compute Ju wrt A
    compute_Ju_A(Ju_A, A, K, v, Jv_A);
    clock.tock("update_u");
    
    clock.tick("update_alpha");
    // update alpha
    arma::mat KTu = K.t() * u;
    arma::mat KTu_lbd = pow(KTu.each_row(), lbd.t());
    arma::vec Pi = prod(KTu_lbd, 1);
    alpha = d % Pi;
    // compute Jalpha wrt lbd (ij)
    compute_Jalpha_lbd(Jalpha_lbd, K, u, Ju_lbd, d, Jd_lbd, Pi, lbd, alpha);
    // compute Jalpha wrt A (ijs)
    compute_Jalpha_A(Jalpha_A, K, u, Ju_A, d, Jd_A, Pi, lbd);
    clock.tock("update_alpha");
    
    clock.tick("update_v");
    // update v
    v = alpha / KTu.each_col();
    // compute Jv wrt lbd and A
    compute_Jv_lbd(Jv_lbd, K, KTu, Ju_lbd, alpha, Jalpha_lbd);
    compute_Jv_A(Jv_A, K, KTu, Ju_A, alpha, Jalpha_A);
    clock.tick("update_v");
    
    clock.tick("update_d");
    // update d (first jacobian and then update d)
    arma::mat Kd = K*d;
    arma::vec dalpha = d % alpha;
    // Jacobian of d wrt lbd
    compute_Jd_lbd(K, Kd, dalpha, d, Jd_lbd, alpha, Jalpha_lbd);
    compute_Jd_A(K, Kd, dalpha, d, Jd_A, alpha, Jalpha_A);
    d = sqrt(dalpha / Kd);
    clock.tick("update_d");
    
    // break loop if nan or inf in u or v
    if (!u.is_finite() | !v.is_finite()) {
      // if u and v run into numerical issues (instability)
      // revert the important vars using the backup
      u = u_prev;
      v = v_prev;
      Ju_lbd = Ju_lbd_prev;
      Jv_lbd = Jv_lbd_prev;
      Ju_A = Ju_A_prev;
      Jv_A = Jv_A_prev;
      break;
    }
    
    ++iter;
    err = norm(A - u % (K*v));
  }
  clock.stop("dsinkjac");
  // Rcpp::List res = Rcpp::List::create(
  //   Rcpp::Named("alpha") = alpha,
  //   Rcpp::Named("Jalpha_lbd") = Jalpha_lbd,
  //   Rcpp::Named("Jalpha_A") = Jalpha_A
  // );
  // return res;
  return {alpha, Jalpha_lbd, Jalpha_A};
}


/////////////////////////////////////////////////////////////////////////
// functions for calculating gradients
/////////////////////////////////////////////////////////////////////////
// TODO: specify loss function

arma::vec sinkgrad(arma::vec &a, arma::mat &C, arma::vec &b, const double reg, 
              const int maxIter=1000,const double zeroTol=1e-8) {
  // init
  int iter = 0;
  double err = 1000.;
  
  arma::mat K = exp(-C/reg);
  // init v and d
  arma::vec u(arma::size(a), arma::fill::ones);
  arma::vec v(arma::size(b), arma::fill::ones);
  // Jacobian wrt `a`
  arma::mat Ju(arma::size(C), arma::fill::zeros);
  arma::mat Jv(arma::size(C), arma::fill::zeros);
  
  arma::vec u_prev, v_prev;
  arma::mat Ju_prev, Jv_prev;
  
  while ((iter < maxIter) & (err > zeroTol)) {
    u_prev = u;
    v_prev = v;
    Ju_prev = Ju;
    Jv_prev = Jv;
    
    arma::vec Kv = K*v;
    u = a / Kv;
    Ju = diagmat(1 / Kv) - diagmat(a / pow(Kv,2)) * K * Jv;
    
    arma::vec KTu = K.t()*u;
    v = b / KTu;
    Jv = -diagmat(b / pow(KTu,2)) * K.t() * Ju;
    
    // break loop if nan or inf in u or v
    if (!u.is_finite() | !v.is_finite()) {
      u = u_prev;
      v = v_prev;
      Ju = Ju_prev;
      Jv = Jv_prev;
      break;
    }
    
    ++iter;
    err = norm(a - u % (K*v)) + norm(b - v % (K.t()*v));
  }
  
  // calculate gradient
  arma::vec grad(arma::size(a), arma::fill::zeros);
  arma::mat KC = K % C;
  for (int i = 0; i < grad.size(); ++i) {
    grad(i) = accu(diagmat(Ju.col(i)) * KC * diagmat(v) +
      diagmat(u) * KC * diagmat(Jv.col(i)));
  }
  return grad;
}

// batch-wise gradient
std::tuple<arma::mat,arma::vec,arma::mat> 
dsinkgrad(
    arma::mat &docs, arma::mat &A, arma::mat &C, arma::mat &Lbd, double reg,
    const int loss, const int maxIter=1000,const double zeroTol=1e-8
) {
  // input
  // docs: input data, N*M
  // A: topics distribution in Sigma^N (parameters), N*K
  // Lbd: weights distribution in Sigma^K (parameter), K*M
  // loss: 0 -> squared sum loss, 1 -> sinkhorn loss
  //
  // output:
  // predicted output (barycenter) Y (N*M)
  // gradient wrt A (N*K*M) and Lbd (K*M)
  // Rcpp::Clock clock;
  
  arma::mat Y(A.n_rows, docs.n_cols, arma::fill::zeros);
  arma::mat grad_loss_Lbd(arma::size(Lbd), arma::fill::zeros);
  arma::cube grad_loss_A(A.n_rows, A.n_cols, docs.n_cols, arma::fill::zeros);
  
  for (size_t m = 0; m < docs.n_cols; ++m) {
    arma::vec y = docs.col(m);
    arma::vec lbd = Lbd.col(m);
    // clock.tick("dsinkjac");
    auto [yhat, J_yhat_lbd, J_yhat_A] = dsinkjac(A, C, lbd, reg, maxIter, zeroTol);
    // clock.tock("dsinkjac");
    
    // clock.tick("sinkgrad");
    arma::vec grad_loss_yhat; // gradient of Loss wrt yhat
    if (loss == 0) {
      grad_loss_yhat = 2*(yhat-y);
    } else if (loss == 1) {
      grad_loss_yhat = sinkgrad(yhat, C, y, reg, maxIter, zeroTol);
    } else {
      Rcpp::stop("loss function not implemented, must be either 0 or 1");
    }
    // clock.tock("sinkgrad");
    
    Y.col(m) = yhat;
    // grad_L_var = grad_L_yhat^T * grad_yhat_var
    // var = lbd or A
    // grad_loss_Lbd.col(m) = J_yhat_lbd.t() * grad_loss_yhat;
    // for (size_t k = 0; k < A.n_cols; ++k) {
    //   grad_loss_A.slice(m).col(k) = J_yhat_A.slice(k).t() * grad_loss_yhat;
    // }
    // Grad of L wrt lbd:
    // J^T softmax * J^T yhat * grad_yhat_lbd
    grad_loss_Lbd.col(m) = softmax_jac(lbd).t() * J_yhat_lbd.t() * grad_loss_yhat;
    for (size_t k = 0; k < A.n_cols; ++k) {
      arma::vec A_k = A.col(k);
      grad_loss_A.slice(m).col(k) = softmax_jac(A_k).t() * 
        J_yhat_A.slice(k).t() * grad_loss_yhat;
    }
  }
  
  // return average grad (instead of by batch)
  arma::vec avggrad_loss_Lbd = sum(grad_loss_Lbd, 1) / grad_loss_Lbd.n_cols;
  arma::mat avggrad_loss_A = sum(grad_loss_A, 2) / grad_loss_A.n_slices;
  
  // clock.stop("time_dsinkgrad");
  // Rcpp::List res = Rcpp::List::create(
  //   Rcpp::Named("Y") = Y,
  //   Rcpp::Named("grad_lbd") = avggrad_loss_Lbd,
  //   Rcpp::Named("grad_A") = avggrad_loss_A
  // );
  // return res;
  return {Y, avggrad_loss_Lbd, avggrad_loss_A};
}


/////////////////////////////////////////////////////////////////////////
// functions for optimization: SGD and Adam
/////////////////////////////////////////////////////////////////////////

template<typename T> // T: arma::vec or arma::mat
void optimize_sgd(T &params, T &grad, const double lr = .001) {
  // Rcpp::Rcout << arma::size(params) << " " << arma::size(grad) << std::endl;
  params -= lr * grad;
}

template<typename T>
void optimize_adam(T &params, T &grad, T &m, T &v, const int t,
                   const double lr = .001, const double eps = 1e-8, 
                   const double beta1 = .9, const double beta2 = .999) {
  // Rcpp::Rcout << arma::size(params) << " " << arma::size(grad) << std::endl;
  m = beta1 * m + (1-beta1) * grad;
  v = beta2 * v + (1-beta2) * pow(grad, 2);
  T mhat = m / (1 - std::pow(beta1, t));
  T vhat = v / (1 - std::pow(beta2, t));
  params -= lr * mhat / (sqrt(vhat) + eps);
}

template<typename T>
void optimize(
    T &params, T &grad, T &m, T &v, 
    const int t, const int method = 1,
    const double lr = .001, const double eps = 1e-8, 
    const double beta1 = .9, const double beta2 = .999
) {
  // optimize parameters with gradients
  // method = 0, vanilla SGD
  // method = 1, Adam
  if (method == 0) {
    optimize_sgd(params, grad, lr);
  } else if (method == 1) {
    optimize_adam(params, grad, m, v, t, lr, eps, beta1, beta2);
  } else {
    Rcpp::stop("optmization method not implemented");
  }
}


/////////////////////////////////////////////////////////////////////////
// WIG model loop
/////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
Rcpp::List wig_cpp(
  arma::mat &docs,                      // input docs as matrix: N*M
  arma::mat &C,                         // distance matrix: N*N
  const int num_topics,                 // num of topics: K
  const int batch_size,                 // batch size
  const int epochs,                     // num of epochs
  const int loss = 1,                   // loss: 0 (squared sum), 1 (Sinkhorn)
  const int optimizer = 1,              // optimizer: 0 (SGD), 1 (Adam)
  const double lr = .001,               // SGD/Adam: learning rate
  const double beta1 = .9,              // Adam: beta1
  const double beta2 = .999,            // Adam: beta2
  const double eps = 1e-8,              // Adam: epsilon
  const double reg = 0.1,               // sinkhorn: regularization const
  const int maxIter = 1000,             // sinkhorn: max iteration
  const double zeroTol = 1e-8,          // sinkhorn: convergence tolerance
  const bool verbose = false            // verbose option
) {
  // input:
  // docs: N*M
  // batch size: n
  Rcpp::Clock clock;
  
  clock.tick("init");
  // 0. init basis `A` and weight `Lbd`
  // A: N*K
  // Lbd: K*M
  arma::mat A(C.n_rows, num_topics, arma::fill::randn);
  arma::mat Lbd(num_topics, docs.n_cols, arma::fill::randn);
  // make sure col-wise sum to one
  softmax(A);
  softmax(Lbd);
  // init output Yhat
  arma::mat Yhat(arma::size(docs), arma::fill::zeros);
  
  // m0 and v0 for Adam optimizer
  arma::mat m_lbd(num_topics, docs.n_cols, arma::fill::zeros);
  arma::mat v_lbd(num_topics, docs.n_cols, arma::fill::zeros);
  arma::mat m_A(C.n_rows, num_topics, arma::fill::zeros); 
  arma::mat v_A(C.n_rows, num_topics, arma::fill::zeros); 
  // t for Adam
  int t = 1;
  clock.tock("init");
  
  // clock.tick("total");
  // outer loop: number of passes of input data
  for (int p = 0; p < epochs; ++p) {
    
    // calculate length of batches
    int batch_num = std::ceil(static_cast<double>(docs.n_cols) / batch_size);
    
    for (int i = 0; i < batch_num; ++i) {
      // if (verbose) {
      //   Rcpp::Rcout << "training epoch: " << p << ", batch: " << i
      //     << std::endl;
      // }
      
      // clock.tick("batch 1");
      // 1. split docs into sub-matrix
      int i_start = i * batch_size;
      int i_end = (i + 1)*batch_size - 1;
      int docsize = (int) docs.n_cols;
      i_end = i_end >= docsize - 1 ? docsize - 1 : i_end;
      // i_end = std::min(i_end, docs.n_cols);
      
      arma::mat docs_batch = docs.cols(i_start, i_end); // N*m
      arma::mat Lbd_batch = Lbd.cols(i_start, i_end);   // K*m
      // clock.tock("batch 1");
      
      // Rcpp::Rcout << docs_batch << std::endl;
      // Rcpp::Rcout << Lbd_batch << std::endl;
      // Rcpp::stop("stop here");
      
      // clock.tick("batch 2");
      // 2. calculate gradient
      // grad_Lbd : 
      auto [yhat, grad_lbd, grad_A] = dsinkgrad(docs_batch, A, C, Lbd_batch, reg,
                                          loss, maxIter, zeroTol);
      arma::mat grad_Lbd(size(Lbd));
      grad_Lbd.each_col() = grad_lbd;
      
      // clock.tock("batch 2");
      
      // clock.tick("batch 3");
      // 3. optimize parameters
      optimize(Lbd, grad_Lbd, m_lbd, v_lbd, t, optimizer,lr, eps, beta1, beta2);
      optimize(A, grad_A, m_A, v_A, t, optimizer, lr, eps, beta1, beta2);
      // clock.tock("batch 3");
      
      // clock.tick("batch 4");
      // 4. write yhat into Yhat
      for (int m = 0; m < docs_batch.n_cols; ++m) {
        Yhat.col(i_start+m) = yhat.col(m);
      }
      // clock.tock("batch 4");
      
      // 5. softmax to ensure the constraints
      // softmax(Lbd);
      // softmax(A);
      
      ++t;
      // check R user interrrupt
      R_CheckUserInterrupt();
    }
  }
  // clock.tick("total");
  clock.stop("naptimes");
  
  Rcpp::List res = Rcpp::List::create(
    Rcpp::Named("topics") = A,
    Rcpp::Named("weight") = Lbd,
    Rcpp::Named("docs_pred") = Yhat
  );
  return res;
}

