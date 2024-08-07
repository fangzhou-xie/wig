
// implement Sinkhorn algorithm and the gradient wrt `a`
// here use manual gradient calculation
// faster and without depending on AD library
// iteratively: u = a ./ (K * v), v = b ./ (transpose(K) * u)

#include <Rcpp.h>
#include <RcppEigen.h>

#include "utils.h"

// compute Jacobian matrix of u wrt a
Eigen::MatrixXd jacobian_u_wrt_a(
  Eigen::VectorXd a,
  Eigen::MatrixXd &K,
  Eigen::VectorXd v,
  Eigen::MatrixXd &Jv
) {
  Eigen::VectorXd Kv = K * v;
  Eigen::MatrixXd j1 = (1 / Kv.array()).matrix().asDiagonal();
  Eigen::MatrixXd j2 = (a.array() / Kv.array().pow(2)).matrix().asDiagonal();
  Eigen::MatrixXd j3 = K * Jv;
  return j1 - j2 * j3;
}

// compute Jacobian matrix of v wrt a
Eigen::MatrixXd jacobian_v_wrt_a(
  Eigen::VectorXd b,
  Eigen::MatrixXd &K,
  Eigen::VectorXd u,
  Eigen::MatrixXd &Ju
) {
  Eigen::VectorXd KTu = K.transpose() * u;
  Eigen::MatrixXd j1 = (b.array() / KTu.array().pow(2)).matrix().asDiagonal();
  Eigen::MatrixXd j2 = K.transpose() * Ju;
  return - j1 * j2;
}

// compute the gradient of loss wrt a
Eigen::VectorXd calc_grad(
  Eigen::MatrixXd &KM,
  Eigen::VectorXd u,
  Eigen::VectorXd v,
  Eigen::MatrixXd Ju_a,
  Eigen::MatrixXd Jv_a
) {
  // calculate gradient for each row of a
  // Julia code:
  // nabla_a = map(
  //   k -> sum(
  //    diagm(Ju_a[:,k])*(K.*M)*diagm(v)+diagm(u)*(K.*M)*diagm(Jv_a[:,k]),
  //   ),
  //   1:size(a, 1),
  // )
  
  int n = Ju_a.cols();
  Eigen::VectorXd g(Ju_a.rows());
  // g.setZero();
  
  // Eigen::MatrixXd KM = K.array() * M.array();
  
  for (int k = 0; k < n; k++) {
    Eigen::MatrixXd g1 = Ju_a.col(k).asDiagonal() * KM * v.asDiagonal();
    Eigen::MatrixXd g2 = u.asDiagonal() * KM * Jv_a.col(k).asDiagonal();
    g[k] = g1.array().sum() + g2.array().sum();
  }
  return g;
}

//////////////////////////////////////////////////////////////
// core computation: sinkhorn algorithm with its gradient
//////////////////////////////////////////////////////////////

// core computation function: for any single sample `b` and basis/topics `a`
// compute the loss and gradient
// gradient is confirmed by Julia Zygote AD
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> sinkhorn(
  const Eigen::MatrixXd &a,
  const Eigen::MatrixXd &M,
  const Eigen::VectorXd &b,
  const Eigen::VectorXd &alpha,
  const double &reg,
  const int maxIter = 1000,
  const double zeroTol = 1e-8
) {
  // a: N * K (latent topics/bases)
  // M: N * N (cost matrix)
  // b: N * 1 (input document as target distribution)
  // alpha: K * 1 (weights of Wasserstein/Sinkhorn barycenter)

  // init
  int iter = 0;
  double err = 1000.;

  Eigen::MatrixXd K = (- M / reg).array().exp();
  Eigen::MatrixXd KM = K.array() * M.array();

  // init u and v
  Eigen::MatrixXd u(a.rows(), a.cols());
  Eigen::MatrixXd v(a.rows(), a.cols());
  Eigen::MatrixXd Kv(a.rows(), a.cols());
  Eigen::MatrixXd KTu(a.rows(), a.cols());
  u.setOnes();
  v.setOnes();

  // init J_a(u) and J_a(v)
  // each correspond to one column of a
  Eigen::MatrixXd J(a.rows(), a.cols());
  J.setZero();
  std::vector<Eigen::MatrixXd> Ju_a_vec(a.cols(), J);
  std::vector<Eigen::MatrixXd> Jv_a_vec(a.cols(), J);
  
  // init output gradient of loss wrt a (same size as a)
  Eigen::MatrixXd nabla(a.rows(), a.cols());
  
  // init u_prev, v_prev, Ju_a_prev, Jv_a_prev
  Eigen::MatrixXd u_prev;
  Eigen::MatrixXd v_prev;
  std::vector<Eigen::MatrixXd> Ju_a_vec_prev;
  std::vector<Eigen::MatrixXd> Jv_a_vec_prev;

  while ((iter < maxIter) & (err > zeroTol)) {
    // TODO: u_prev and v_prev for the early stopping
    u_prev = u;
    v_prev = v;
    Ju_a_vec_prev = Ju_a_vec;
    Jv_a_vec_prev = Jv_a_vec;
    
    // Julia:
    // u = a ./ (K * v)
    Kv = K * v;
    for (int i = 0; i < a.cols(); ++i) {
      u.col(i) = a.col(i).array() / Kv.col(i).array();
      Ju_a_vec[i] = jacobian_u_wrt_a(a.col(i), K, v.col(i), Jv_a_vec[i]);
    }
    
    // Julia:
    // v = b ./ (K' * u)
    KTu = K.transpose() * u;
    for (int i = 0; i < a.cols(); ++i) {
      v.col(i) = b.array() / KTu.col(i).array();
      Jv_a_vec[i] = jacobian_v_wrt_a(b, K, u.col(i), Ju_a_vec[i]);
    }

    // TODO: early stopping criterion (in case of not converging)
    // check u or v or Kv or KTu is nan or inf
    if (
        u.array().isNaN().any() | u.array().isInf().any() |
          v.array().isNaN().any() | v.array().isInf().any() |
          Kv.array().isNaN().any() | Kv.array().isInf().any() |
          KTu.array().isNaN().any() | KTu.array().isInf().any()
    ) {
      u = u_prev;
      v = v_prev;
      Ju_a_vec = Ju_a_vec_prev;
      Jv_a_vec = Jv_a_vec_prev;
    }
    
    // check R user interrrupt
    R_CheckUserInterrupt();
    
    // regular stopping criterion
    ++iter;
    err = (a - u * Kv).squaredNorm() + (b - v * KTu).squaredNorm();
  }
  
  
  // after the loop, calculate the gradient
  Eigen::VectorXd grad_alpha(alpha.rows());
  for (int i = 0; i < a.cols(); ++i) {
    Eigen::MatrixXd T = u.col(i).asDiagonal() * K * v.col(i).asDiagonal();
    grad_alpha[i] = (T.array() * M.array()).array().sum();
    nabla.col(i) = calc_grad(KM, u.col(i), v.col(i), Ju_a_vec[i], Jv_a_vec[i]);
  }
  
  // calculate gradient wrt `a` and `lbd`
  Eigen::MatrixXd grad_a = nabla * alpha.asDiagonal();
  
  
  // Rcpp::List res = Rcpp::List::create(
  //   Rcpp::Named("u") = u,
  //   Rcpp::Named("v") = v,
  //   Rcpp::Named("K") = K,
  //   Rcpp::Named("gradient_a") = grad_a,
  //   Rcpp::Named("gradient_alpha") = grad_alpha,
  //   Rcpp::Named("iter") = iter
  // );
  // return res;
  return {grad_a, grad_alpha};
}

// batch sample of sinkhorn gradients
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> sinkhorn_batch(
  const Eigen::MatrixXd &a,
  const Eigen::MatrixXd &M,
  const Eigen::MatrixXd &b,
  const Eigen::MatrixXd &alpha,
  const double reg,
  const int maxIter = 1000,
  const double zeroTol = 1e-8
) {
  Eigen::MatrixXd grad_a(a.rows(), a.cols());
  Eigen::VectorXd grad_alpha(alpha.rows(), alpha.cols());
  grad_a.setZero();
  grad_alpha.setZero();
  
  for (int i = 0; i < b.cols(); ++i) {
    auto [grad_a_i, grad_alpha_i] = 
      sinkhorn(a, M, b.col(i), alpha.col(i), reg, maxIter, zeroTol);
    grad_a += grad_a_i;
    grad_alpha += grad_alpha_i;
  }
  
  // take average by batch_size
  grad_a /= b.cols();
  grad_alpha /= b.cols();
  return {grad_a, grad_alpha};
}

//////////////////////////////////////////////////////////////
// core computation 2: given optimized weight and topics
// compute the predicted barycenter
//////////////////////////////////////////////////////////////

Eigen::VectorXd barycenter_single(
    const Eigen::MatrixXd &M,
    const Eigen::VectorXd &b,
    const Eigen::VectorXd &alpha,
    const double &reg,
    const int maxIter = 1000,
    const double zeroTol = 1e-8
) {
  // WTS: a: N * 1
  // M: N * N
  // b: N * K
  // alpha: K * 1
  
  // init
  int iter = 0;
  double err = 1000.;
  
  Eigen::MatrixXd K = (- M / reg).array().exp();
  
  // init u and v
  Eigen::MatrixXd u(b.rows(), b.cols());
  Eigen::MatrixXd v(b.rows(), b.cols());
  Eigen::MatrixXd Kv(b.rows(), b.cols());
  Eigen::MatrixXd KTu(b.rows(), b.cols());
  u.setOnes();
  v.setOnes();
  
  // init empty a for the end result
  Eigen::VectorXd a(b.rows());
  Eigen::MatrixXd Kv_lbd(b.rows(), b.cols());
  
  // init u_prev, v_prev
  Eigen::MatrixXd u_prev;
  Eigen::MatrixXd v_prev;
  
  while ((iter < maxIter) & (err > zeroTol)) {
    KTu = K.transpose() * u;
    for (int i = 0; i < b.cols(); ++i) {
      v.col(i) = b.col(i).array() / KTu.col(i).array();
    }
    
    Kv = K * v;
    for (int i = 0; i < b.cols(); ++i) {
      Kv_lbd.col(i) = Kv.col(i).array().pow(alpha[i]);
    }
    for (int i = 0; i < b.rows(); ++i) {
      a[i] = Kv_lbd.row(i).array().prod();
    }
    
    for (int i = 0; i < b.cols(); ++i) {
      u.col(i) = a.array() / Kv.col(i).array();
    }
    
    // early stopping criterion (in case of not converging)
    // check u or v or Kv or KTu is nan or inf
    if (
        u.array().isNaN().any() | u.array().isInf().any() |
          v.array().isNaN().any() | v.array().isInf().any() |
          Kv.array().isNaN().any() | Kv.array().isInf().any() |
          KTu.array().isNaN().any() | KTu.array().isInf().any()
    ) {
      u = u_prev;
      v = v_prev;
    }
    
    // check R user interrrupt
    R_CheckUserInterrupt();
    
    // regular stopping criterion
    ++iter;
    err = (a - u * Kv).squaredNorm() + (b - v * KTu).squaredNorm();
  }
  
  return a;
}

Eigen::MatrixXd barycenter(
    const Eigen::MatrixXd &M,
    const Eigen::VectorXd &b,
    const Eigen::MatrixXd &Alpha,
    const double &reg,
    const int maxIter = 1000,
    const double zeroTol = 1e-8
) {
  Eigen::MatrixXd docs(M.rows(), Alpha.cols()); // N * M
  for (int j = 0; j < Alpha.cols(); ++j) {
    docs.col(j) = barycenter_single(M, b, Alpha.col(j), reg, maxIter, zeroTol);
  }
  return docs;
}


//////////////////////////////////////////////////////////////
// optimization steps: SGD and Adam
// for each batch, aggregate gradients and optimize
//////////////////////////////////////////////////////////////

// TODO: optimizer function (SGD and Adam)
void optimize(
    Eigen::MatrixXd &parameter,
    const Eigen::MatrixXd &gradient,
    const std::string optimizer,
    const double lr
) {
  if (optimizer == "sgd") {
    // batch-wise update: 
    // `gradient` is the average of gradients in current batch
    parameter = parameter.array() - lr * gradient.array();
  } else if (optimizer == "adam") {
    Rcpp::stop(
      "Optimizer 'adam' not implemented, use either 'sgd' or 'adam' instead!", 
      optimizer
    );
  } else {
    Rcpp::stop(
      "Optimizer %s not implemented, use either 'sgd' or 'adam' instead!", 
      optimizer
    );
  }
  // ensure softmax constraint
  softmax(parameter);
}

Eigen::MatrixXd slice(Eigen::MatrixXd mat, int batch_size, int batch_id) {
  // given batch_size and batch_id, calculate indices to be used for block API
  int batch_start_id = batch_size * batch_id;
  int batch_size_adj = mat.cols() - batch_size - batch_start_id;
  batch_size = batch_size_adj >= 0 ? batch_size : batch_size + batch_size_adj;
  return mat.block(0, batch_start_id, mat.rows(), batch_size);
}

// [[Rcpp::export("wig_cpp")]]
Rcpp::List wig(
  Eigen::Map<Eigen::MatrixXd> docs,    // input docs as matrix: N*M
  Eigen::Map<Eigen::MatrixXd> M,       // distance matrix: N*N
  int num_topics,                      // num of topics: K
  int batch_size,                      // batch_size
  int epochs,                          // num of epochs of input data
  std::string optimizer,               // optimization method: SGD or Adam
  double lr,                           // optimizer learning rate
  double reg,                          // Sinkhorn regularization
  int maxIter = 1000,                  // maximum iteration per Sinkhorn call
  double zeroTol = 1e-8,               // Sinkhorn convergence tolerance
  bool verbose = false                 // whether to print some info
) {
  // input: 
  // docs: N * M
  // batch_size: n
  // first copy R objects as mutatable Eigen objs
  // Eigen::MatrixXd docs2 = docs;
  // Eigen::MatrixXd M2 = M;
  
  // convert R character into C++ std::string
  // std::string opt = Rcpp::as<std::string>(optimizer);
  
  // init some vars
  
  // TODO: split docs into chunks/batches: N*m, N*m, ..., N*m, N*m
  // loop for each chunk/batch
  // update parameters
  // reweight by softmax (to satisfy distribution constraint, i.e. sum to 1)
  
  // 0. init basis `a` and weight `alpha`
  // a     : N * K
  // Alpha : K * M
  Eigen::MatrixXd a = init(docs.rows(), num_topics);
  Eigen::MatrixXd alpha = init(num_topics, docs.cols());
  // make sure `a` and `alpha` satisfy colsum = 1
  softmax(a);
  softmax(alpha);
  
  // outer loop: number of passes of input data
  for (int p = 0; p < epochs; ++p) {
    
    // calculate length of the batches needed
    int batch_num = std::ceil(static_cast<double>(docs.cols()) / batch_size);
    
    for (int i = 0; i < batch_num; ++i) {
      // 1. split docs into vector of Eigen::MatrixXd
      // slice docs and weight matrix using same method (same num of cols)
      Eigen::MatrixXd docs_batch = slice(docs, batch_size, i);
      Eigen::MatrixXd alpha_batch = slice(alpha, batch_size, i);
      
      // 2. calculate gradient from the Sinkhorn loss function
      auto [grad_a, grad_alpha] = 
        sinkhorn_batch(a, M, docs_batch, alpha_batch, reg, maxIter, zeroTol);
      
      // 3. run optimize step with softmax constraint
      optimize(a, grad_a, optimizer, lr);
      optimize(alpha, grad_alpha, optimizer, lr);
      
      // check R user interrrupt
      R_CheckUserInterrupt();
    }
  }
  
  // TODO: recover the reconstructed docs from the alpha and weight
  Eigen::MatrixXd docs_pred = barycenter(M, a, alpha, reg, maxIter, zeroTol);
  
  Rcpp::List res = Rcpp::List::create(
    Rcpp::Named("topics") = a,
    Rcpp::Named("weight") = alpha,
    Rcpp::Named("docs_pred") = docs_pred
  );
  return res;
}
