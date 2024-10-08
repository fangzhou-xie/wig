# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

wig_cpp <- function(docs, C, num_topics, batch_size, epochs, loss = 1L, optimizer = 1L, lr = .001, beta1 = .9, beta2 = .999, eps = 1e-8, reg = 0.1, maxIter = 1000L, zeroTol = 1e-8, verbose = FALSE) {
    .Call(`_wig_wig_cpp`, docs, C, num_topics, batch_size, epochs, loss, optimizer, lr, beta1, beta2, eps, reg, maxIter, zeroTol, verbose)
}

euclidean_cpp <- function(a) {
    .Call(`_wig_euclidean`, a)
}

