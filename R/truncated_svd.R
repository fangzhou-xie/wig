
# implement truncated SVD for dimension reduction
#' 
tsvd <- function(M, k = 1) {
  M_svd <- svd(M)
  M_svd$u[,1:k] %*% diag(M_svd$d[1:k], k, k)
}
