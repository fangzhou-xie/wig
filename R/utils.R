

euclidean <- function(embedding) {
  dist_mat <- euclidean_cpp(embedding)
  colnames(dist_mat) <- rownames(embedding)
  rownames(dist_mat) <- rownames(embedding)
  dist_mat
}

doc2dist <- function(doc_tokens, dict) {
  docmat <- doc2dist_cpp(doc_tokens, dict)
  rownames(docmat) <- dict
  docmat
}

# docdict2dist <- function(doc_tokens, dict) {
#   docdist <- matrix(0, nrow = length(dict), ncol = length(doc_tokens))
#   # docdist: (N+1)*M
#   # additional token for all other tokens that can't be found in vocab
#   
#   for (j in 1:length(doc_tokens)) {
#     doc <- doc_tokens[[j]]
#     for (tk in doc) {
#       # TODO: if token not found in the dict (out of vocab)
#       tk_ind <- if (tk %in% dict) which(tk == dict) else which("" == dict)
#       docdist[tk_ind, j] <- docdist[tk_ind, j] + 1
#     }
#     docdist[,j] <- docdist[,j] / sum(docdist[,j])
#   }
#   # rownames(docdist) <- dict
#   docdist
# }

