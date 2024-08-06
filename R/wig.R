
# main WIG model function

#' WIG model specification
#' 
#' @export 
wig_spec <- function(
    wig_spec = list(             # WIG model parameters
      num_topics = 4,
      batch_size = 64,
      epochs = 5
    ),
    tokenizer_spec = list(       # `tokenizers::tokenize_words` args
      stopwords = stopwords::stopwords("en")
    ),
    word2vec_spec = list(        # `word2vec::word2vec` args
      type = "cbow",
      stopwords = stopwords::stopwords("en"),
      dim = 10,
      iter = 20
    ),      
    sinkhorn_spec = list(        # sinkhorn algorithm parameters
      reg = .1,
      maxIter = 1000,
      zeroTol = 1e-8
    ),
    optimizer_spec = list(
      optimizer = c("sgd", "adam"),
      lr = 0.005
    )
) {
  # combine wig_spec, sinkhorn_spec, optimizer_spec into one list
  wig_spec_combined <- append(wig_spec, sinkhorn_spec)
  wig_spec_combined <- append(wig_spec_combined, optimizer_spec)
  
  list(
    wig_spec = wig_spec_combined, 
    tokenizer_spec = tokenizer_spec,
    word2vec_spec = word2vec_spec
  )
}

#' WIG model training
#' 
#' @export
wig <- function(dates, docs, spec = wig_spec(), 
                group_time = "months",
                svd_method = c("topics", "docs"),
                standardize = TRUE, verbose = FALSE) {
  
  # TODO: metaprogramming for columns of dataframe/tibble
  # date_col
  # docs_col
  
  # dts: vector of type `datetime`
  # docs: vector of character
  # group_time: time level to group the results: yearly, monthly, weekly, daily
  
  if (verbose) {
    print("")
  }
  
  # 1. tokenize documents
  tokenizer_spec <- append(list(x = docs), spec$tokenizer_spec)
  docs_token <- do.call(tokenizers::tokenize_words, args = tokenizer_spec)
  
  # 2. train word2vec model for embeddings
  # append word2vec parameters into the input docs
  word2vec_spec <- append(list(x = docs_token), spec$word2vec_spec)
  word2vec_model <- do.call(word2vec::word2vec, args = word2vec_spec)
  word2vec_embeddings <- as.matrix(word2vec_model)
  dict <- rownames(word2vec_embeddings)
  dist_mat <- euclidean(word2vec_embeddings)
  
  # 3. prepare wig model spec
  # convert input tokenized docs into word distributions
  docs_mat <- docdict2dist(docs_token, dict)
  wig_spec <- append(list(docs = docs_mat, M = dist_mat), spec$wig_spec)
  
  # 4. train WIG model
  wig_model <- do.call(wig_cpp, args = wig_spec)
  
  # get topics and weights
  topics <- wig_model$topics
  weight <- wig_model$weight
  colnames(topics) <- colnames(word2vec_embeddings)

  
  # 5. generate index
  if (svd_method == "topics") {
    # 5.1 original WIG
    # topics: N * K
    # weight: K * M
    
    topics_svd <- t(tsvd(t(topics), k = 1))
    wig_index <- c(topics_svd %*% weight)
  } else if (svd_method == "docs") {
    # 5.2 modified: SVD on reconstructed docs
    docs_pred <- wig_model$docs_pred
    wig_index <- c(tsvd(t(docs_pred), k = 1))
  } else {
    stop("`svd_method` not implemented")
  }
  
  # 6. regroup by time
  dts_end <- xts::endpoints(dts, on = group_time)
  # wig_index_list <- vector(mode = "list", length = length(dts_end) - 1)
  dts_vec <- numeric(length(dts_end) - 1)
  wig_index_vec <- numeric(length(dts_end) - 1)
  for (i in 2:length(dts_end)) {
    dts_vec[i-1] <- dts[dts_end[i-1]+1]
    wig_index_vec[i-1] <- sum(wig_index[(dts_end[i-1]+1):dts_end[i]])
  }
  
  # return the time re-grouped WIG index
  # TODO: add other functionalities, i.e. display topic tokens
  data.frame(Date = dts_vec, WIG = standardize_cpp(wig_index_vec))
}


# TODO: WIG model prediction? other utilities (get topics etc.)
