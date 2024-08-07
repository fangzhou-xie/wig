
# test the wig model

library(tidyverse)
library(wig)

headlines <- wig::headlines

head(headlines)
names(headlines)

sen <- "this is a sentence"
tok <- tokenizers::tokenize_words(sen)
emb <- word2vec::word2vec(tok)

toks <- tokenizers::tokenize_words(headlines$headlines[1:1000])
model <- word2vec::word2vec(toks)

model_mat <- as.matrix(model)
model_mat[1:10,1:8]

"another" %in% rownames(model_mat)

predict(model, newdata = c("another"), type = "embedding")
predict(model, newdata = c("another one"), type = "embedding")


class(model)


Rcpp::sourceCpp(
  here::here("src", "dsinkgrad.cpp")
)


docs <- wig::headlines$headlines[1:20]
docs_token <- tokenizers::tokenize_words(docs)

word2vec_model <- word2vec::word2vec(docs_token, dim = 50, min_count = 5)
word2vec_embed <- rbind(as.matrix(word2vec_model), rep(0, 50))

word2vec_embed[1:10,1:8]

dict <- rownames(word2vec_embed)
dist_mat <- wig:::euclidean(word2vec_embed)

docs_mat <- wig:::docdict2dist(docs_token, dict)

Rcpp::sourceCpp("src/dsinkgrad.cpp")
wig_model <- wig_cpp(docs_mat, dist_mat, 
                     num_topics = 6, batch_size = 10, epochs = 2, 
                     maxIter = 2, reg = 1,
                     optimizer = 1L, verbose = TRUE)

wig_model
topics <- wig_model$topics
rownames(topics) <- rownames(word2vec_embed)

# TODO: log-domain computation with gradient (numerically stable)




docs <- cbind(
  c(.2,.2,.6),
  c(.2,.1,.7),
  c(.6,.2,.2)
)
C <- rbind(c(.1,.3,.2),c(.5,.1,.5),c(.4,.2,.1))

Rcpp::sourceCpp("src/dsinkgrad.cpp")

wig_model <- wig_cpp(docs, C, num_topics = 2, batch_size = 1, epochs = 10, optimizer = 1L)
wig_model
wig_model$topics
wig_model$weight
wig_model$docs_pred




Rcpp::cppFunction("arma::mat softmax_grad(arma::vec v) {
  arma::vec y = exp(v) / accu(exp(v));
  return diagmat(y) - y * y.t();
}", depends = "RcppArmadillo")

x <- c(1,2,3)
softmax_grad(x)

wig_model <- wig_cpp(
  docs_mat, 
  dist_mat,
  2, 4, 4
)


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



Rcpp::sourceCpp(here::here("src", "euclidean.cpp"))
# M <- matrix(c(.1, .5, .5, .1), ncol = 2)

dist_mat <- euclidean_cpp(word2vec_embed)

sum((word2vec_embed[1,] - word2vec_embed[2,]) ^ 2)






library(fastText)

printUsage()
printNNUsage()






# write the docs into temp dir 
temp_doc_path <- file.path(tempdir(), "doc.txt")
write.csv(headlines$headlines, temp_doc_path)

list_params = list(command = 'supervised',
                   lr = 0.1,
                   dim = 200,
                   input = temp_doc_path,
                   output = file.path(tempdir(), "model"),
                   verbose = 2,
                   thread = 1)
res = fasttext_interface(list_params,
                         path_output = file.path(tempdir(),"model_logs.txt"),
                         MilliSecs = 100)

fs::dir_ls(file.path(tempdir()))

# # read the embedding vector in df
# embed_df <- read.delim(
#   file.path(tempdir(), "model.vec"),
#   sep = " ",
#   skip = 1
# )
# 
# readLines(file.path(tempdir(), "model.vec"), n = 3)
# 
# temp_test_file <- file.path(tempdir(), "temp_text.txt")
# writeLines("the", temp_test_file)

list_params <- list(
  command = "predict",
  model = file.path(tempdir(), "model.bin"),
  test_data = temp_test_file
)
res = fasttext_interface(list_params, path_output = file.path(tempdir(), "predict.txt"))


X <- cbind(
  matrix(rnorm(20), nrow = 5, ncol = 4),
  rep(0, 5)
)

eigen(t(X) %*% X)
