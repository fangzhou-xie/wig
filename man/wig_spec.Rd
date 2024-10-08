% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/wig.R
\name{wig_spec}
\alias{wig_spec}
\title{WIG model specification}
\usage{
wig_spec(
  wig_spec = list(num_topics = 4, batch_size = 64, epochs = 5),
  tokenizer_spec = list(stopwords = stopwords::stopwords("en")),
  word2vec_spec = list(type = "cbow", stopwords = stopwords::stopwords("en"), dim = 10,
    iter = 20),
  sinkhorn_spec = list(reg = 0.1, maxIter = 1000, zeroTol = 1e-08),
  optimizer_spec = list(optimizer = c("sgd", "adam"), lr = 0.005)
)
}
\description{
WIG model specification
}
