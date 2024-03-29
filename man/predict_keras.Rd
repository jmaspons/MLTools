% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/keras_helpers.R
\name{predict_keras}
\alias{predict_keras}
\title{Predict with keras}
\usage{
predict_keras(
  modelNN,
  predInput,
  maskNA = NULL,
  scaleInput = FALSE,
  col_means_train,
  col_stddevs_train,
  batch_size = NULL,
  filename = "",
  tempdirRaster = NULL,
  nCoresRaster = 2
)
}
\arguments{
\item{modelNN}{an \code{\link[keras:keras]{keras::keras()}} model.}

\item{predInput}{\code{data.frame} or \code{raster} with colnames or layer names matching the expected input for modelRF.}

\item{maskNA}{value to assign to \code{NA}s after scaling and passed to \code{\link[keras:layer_masking]{keras::layer_masking()}}.}

\item{scaleInput}{if \code{TRUE}, scale \code{predInput} with \code{col_means_train} and col \code{col_stddevs_train}.}

\item{col_means_train}{the original mean of the \code{predInput} columns.}

\item{col_stddevs_train}{the original sd of the \code{predInput} columns.}

\item{batch_size}{for fit and predict functions. The bigger the better if it fits your available memory. Integer or "all".}

\item{filename}{the file to write the raster predictions.}

\item{tempdirRaster}{path to a directory to save temporal raster files.}

\item{nCoresRaster}{number of cores used for parallelized raster cores. Use half of the available cores by default.}
}
\description{
Predict with keras
}
