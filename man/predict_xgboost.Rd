% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/xgboost_helpers.R
\name{predict_xgboost}
\alias{predict_xgboost}
\title{Predict with xgboost}
\usage{
predict_xgboost(
  modelXGB,
  predInput,
  scaleInput = FALSE,
  col_means_train,
  col_stddevs_train,
  filename = "",
  tempdirRaster = NULL,
  nCoresRaster = 2
)
}
\arguments{
\item{modelXGB}{an \code{\link[xgboost:xgb.train]{xgboost::xgboost()}} model.}

\item{predInput}{\code{data.frame} or \code{raster} with colnames or layer names matching the expected input for modelRF.}

\item{scaleInput}{if \code{TRUE}, scale \code{predInput} with \code{col_means_train} and col \code{col_stddevs_train}.}

\item{col_means_train}{the original mean of the \code{predInput} columns.}

\item{col_stddevs_train}{the original sd of the \code{predInput} columns.}

\item{filename}{the file to write the raster predictions.}

\item{tempdirRaster}{path to a directory to save temporal raster files.}

\item{nCoresRaster}{number of cores used for parallelized raster cores. Use half of the available cores by default.}
}
\description{
Predict with xgboost
}
