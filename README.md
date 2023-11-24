
<!-- README.md is generated from README.Rmd. Please edit that file -->

# NNTools

<!-- badges: start -->

[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
<!-- badges: end -->

The goal of NNTools is to facilitate the use of Machine Learning’
technics wrapping all the pipeline in easy to use functions (`pipe_*`).
The functions take care of the cross-validation, data scaling and return
the predictions (also for rasters), SHAP in a single call.

## Pre-requisites

Install keras and tensorflow (only needed for `pipe_keras` and
`pipe_keras_timeseries`):

``` r
keras::install_keras()
```

## Installation

You can install the development version of NNTools like so:

``` r
# install.packages("remotes")
remotes::install_github("jmaspons/MLTools")

## The package "data.table" (>= 1.14.9) is required.
# data.table::update_dev_pkg() # requires unreleased features for reshaping 3D data
```
