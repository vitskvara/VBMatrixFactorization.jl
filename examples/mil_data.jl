using JLD
using PyPlot
using StatsBase
using Distributions
using VBMatrixFactorization

include("mil_util.jl")

mil_path = "../../data/milproblems/";
readdir(mil_path)