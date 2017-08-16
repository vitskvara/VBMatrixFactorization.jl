using JLD
using PyPlot
using StatsBase
using Distributions
using VBMatrixFactorization

include("mil_util.jl")

# inputs
solver = "basic" # basic/sparse for non/full ARD on A matrix in vbmf
H = 2 # inner dimension of the factorization
nclass_iter = 10 # how many times should be bags randomly assigned and classification tested over one percentage of known labels

########################
# DEFINE YOUR io PATHS #
########################
mil_path = "/home/vit/Dropbox/vyzkum/cisco/data/milproblems" # where the MIL .jld files are
output_path = string("/home/vit/Dropbox/vyzkum/cisco/data/vbmf_classification/$solver_$H_$nclass_iter") # where output is stored

files = readdir(mil_path)
nfiles = size(files)[1]
file_inds = 1:1 # which MIL files you want to use

#main("sparse", 1, 50, 1:nfiles)
#main("basic", 2, 50, 1:1)
validate_datasets(solver, H, nclass_iter, file_inds, mil_path, output_path)
