using JLD
using PyPlot
using StatsBase
using Distributions
using VBMatrixFactorization

include("mil_util.jl")

# inputs for the validation function
verb = true
inputs = Dict()
#inputs["p_vec"] =  [0.01, 0.02, 0.05, 0.1, 0.33, 0.5, 0.75, 0.9] # the vector percentages of known labels 
inputs["p_vec"] =  [0.01, 0.02, 0.05] 
inputs["nclass_iter"] = 100 # how many times should be bags randomly assigned and classification tested over one percentage of known labels
inputs["niter"] = 100 # iterations for vbmf solver
inputs["eps"] = 1e-3 # the convergence limit for vbmf
inputs["solver"] = "sparse" # basic/sparse for non/full ARD on A matrix in vbmf
inputs["H"] = 5 # inner dimension of the factorization
inputs["dataset_name"] = ""
inputs["scale_y"] = true

########################
# DEFINE YOUR io PATHS #
########################
mil_path = "/home/vit/Dropbox/vyzkum/cisco/data/milproblems" # where the MIL .jld files are
output_path = string("/home/vit/Dropbox/vyzkum/cisco/data/vbmf_classification/", inputs["solver"],
    "_", inputs["H"], "_", inputs["nclass_iter"]) # where output is stored

# define which files to use
files = readdir(mil_path)
nfiles = size(files)[1]
file_inds = 1:1 # which MIL files you want to use
# file_inds = 1:nfiles # use all the files

# run the main function
#main("sparse", 1, 50, 1:nfiles)
#main("basic", 2, 50, 1:1)
@time warmup(mil_path)
@time validate_datasets(inputs, file_inds, mil_path, output_path, verb = verb)