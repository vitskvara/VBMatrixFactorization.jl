using JLD
using StatsBase
using VBMatrixFactorization

include("mil_util.jl")

# inputs for the validation function
verb = true
inputs = Dict()
#inputs["p_vec"] =  [0.01, 0.02, 0.05, 0.1, 0.33, 0.5, 0.75, 0.9] # the vector percentages of known labels 
inputs["p_vec"] =  [] 
inputs["nclass_iter"] = 25 # how many times should be bags randomly assigned and classification tested over one percentage of known labels
inputs["niter"] = 20 # iterations for vbmf solver
inputs["eps"] = 5e-2 # the convergence limit for vbmf
inputs["solver"] = "sparse" # basic/sparse for non/full ARD on A matrix in vbmf
inputs["H"] = 20 # inner dimension of the factorization
inputs["dataset_name"] = ""
inputs["scale_y"] = true # should Y be scaled to standard distribution? 
inputs["scale"] = 10.0 # how are they scaled? - could be better for numerical optimization
inputs["use_cvs"] = true # should cv_indexes be also used?
inputs["diag_var"] = false
inputs["class_alg"] = "dual" # "ols"/"rls"/"vbls"/"min_err"/"lower_bound"/"dual"
inputs["H1"] = 1
inputs["threshold"] = 0.05 # threshold value for min_err/lower_bound classification
inputs["full_cvs"] = true
########################
# DEFINE YOUR io PATHS #
########################
# run this again with no scaling (at least on some subsample)
# run it for Musk1 fwith large H
# rerun for Musk2
mil_path = "/home/vit/Dropbox/vyzkum/cisco/data/milproblems" # where the MIL .jld files are
output_path = string("/home/vit/Dropbox/vyzkum/cisco/data/vbmf_classification/$(inputs["solver"])",
    "_$(inputs["H"])_$(inputs["H1"])_$(inputs["nclass_iter"])") # where output is stored

### axolotl paths ###
#mil_path = "/home/skvara/work/cisco/data/milproblems" # where the MIL .jld files are
#output_path = string("/home/skvara/work/cisco/data/vbmf_classification/$(inputs["solver"])",
#    "_$(inputs["H"])_$(inputs["H1"])_$(inputs["nclass_iter"])") # where output is stored

if inputs["diag_var"]
    output_path = string(output_path, "_het")
else
    output_path = string(output_path, "_hom")
end
output_path = string(output_path, "_$(inputs["class_alg"])")
output_path = string(output_path, "_$(inputs["threshold"])")
if inputs["full_cvs"] == true
   output_path = string(output_path, "_fullcvs") 
else
    output_path = string(output_path, "_cvs")
end 

# define which files to use
files = readdir(mil_path)
nfiles = size(files)[1]
#file_inds = 1:1 # which MIL files you want to use
#file_inds = cat(1, [1, nfiles], 2:(nfiles-1)) # use all the files
# file_inds = nfiles-1:nfiles-1
file_inds = [1, 6]
# run the main function
#main("sparse", 1, 50, 1:nfiles)
#main("basic", 2, 50, 1:1)
@time warmup(mil_path)
@time validate_datasets(inputs, file_inds, mil_path, output_path, verb = verb)

println(string("saving to ", output_path))
#res = load(string(output_path, "/BrownCreeper_sparse_2_1.jld"))
#table_summary(res)

#
#inputs["H"] = 10
#inputs["H1"] = 5
#output_path = string("/home/skvara/work/cisco/data/vbmf_classification/$(inputs["solver"])",
#    "_$(inputs["H"])_$(inputs["H1"])_$(inputs["nclass_iter"])") # where output is stored
#if inputs["diag_var"]
#    output_path = string(output_path, "_het")
#else
#    output_path = string(output_path, "_hom")
#end
#output_path = string(output_path, "_$(inputs["class_alg"])")
#@time validate_datasets(inputs, file_inds, mil_path, output_path, verb = verb)

#plot_statistics(res, save_path = ".")

#file_inds = 6:10
#@time validate_datasets(inputs, file_inds, mil_path, output_path, verb = verb)

#file_inds = 11:15
#@time validate_datasets(inputs, file_inds, mil_path, output_path, verb = verb)

#file_inds = 16:20
#@time validate_datasets(inputs, file_inds, mil_path, output_path, verb = verb)

#file_inds = 21:nfiles
#@time validate_datasets(inputs, file_inds, mil_path, output_path, verb = verb)