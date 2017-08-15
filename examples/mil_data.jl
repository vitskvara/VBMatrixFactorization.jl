using JLD
using PyPlot
using StatsBase
using Distributions
using VBMatrixFactorization

include("mil_util.jl")

########################
# DEFINE YOUR MIL PATH #
########################

mil_path = "/home/vit/Dropbox/vyzkum/cisco/data/milproblems";
files = readdir(mil_path)
println("The directory $mil_path contains the following files:")
for file in files
    println(file)
end
println("")

# inputs for the validation function
inputs = Dict()
inputs["p_vec"] =  [0.01, 0.02, 0.05, 0.1, 0.33, 0.5, 0.75, 0.9] # the vector percentages of known labels 
inputs["nclass_iter"] = 50 # number of iterations over a p_vec element
inputs["niter"] = 200 # iterations for vbmf solver
inputs["eps"] = 1e-3 # the convergence limit for vbmf
inputs["solver"] = "basic" # basic/sparse
inputs["H"] = 1 # inner dimension of the factorization
inputs["dataset_name"] = ""
verb = false
output_path = string("/home/vit/Dropbox/vyzkum/cisco/data/vbmf_classification/", inputs["solver"], "_", inputs["H"], "_", inputs["nclass_iter"])
mkpath(output_path)

# loop through all the files, train them using vbmf, then validate the classification using a testing dataset
# then save the results
tic(); # for performance measurement
for file in files
    dataset_name, suf = split(file, ".")
    if suf != "jld" # if the file is not a .jld file, move on
        continue
    end

    println("Processing file $file...")

    # load the data
    data = load(string(mil_path, "/", file));

    # perform testing of the classification on the dataset
    inputs["dataset_name"] = dataset_name
    res_mat = validate_dataset(data, inputs, verb = verb)

    # save the outputs and inputs
    fname = string(dataset_name, "_", inputs["solver"], "_", inputs["H"], "_", inputs["nclass_iter"])
    save("$output_path/$fname.jld", "res_mat", res_mat, "inputs", inputs)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   

    println("Done.")
    println()
end
toc()