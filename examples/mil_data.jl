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

data = load(string(mil_path, "/BrownCreeper.jld"))