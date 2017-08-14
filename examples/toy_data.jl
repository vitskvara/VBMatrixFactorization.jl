using JLD
using PyPlot
using StatsBase
using Distributions
using VBMatrixFactorization

function toy_matrix(L, M, H, std)
    # B is a random matrix whose columns we select from
    B = randn(L, H) 
    # A is the sparse matrix that selects the columns from A
    A = zeros(M, H)
    for m in 1:M
        A[m, rand(1:H)] = 1
    end
    
    Y = B*A' + rand(Normal(0, std), L, M)
    return Y, A, B
end

L = 10
M = 20
H = 10
Y_toy, A_toy, B_toy = toy_matrix(L, M, H, 0.05);

# first, initialize a variable of the vbmf_parameters type
# then use it to run the basic vbmf algorithm
# here, the whole run is saved into the data_path folder
# if desc is not specified, then the subfolder with the actual data 
# is named using date and time

println(" ----------- Basic VB Matrix factorization ---------------- ")
println("")
params_init = VBMatrixFactorization.vbmf_init(L, M, H, ca = 0.1, cb = 0.1, sigma2 = 0.1);
data_path = "./data"
res_vbmf = VBMatrixFactorization.vbmf(Y_toy, params_init, 100, est_covs = true, est_var = true, verb = true, 
    logdir = data_path, desc = "vbmf_test");
err = norm(Y_toy - res_vbmf.YHat)
println("||Y - Yhat|| = $err")

# this is how the log is loaded and used
vbmf_log, data, priors, start_params = VBMatrixFactorization.load_log(string(data_path,"/vbmf_test"));
params_vbmf = VBMatrixFactorization.vbmf_parameters(); # dummy variable to store individual step data

iter = 3
time_slice = VBMatrixFactorization.extract_params!(vbmf_log, iter, params_vbmf);
err = norm(Y_toy - time_slice.YHat)
println("in iteration number $iter, the error was ||Y - Yhat|| = $err")
println("")

# vbmf_sparse usage
println(" ----------- VB Matrix factorization with sparse A ---------------- ")
println("")
params_sparse_init = VBMatrixFactorization.vbmf_sparse_init(L, M, H, ca = 0.1, cb = 0.1, sigma2 = 0.1);
#params_sparse_init.AHat = A_toy
sparse_vbmf = VBMatrixFactorization.vbmf_sparse(Y_toy, params_sparse_init, 100, est_var = true, verb = true, 
    logdir = data_path, desc = "sparse_test");
err = norm(Y_toy - sparse_vbmf.YHat)
println("||Y - Yhat|| = $err")
println("")
# compare the results

println(" original A          basic reconstruction          sparse reconstruction with true start")
for m in 1:M
    println(A_toy[m,:],"   ", res_vbmf.AHat[m,:], "   ", sparse_vbmf.AHat[m,:])
end
