"""
   vbmf_parameters

Compound type for vbmf computation. Contains the following fields:\n 
    
    L::Int - number of rows of the original matrix 
    M::Int - number of columns of the original matrix
    H::Int - internal product dimension
    H1::Int - number of columns of B that should belong to a contaminated class
    labels::Array{Int64, 1} - which columns of Y are labeled as non-contaminated
    AHat::Array{Float64, 2} - mean value of A, size (M, H)
    BHat::Array{Float64, 2} - mean value of B, size (L, H)
    SigmaA::Array{Float64, 2} - covariance of A, size (H, H)
    SigmaB::Array{Float64, 2} - covariance of B, size (H, H)
    CA::Array{Float64, 2} - prior covariance of A, size (H, H)
    CB::Array{Float64, 2} - prior covariance of B, size (H, H)
    invCA::Array{Float64, 2} - inverse of cA
    invCB::Array{Float64, 2} - inverse of cB
    sigma::Float64 - variance of data
    YHat::Array{Float64, 2} - estimate of Y, size (L, M)
"""
type vbmf_parameters
    L::Int
    M::Int
    H::Int
    H1::Int
    labels::Array{Int64,1}
    AHat::Array{Float64, 2}
    BHat::Array{Float64, 2}
    SigmaA::Array{Float64, 2}
    SigmaB::Array{Float64, 2}
    CA::Array{Float64, 2}
    CB::Array{Float64, 2}
    invCA::Array{Float64, 2}
    invCB::Array{Float64, 2}
    sigma2::Float64
    YHat::Array{Float64, 2}
    
    vbmf_parameters() = new()
end

"""
    vbmf_init(L::Int, M::Int, H::Int; ca::Float64, cb::Float64, sigma::Float64, H1::Int = 0, 
    labels::Array{Int64,1} = Array{Int64,1}())

Returns an initialized structure of type vbmf_parameters.
"""
function vbmf_init(Y::Array{Float64,2}, H::Int; ca::Float64 = 1.0, cb::Float64 = 1.0, sigma2::Float64 = 1.0, 
    H1::Int = 0, labels::Array{Int64,1} = Array{Int64,1}())
    params = vbmf_parameters()
    L, M = size(Y)

    params.L, params.M = L, M 
    params.H = H
    params.H1 = H1
    params.labels = labels

    params.AHat = randn(M, H)
    # now set zeroes to where the columns of Y labeled as non-infected are
    # in A, these are rows
    params.AHat[labels, end-H1+1:end] = 0.0
    params.BHat = randn(L, H)
    params.SigmaA = zeros(H, H)
    params.SigmaB = zeros(H, H)
    params.CA = ca*eye(H)
    params.CB = cb*eye(H)
    params.invCA = inv(params.CA)
    params.invCB = inv(params.CB)
    params.sigma2 = sigma2
    params.YHat = params.BHat*params.AHat'
    
    return params
end

"""
   copy(params_in::vbmf_parameters)

Copy function for vbmfa_parameters. 
"""
function copy(params_in::vbmf_parameters)
    params = vbmf_parameters()
    
    for field in fieldnames(params_in)
        setfield!(params, field, getfield(params_in, field))
    end
    
    return params
end

"""
    updateA!(Y::Array{Float64,2}, params::vbmf_parameters)

Updates mean and covariance of the A matrix.
"""
function updateA!(Y::Array{Float64,2}, params::vbmf_parameters)
    params.SigmaA = params.sigma2*inv(params.BHat'*params.BHat + 
        params.L*params.SigmaB + params.sigma2*params.invCA)
    params.AHat =  Y'*params.BHat*params.SigmaA/params.sigma2
    # now set zeroes to where the columns of Y labeled as non-infected are
    # in A, these are rows
    params.AHat[params.labels, end-params.H1+1:end] = 0.0
end

"""
    updateB!(Y::Array{Float64,2}, params::vbmf_parameters)

Updates mean and covariance of the B matrix.
"""
function updateB!(Y::Array{Float64,2}, params::vbmf_parameters)
    params.SigmaB = params.sigma2*inv(params.AHat'*params.AHat + 
        params.M*params.SigmaA + params.sigma2*params.invCB)
    params.BHat =  Y*params.AHat*params.SigmaB/params.sigma2
end

"""
    updateYHat!(params::vbmf_parameters)

Updates estimate of Y.
"""
function updateYHat!(params::vbmf_parameters)
    params.YHat = params.BHat*params.AHat'
end

"""
    updateCA!(params::vbmf_parameters)

Updates the estimate of CA.
"""
function updateCA!(params::vbmf_parameters)
    for h in 1:params.H
        params.CA[h,h] = norm2(params.AHat[:,h])/params.M + params.SigmaA[h,h]
    end
    params.invCA = inv(params.CA)
end

"""
    updateCB!(params::vbmf_parameters)

Updates the estimate of CB.
"""
function updateCB!(params::vbmf_parameters)
    for h in 1:params.H
        params.CB[h,h] = norm2(params.BHat[:,h])/params.L + params.SigmaB[h,h]
    end
    params.invCB = inv(params.CB)
end

"""
    updateSigma2!(Y::Array{Float64,2}, params::vbmf_parameters)

Updates estimate of sigma^2, the measurement noise.
"""
function updateSigma2!(Y::Array{Float64,2}, params::vbmf_parameters)
    params.sigma2 = (norm2(Y) - trace(2*Y'*params.BHat*params.AHat') + 
        trace((params.AHat'*params.AHat + params.M*params.SigmaA)*
            (params.BHat'*params.BHat + params.L*params.SigmaB)))/(params.L*params.M)
end

"""
    vbmf!(Y::Array{Float64, 2}, params::vbmf_parameters, niter::Int, eps::Float64 = 1e-6, est_covs = false, est_var = false)

Computes variational bayes matrix factorization of Y = AB' + E. Independence of A and B is assumed. 
Estimation of prior covariance cA and cB and of variance sigma can be turned on and off. Estimates 
of cA and cB are empirical.
The prior model is following:
    
    p(Y|A,B) = N(Y|BA^T, sigma^2*I)
    p(A) = N(A|0, C_A), C_A = diag(c_a)
    p(B) = N(B|0, C_B), C_B = diag(c_b)

The params argument with initialized data is modified and contains the resulting estimates after the 
algorithm stops.
"""
function vbmf!(Y::Array{Float64, 2}, params::vbmf_parameters, niter::Int; eps::Float64 = 1e-6, est_covs::Bool = false, 
    est_var::Bool = false, logdir = "", desc = "", verb = false)
    priors = Dict()

    # create the log dictionary
    log = false
    if logdir !=""
        log = true
        logVar = create_log(params)
    end

    # choice of convergence control variable
    convergence_var = :BHat
    old = getfield(params, convergence_var)
    d = eps + 1.0 # delta
    i = 1

    # run the loop for a given number of iterations
    while ((i <= niter) && (d > eps))
        updateA!(Y, params)
        updateB!(Y, params)

        if est_covs
            updateCA!(params)
            updateCB!(params)
        end

        if est_var
            updateSigma2!(Y, params)
        end

        if log
            update_log!(logVar, params)
        end

        # check convergence
        d = delta(getfield(params, convergence_var), old)
        old = getfield(params, convergence_var)
        i += 1
    end    

    # finally, compute the estimate of Y
    updateYHat!(params)

    # convergence info
    if verb
        print("Factorization finished after ", i-1, " iterations, eps = ", d, "\n")
    end
    
    # save inputs and outputs
    if log
        println("Saving outputs and inputs under ", logdir, "/")
        save_log(logVar, Y, priors, logdir, desc = desc)
    end

    return params
end

"""
    vbmf(Y::Array{Float64, 2}, params_in::vbmf_parameters, niter::Int, eps::Float64 = 1e-6, est_covs = false, est_var = false)

Calls vbmf_sparse!() but copies the params_in argument so that it is not modified and can be reused.
"""
function vbmf(Y::Array{Float64, 2}, params_in::vbmf_parameters, niter::Int; eps::Float64 = 1e-6, est_covs::Bool = false, 
    est_var::Bool = false, logdir = "", desc = "", verb = false)
    # copy the input params
    params = copy(params_in)

    # run the algorithm
    vbmf!(Y, params, niter, eps = eps, est_covs = est_covs, est_var = est_var, 
        logdir = logdir, desc = desc, verb = verb)

    return params
end