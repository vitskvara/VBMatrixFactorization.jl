"""
   vbmf_sparse_parameters

Compound type for vbmf_sparse computation. Contains the following fields:\n 
    
    L::Int - number of rows of the original matrix 
    M::Int - number of columns of the original matrix
    H::Int - internal product dimension
    H1::Int - number of columns of B that should belong to a contaminated class
    labels::Array{Int64, 1} - which columns of Y are labeled as non-contaminated

    AHat::Array{Float64, 2} - mean value of A, size (M, H)
    SigmaA::Array{Float64, 2} - covariance of A, size (H, H)
    AVecHat::Array{Float64, 1} - mean value of vec(A), size (MH,1)
    SigmaAVec::Array{Float64, 2} - covariance of vec(A), size (MH, MH)
    invSigmaAVec::Array{Float64, 2} - inverse of covariance of vec(A), size (MH, MH)
    AVecTAvec::::Array{Float64, 1} - contains mean values of vec(A)_ij^2, size (MH,1)
    
    BHat::Array{Float64, 2} - mean value of B, size (L, H)
    SigmaBHat::Array{Float64, 2} - covariance of B, size (H, H)
    Gamma::Array{Float64, 2} - sparse matrix with blocks of b_ij*eye(M), size (LM, HM)
    GammaTGamma::Array{Float64, 2} - mean value of Gamma^T Gamma, size (HM, HM)

    CA::Array{Float64, 2} - prior covariance of vec(A), size (MH, MH)
    invCA::Array{Float64, 2} - inverse of cA
    alpha0::Float64 - shape of CA gamma prior
    beta0::Float64 - scale of CA gamma prior
    alpha::Float64 - shape of CA gamma posterior
    beta::Array{Float64,1} - scale of CA gamma posterior, size (MH, 1)

    CB::Array{Float64, 2} - prior covariance of B, size (H, H)
    invCB::Array{Float64, 2} - inverse of cB

    sigma::Float64 - variance of data
    YHat::Array{Float64, 2} - estimate of Y, size (L, M)
"""
type vbmf_sparse_parameters
    L::Int
    M::Int
    H::Int
    H1::Int
    labels::Array{Int64,1}

    AHat::Array{Float64, 2}
    SigmaA::Array{Float64, 2}
    AVecHat::Array{Float64, 1}
    SigmaAVec::Array{Float64, 2}
    invSigmaAVec::Array{Float64, 2}
    AVecTAvec::Array{Float64, 1}    
    
    BHat::Array{Float64, 2}
    SigmaB::Array{Float64, 2}
    Gamma::Array{Float64, 2}
    GammaTGamma::Array{Float64, 2}

    CA::Array{Float64, 2}
    invCA::Array{Float64, 2}
    alpha0::Float64
    beta0::Float64
    alpha::Float64
    beta::Array{Float64,1}

    CB::Array{Float64, 2}
    invCB::Array{Float64, 2}

    sigma2::Float64
    YHat::Array{Float64, 2}
    
    vbmf_sparse_parameters() = new()
end

"""
    vbmf_sparse_init(L::Int, M::Int, H::Int; ca::Float64 = 1.0, alpha0::Float64 = 1e-10,
    beta0::Float64 = 1e-10, cb::Float64 = 1.0, sigma2::Float64 = 1.0, H1::Int = 0, 
    labels::Array{Int64,1} = Array{Int64,1}())

Returns an initialized structure of type vbmf_sparse_parameters.
"""
function vbmf_sparse_init(L::Int, M::Int, H::Int; ca::Float64 = 1.0, alpha0::Float64 = 1e-10,
 beta0::Float64 = 1e-10, cb::Float64 = 1.0, sigma2::Float64 = 1.0, H1::Int = 0, labels::Array{Int64,1} = Array{Int64,1}())
    params = vbmf_sparse_parameters()

    params.L = L
    params.M = M
    params.H = H
    params.H1 = H1
    params.labels = labels

    params.AHat = randn(M, H)
    # now set zeroes to where the columns of Y labeled as non-infected are
    # in A, these are rows
    params.AHat[labels, end-H1+1:end] = 0.0
    params.SigmaA = zeros(H, H)
    params.AVecHat = reshape(params.AHat, M*H)
    params.SigmaAVec = eye(M*H, M*H)
    params.invSigmaAVec = eye(M*H, M*H)
    params.AVecTAvec = params.AVecHat.^2

    params.BHat = randn(L, H)
    params.SigmaB = zeros(H, H)
    params.Gamma = zeros(L*M, H*M)
    fill_gamma!(params)

    params.CA = ca*eye(M*H)
    params.invCA = 1/ca*eye(M*H)
    params.alpha0 = alpha0
    params.beta0 = beta0
    params.alpha = alpha0
    params.beta = beta0*ones(M*H)

    params.CB = cb*eye(H)
    params.invCB = 1/cb*eye(H)

    params.sigma2 = sigma2
    params.YHat = params.BHat*params.AHat'
    
    return params
end

"""
   copy(params_in::vbmf_sparse_parameters)

Copy function for vbmfa_sparse_parameters. 
"""
function copy(params_in::vbmf_sparse_parameters)
    params = vbmf_sparse_parameters()
    
    for field in fieldnames(params_in)
        setfield!(params, field, copy(getfield(params_in, field)))
    end
    
    return params
end

"""
   fill_gamma!(params::vbmf_sparse_parameters)

Fills the Gamma matrix using current values of BHat. Gamma 
composes of blocks of diagonal matrices with elements of BHat on the diagonal.
"""
function fill_gamma!(params::vbmf_sparse_parameters)
    for l in 1:params.L
        for h in 1:params.H
            params.Gamma[((l-1)*params.M+1):l*params.M, ((h-1)*params.M+1):h*params.M] = eye(params.M)*params.BHat[l,h]
        end
    end
    params.GammaTGamma = params.Gamma'*params.Gamma
    for h in 1:params.H
        params.GammaTGamma[((h-1)*params.M+1):h*params.M, ((h-1)*params.M+1):h*params.M] += eye(params.M)*params.SigmaB[h,h]*params.L
    end
end

"""
    updateA!(Y::Array{Float64,2}, params::vbmf_sparse_parameters; full_cov::Bool)

Updates mean and covariance of vec(A) and also of the A matrix. If full_cov is true, 
then inverse of full covariance matrix is computed, otherwise just the diagonal is estimated.
"""
function updateA!(Y::Array{Float64,2}, params::vbmf_sparse_parameters; full_cov::Bool = false)
    params.invSigmaAVec = 1/params.sigma2*params.GammaTGamma + params.CA
    if full_cov
        params.SigmaAVec = inv(params.invSigmaAVec)
    else
        params.SigmaAVec = diagm(ones(params.M*params.H)./diag(params.invSigmaAVec))
    end
    params.AVecHat = 1/params.sigma2*params.SigmaAVec*params.Gamma'*reshape(Y, params.L*params.M)
    # now set zeroes to where the columns of Y labeled as non-infected are
    # in A, these are rows
    params.AHat = reshape(params.AVecHat, params.M, params.H)
    params.AHat[params.labels, end-params.H1+1:end] = 0.0
    params.AVecHat = reshape(params.AHat, params.M*params.H)
    params.AVecTAvec = params.AVecHat.^2 + diag(params.SigmaAVec)

    # this should be revisited !!!!!
    for h in 1:params.H
        params.SigmaA[h,h] = mean(diag(params.SigmaAVec)[((h-1)*params.M+1):h*params.M])
    end
end

"""
    updateB!(Y::Array{Float64,2}, params::vbmf_sparse_parameters)

Updates mean and covariance of the B matrix.
"""
function updateB!(Y::Array{Float64,2}, params::vbmf_sparse_parameters)
    params.SigmaB = params.sigma2*inv(params.AHat'*params.AHat + 
        params.M*params.SigmaA + params.sigma2*params.invCB)
    params.BHat =  Y*params.AHat*params.SigmaB/params.sigma2
    # also, update gamma
    fill_gamma!(params)
end

"""
    updateYHat!(params::vbmf_sparse_parameters)

Updates estimate of Y.
"""
function updateYHat!(params::vbmf_sparse_parameters)
    params.YHat = params.BHat*params.AHat'
end

"""
    updateCA!(params::vbmf_sparse_parameters)

Updates the estimate of CA.
"""
function updateCA!(params::vbmf_sparse_parameters)
    params.alpha = params.alpha0 + 0.5
    params.beta = params.beta0*ones(params.M*params.H) + 1/2*params.AVecTAvec
    params.CA = diagm(params.alpha*ones(params.M*params.H)./params.beta)
    params.invCA = diagm(params.beta./(params.alpha*ones(params.M*params.H)))
end

"""
    updateCB!(params::vbmf_sparse_parameters)

Updates the estimate of CB.
"""
function updateCB!(params::vbmf_sparse_parameters)
    for h in 1:params.H
        params.CB[h,h] = norm2(params.BHat[:,h])/params.L + params.SigmaB[h,h]
    end
    params.invCB = inv(params.CB)
end

"""
    updateSigma2!(Y::Array{Float64,2}, params::vbmf_sparse_parameters)

Updates estimate of sigma^2, the measurement noise.
"""
function updateSigma2!(Y::Array{Float64,2}, params::vbmf_sparse_parameters)
    params.sigma2 = (norm2(Y) - trace(2*Y'*params.BHat*params.AHat') + 
        trace((params.AHat'*params.AHat + params.M*params.SigmaA)*
            (params.BHat'*params.BHat + params.L*params.SigmaB)))/(params.L*params.M)
end

"""
    vbmf(Y::Array{Float64, 2}, params_in::vbmf_sparse_parameters, niter::Int, eps::Float64 = 1e-6, est_covs = false, 
    est_var = false, full_cov::Bool = false, logdir = "", desc = "")

Computes variational bayes matrix factorization of Y = AB' + E. Independence of A and B is assumed. 
Estimation of prior covariance CB and of variance sigma can be turned on and off. Estimates 
of cB are empirical. ARD property is imposed upon vec(A) through gamma prior CA and estimation of covariance. 
The prior model is following:
    
    p(Y|A,B) = N(Y|BA^T, sigma^2*I)
    p(vec(A)) = N(vec(A)|0, C_A), C_A = diag(c_a)
    p(B) = N(B|0, C_B), C_B = diag(c_b)
    p(C_A) = G(C_A| alpha0, beta0)

"""
function vbmf_sparse(Y::Array{Float64, 2}, params_in::vbmf_sparse_parameters, niter::Int; eps::Float64 = 1e-6, est_covs::Bool = false, 
    est_var::Bool = false, full_cov::Bool = false, logdir = "", desc = "")
    params = copy(params_in)
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
        updateB!(Y, params)
        updateA!(Y, params, full_cov = full_cov)
        
        updateCA!(params)
        
        if est_covs
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
    print("Factorization finished after ", i, " iterations, eps = ", d, "\n")

    # save inputs and outputs
    if log
        println("Saving outputs and inputs under ", logdir, "/")
        save_log(logVar, Y, params_in, priors, logdir, desc = desc)
    end

    return params
end

