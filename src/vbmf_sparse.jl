"""
   vbmf_sparse_parameters

Compound type for vbmf_sparse computation. Contains the following fields:\n 
    
    L::Int - number of rows of the original matrix 
    M::Int - number of columns of the original matrix
    H::Int - internal product dimension
    H1::Int - number of columns of B that should belong to a contaminated class
    labels::Array{Int64, 1} - which columns of Y are labeled as non-contaminated

    AHat::Array{Float64, 2} - mean value of A, size (M, H)
    ATVecHat::Array{Float64, 1} - mean value of vec(A^T), size (MH,1)
    SigmaATVec::Array{Float64, 2} - covariance of vec(A^T), size (MH, MH)
    invSigmaATVec::Array{Float64, 2} - inverse of covariance of vec(A^T), size (MH, MH)
    
    BHat::Array{Float64, 2} - mean value of B, size (L, H)
    SigmaB::Array{Float64, 2} - covariance of B matrix, size (H, H)

    CA::Array{Float64, 1} - diagonal of the prior covariance of vec(A^T), size (MH)
    alpha0::Float64 - shape of CA gamma prior
    beta0::Float64 - scale of CA gamma prior
    alpha::Float64 - shape of CA gamma posterior
    beta::Array{Float64,1} - scale of CA gamma posterior, size (MH, 1)

    CB::Array{Float64, 1} - diagonal of prior covariance of B, size (H)
    gamma0::Float64 - shape of CB gamma prior
    delta0::Float64 - scale of CB gamma prior
    gamma::Float64 - shape of CB posterior
    delta::Array{Float64,1} - scale of CB posterior, size (H)

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
    ATVecHat::Array{Float64, 1}
    SigmaATVec::Array{Float64, 2}
    invSigmaATVec::Array{Float64, 2}
    
    BHat::Array{Float64, 2}
    SigmaB::Array{Float64, 2}

    CA::Array{Float64, 1}
    alpha0::Float64
    beta0::Float64
    alpha::Float64
    beta::Array{Float64,1}

    CB::Array{Float64, 1}
    gamma0::Float64
    delta0::Float64
    gamma::Float64
    delta::Array{Float64,1}

    sigma2::Float64
    YHat::Array{Float64, 2}
    
    vbmf_sparse_parameters() = new()
end

"""
    vbmf_sparse_init(L::Int, M::Int, H::Int; ca::Float64 = 1.0, alpha0::Float64 = 1e-10,
    beta0::Float64 = 1e-10, cb::Float64 = 1.0, gamma0::Float64 = 1e-10, delta0::Float64 = 1e-10,
    sigma2::Float64 = 1.0, H1::Int = 0, labels::Array{Int64,1} = Array{Int64,1}())

Returns an initialized structure of type vbmf_sparse_parameters.
"""
function vbmf_sparse_init(L::Int, M::Int, H::Int; ca::Float64 = 1.0, alpha0::Float64 = 1e-10,
 beta0::Float64 = 1e-10, cb::Float64 = 1.0, gamma0::Float64 = 1e-10, delta0::Float64 = 1e-10,
 sigma2::Float64 = 1.0, H1::Int = 0, labels::Array{Int64,1} = Array{Int64,1}())
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
    params.ATVecHat = reshape(params.AHat', M*H)
    params.SigmaATVec = eye(M*H, M*H)
    params.invSigmaATVec = eye(M*H, M*H)

    params.BHat = randn(L, H)
    params.SigmaB = zeros(H, H)

    params.CA = ca*ones(M*H)
    params.alpha0 = alpha0
    params.beta0 = beta0
    params.alpha = alpha0
    params.beta = beta0*ones(M*H)

    params.CB = cb*ones(H)  
    params.gamma0 = gamma0
    params.delta0 = delta0
    params.gamma = gamma0
    params.delta = delta0*ones(H)

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
    updateA!(Y::Array{Float64,2}, params::vbmf_sparse_parameters; full_cov::Bool)

Updates mean and covariance of vec(A^T) and also of the A matrix. If full_cov is true, 
then inverse of full covariance matrix is computed, otherwise just the diagonal is estimated.
"""
function updateA!(Y::Array{Float64,2}, params::vbmf_sparse_parameters; full_cov::Bool = false)
    params.invSigmaATVec = 1/params.sigma2*kron(eye(params.M), (params.BHat'*params.BHat + params.L*params.SigmaB)) + diagm(params.CA)
    if full_cov
        params.SigmaATVec = inv(params.invSigmaATVec)
    else
        params.SigmaATVec = diagm(ones(params.M*params.H)./diag(params.invSigmaATVec))
    end
    params.ATVecHat = 1/params.sigma2*params.SigmaATVec*reshape(params.BHat'*Y, params.H*params.M)
    # now set zeroes to where the columns of Y labeled as non-infected are
    # in A, these are rows
    params.AHat = reshape(params.ATVecHat, params.H, params.M)'
    params.AHat[params.labels, end-params.H1+1:end] = 0.0
    params.ATVecHat = reshape(params.AHat', params.M*params.H)
end

"""
    updateB!(Y::Array{Float64,2}, params::vbmf_sparse_parameters)

Updates mean and covariance of the B matrix.
"""
function updateB!(Y::Array{Float64,2}, params::vbmf_sparse_parameters)
    
    # first, compute the inverse of the covariance
    # and add all the diagonal submatrices from covariance of A
    params.SigmaB = diagm(params.CB) + 1/params.sigma2*params.AHat'*params.AHat
    for m in 1:params.M
        params.SigmaB += 1/params.sigma2*params.SigmaATVec[(m-1)*params.H+1:m*params.H, (m-1)*params.H+1:m*params.H] 
    end
    #invert it
    params.SigmaB = inv(params.SigmaB)
    params.BHat = 1/params.sigma2*Y*params.AHat*params.SigmaB
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
    params.beta = params.beta0*ones(params.M*params.H) + 
    1/2*(params.ATVecHat.*params.ATVecHat + diag(params.SigmaATVec))
    params.CA = params.alpha*ones(params.M*params.H)./params.beta
end

"""
    updateCB!(params::vbmf_sparse_parameters)

Updates the estimate of CB.
"""
function updateCB!(params::vbmf_sparse_parameters)
    params.gamma = params.gamma0 + params.L/2
    for h in 1:params.H
        params.delta[h] = params.delta0 + 1/2*(params.BHat[:,h]'*params.BHat[:,h])[1] + 1/2*params.SigmaB[h,h]
        params.CB[h] = params.gamma/params.delta[h]
    end
end

"""
    updateSigma2!(Y::Array{Float64,2}, params::vbmf_sparse_parameters)

Updates estimate of sigma^2, the measurement noise.
"""
function updateSigma2!(Y::Array{Float64,2}, params::vbmf_sparse_parameters)
    SigmaA = zeros(params.H, params.H)
    for m in 1:params.M
        SigmaA += params.SigmaATVec[(m-1)*params.H+1:m*params.H, (m-1)*params.H+1:m*params.H] 
    end

    params.sigma2 = (norm2(Y) - trace(2*Y'*params.BHat*params.AHat') + 
        trace((params.AHat'*params.AHat + SigmaA)*
            (params.BHat'*params.BHat + params.L*params.SigmaB)))/(params.L*params.M)
end

"""
    vbmf(Y::Array{Float64, 2}, params_in::vbmf_sparse_parameters, niter::Int, eps::Float64 = 1e-6, 
    est_var = false, full_cov::Bool = false, logdir = "", desc = "")

Computes variational bayes matrix factorization of Y = AB' + E. Independence of A and B is assumed. 
Estimation of variance sigma2 can be turned on and off. ARD property is imposed upon columns of B and also
upon vec(A) through gamma prior CB and CA and estimation of covariance. 
The prior model is following:
    
    p(Y|A,B) = N(Y|BA^T, sigma^2*I)
    p(vec(A^T)) = N(vec(A^T)|0, invCA), CA = diag(c_a)
    p(B_h) = MN(B|0, I, invCB), CB = diag(c1, ..., cH)
    p(CA) = G(C_A| alpha0, beta0)
    p(CB) = G(C_B| gamma0, delta0)

"""
function vbmf_sparse(Y::Array{Float64, 2}, params_in::vbmf_sparse_parameters, niter::Int; eps::Float64 = 1e-6,
    est_var::Bool = false, full_cov::Bool = false, logdir = "", desc = "", verb = false)
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
    old = copy(getfield(params, convergence_var))
    d = eps + 1.0 # delta
    i = 1

    # run the loop for a given number of iterations
    while ((i <= niter) && (d > eps))
        updateA!(Y, params, full_cov = full_cov)
        updateB!(Y, params)
        
        updateCA!(params)
        updateCB!(params)

        if est_var
            updateSigma2!(Y, params)
        end

        if log
            update_log!(logVar, params)
        end

        # check convergence
        d = delta(getfield(params, convergence_var), old)
        old = copy(getfield(params, convergence_var))
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
        save_log(logVar, Y, params_in, priors, logdir, desc = desc)
    end

    return params
end

