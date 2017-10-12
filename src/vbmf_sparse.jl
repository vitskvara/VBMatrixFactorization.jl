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
    diagSigmaATVec::Array{Float64,1} - diagonal of covariance of vec(A^T), size (MH)
    invSigmaATVec::Array{Float64, 2} - inverse of covariance of vec(A^T), size (MH, MH)
    SigmaA::Array{Float64,2} - covariance of A matrix, size (H, H)

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

    sigmaHat::Float64 - variance of data - homoscedastic case
    eta0::Float64 - shape of sigma gamma prior
    zeta0::Float64 - scale of sigma gamma prior
    eta::Float64 - shape of homoscedastic sigma posterior
    zeta::Float64 - scale of homoscedastic sigma posterior 
    sigmaVecHat::Array{Float64,1} - variance of data (rows) - heteroscedastic case, size (L)
    etaVec::Array{Float64,1} - shape of heteroscedastic sigma posterior, size (L)
    zetaVec::Array{Float64,1} - scale of heteroscedastic sigma posterior, size (L)

    YHat::Array{Float64, 2} - estimate of Y, size (L, M)
    trYTY::Float64 - trace(Y^T*Y), saved so that it does not have to be recomputed
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
    diagSigmaATVec::Array{Float64,1}
    invSigmaATVec::Array{Float64, 2}
    SigmaA::Array{Float64,2}
    
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

    sigmaHat::Float64
    eta0::Float64
    zeta0::Float64
    eta::Float64
    zeta::Float64
    sigmaVecHat::Array{Float64,1}
    etaVec::Array{Float64,1}
    zetaVec::Array{Float64,1}

    YHat::Array{Float64, 2}
    trYTY::Float64
    
    vbmf_sparse_parameters() = new()
end

"""
    vbmf_sparse_init(Y::Array{Float64,2}, H::Int; ca::Float64 = 1.0, 
    alpha0::Float64 = 1e-10, beta0::Float64 = 1e-10, cb::Float64 = 1.0, 
    gamma0::Float64 = 1e-10, delta0::Float64 = 1e-10,
    sigma::Float64 = 1.0, eta0::Float64 = 1e-10, zeta0::Float64 = 1e-10,
    H1::Int = 0, labels::Array{Int64,1} = Array{Int64,1}())

Returns an initialized structure of type vbmf_sparse_parameters.
"""
function vbmf_sparse_init(Y::Array{Float64,2}, H::Int; ca::Float64 = 1.0, 
    alpha0::Float64 = 1e-10, beta0::Float64 = 1e-10, cb::Float64 = 1.0, 
    gamma0::Float64 = 1e-10, delta0::Float64 = 1e-10,
    sigma::Float64 = 1.0, eta0::Float64 = 1e-10, zeta0::Float64 = 1e-10,
    H1::Int = 0, labels::Array{Int64,1} = Array{Int64,1}())
    params = vbmf_sparse_parameters()
    L, M = size(Y)

    params.L, params.M = L, M 
    params.H = H
    params.H1 = H1
    params.labels = labels

    params.AHat = randn(M, H)
    # now set zeroes to where the columns of Y labeled as non-infected are
    # in A, these are rows
    params.AHat[labels, end-H1+1:end] = 0.0
    params.ATVecHat = reshape(params.AHat', M*H)
    params.SigmaATVec = eye(M*H, M*H)
    params.diagSigmaATVec = ones(M*H)
    params.invSigmaATVec = eye(M*H, M*H)
    params.SigmaA = zeros(H, H)

    params.BHat = randn(L, H)
    params.SigmaB = zeros(H, H)

    params.CA = ca*ones(M*H)
    params.alpha0 = alpha0
    params.beta0 = beta0
    params.alpha = params.alpha0 + 0.5
    params.beta = beta0*ones(M*H)

    params.CB = cb*ones(H)  
    params.gamma0 = gamma0
    params.delta0 = delta0
    params.gamma = gamma0 + L/2
    params.delta = delta0*ones(H)

    params.sigmaHat = sigma
    params.eta0 = eta0
    params.zeta0 = zeta0
    params.eta = eta0 + L*M/2
    params.zeta = zeta0
    params.sigmaVecHat = sigma*ones(L)
    params.etaVec = (eta0 + M/2)*ones(L)
    params.zetaVec = zeta0*ones(L)

    params.YHat = params.BHat*params.AHat'
    params.trYTY = traceXTY(Y, Y)
    
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
    updateA!(Y::Array{Float64,2}, params::vbmf_sparse_parameters; full_cov::Bool, diag_var::Bool = false)

Updates mean and covariance of vec(A^T) and also of the A matrix. If full_cov is true, 
then inverse of full covariance matrix is computed, otherwise just the diagonal is estimated.
"""
function updateA!(Y::Array{Float64,2}, params::vbmf_sparse_parameters; full_cov::Bool = false, diag_var::Bool = false)
    # either just the diagonal or the full covariance
    if full_cov
        # compute the inverse of covariance
        if diag_var
            params.invSigmaATVec = kron(eye(params.M), 
                (params.BHat'*diagm(params.sigmaVecHat)*params.BHat + params.L*mean(params.sigmaVecHat)*params.SigmaB)) + diagm(params.CA)
        else
            params.invSigmaATVec = params.sigmaHat*kron(eye(params.M), 
                (params.BHat'*params.BHat + params.L*params.SigmaB)) + diagm(params.CA)
        end

        params.SigmaATVec = inv(params.invSigmaATVec)
        params.diagSigmaATVec = diag(params.SigmaATVec)

        # now the mean
        if diag_var
            params.ATVecHat = params.SigmaATVec*reshape(params.BHat'*diagm(params.sigmaVecHat)*Y, params.H*params.M)
        else
            params.ATVecHat = params.sigmaHat*params.SigmaATVec*reshape(params.BHat'*Y, params.H*params.M)
        end

        # now compute the covariance of AHat
        params.SigmaA = zeros(params.H, params.H)
        for m in 1:params.M
            params.SigmaA += view(params.SigmaATVec, (m-1)*params.H+1:m*params.H, (m-1)*params.H+1:m*params.H)
        end

    else # this updates just the diagonal of the covariance of A
        # compute the inverse of covariance
        # only use for large problems, for small problems the full covariance is more precise
        if diag_var
            for h in 1:params.H
                # instead of the full matrix multiplication as above, we compute just 
                # the diagonal
                params.diagSigmaATVec[h] = norm2(params.BHat[:,h].*params.sigmaVecHat) + params.L*mean(params.sigmaVecHat)*params.SigmaB[h,h]
            end
        else         # this is computed in case the variance of data is homoscedastic
            for h in 1:params.H
                # instead of the full matrix multiplication as above, we compute just 
                # the diagonal
                params.diagSigmaATVec[h] = params.sigmaHat*norm2(params.BHat[:,h]) + params.L*params.SigmaB[h,h]
            end
        end
        # we filled just the first H elements, so we just copy it to fill the rest
        params.diagSigmaATVec[params.H+1:end] = repeat(params.diagSigmaATVec[1:params.H], inner = params.M-1)
        # also add the CA vector
        params.diagSigmaATVec += params.CA

        # finally, invert it
        params.diagSigmaATVec = 1./params.diagSigmaATVec

        # now the mean
        if diag_var
            params.ATVecHat = params.diagSigmaATVec.*reshape(params.BHat'*diagm(params.sigmaVecHat)*Y, params.H*params.M)
        else
            params.ATVecHat = params.sigmaHat*params.diagSigmaATVec.*reshape(params.BHat'*Y, params.H*params.M)
        end

        # now compute the covariance of AHat
        params.SigmaA = zeros(params.H, params.H)
        for m in 1:params.M
            params.SigmaA += diagm(view(params.diagSigmaATVec, (m-1)*params.H+1:m*params.H))
        end
    end

    # now set zeroes to where the columns of Y labeled as non-infected are
    # in A, these are rows
    params.AHat = reshape(params.ATVecHat, params.H, params.M)'
    params.AHat[params.labels, end-params.H1+1:end] = 0.0
    params.ATVecHat = reshape(params.AHat', params.M*params.H)
end

"""
    updateB!(Y::Array{Float64,2}, params::vbmf_sparse_parameters, diag_var::Bool = false)

Updates mean and covariance of the B matrix.
"""
function updateB!(Y::Array{Float64,2}, params::vbmf_sparse_parameters;  diag_var::Bool = false)
    # first, compute the inverse of the covariance
    if diag_var
        params.SigmaB = diagm(params.CB) + mean(params.sigmaVecHat)*(params.AHat'*params.AHat + params.SigmaA)
        #invert it
        params.SigmaB = inv(params.SigmaB)
        # now compute the mean
        params.BHat = diagm(params.sigmaVecHat)*Y*params.AHat*params.SigmaB
    else
        params.SigmaB = diagm(params.CB) + params.sigmaHat*(params.AHat'*params.AHat + params.SigmaA)
        #invert it
        params.SigmaB = inv(params.SigmaB)
        params.BHat = params.sigmaHat*Y*params.AHat*params.SigmaB
    end
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
    params.beta = params.beta0*ones(params.M*params.H) + 
    1/2*(params.ATVecHat.*params.ATVecHat + params.diagSigmaATVec)
    params.CA = params.alpha*ones(params.M*params.H)./params.beta
end

"""
    updateCB!(params::vbmf_sparse_parameters)

Updates the estimate of CB.
"""
function updateCB!(params::vbmf_sparse_parameters)
    for h in 1:params.H
        params.delta[h] = params.delta0 + 1/2*(params.BHat[:,h]'*params.BHat[:,h])[1] + 1/2*params.SigmaB[h,h]
        params.CB[h] = params.gamma/params.delta[h]
    end
end

"""
    updateSigma!(Y::Array{Float64,2}, params::vbmf_sparse_parameters, diag_var::Bool = false)

Updates estimate of the measurement variance.
"""
function updateSigma!(Y::Array{Float64,2}, params::vbmf_sparse_parameters; diag_var::Bool = false)
    if diag_var
        for l in 1:params.L
            params.zetaVec[l] = params.zeta0 + 1/2*norm2(Y[l,:]) - sum(Y[l,:].*(params.AHat*params.BHat[l,:])) +
                1/2*traceXTY(params.AHat'*params.AHat + params.SigmaA, 
                    params.BHat[l,:]*params.BHat[l,:]' + params.SigmaB)
            
            params.sigmaVecHat[l] = params.etaVec[l]/params.zetaVec[l]
        end
    else
        params.zeta = params.zeta0 + 1/2*params.trYTY - traceXTY(params.BHat, Y*params.AHat) + 
            1/2*traceXTY(params.AHat'*params.AHat + params.SigmaA, 
                params.BHat'*params.BHat + params.L*params.SigmaB)

        params.sigmaHat = params.eta/params.zeta
    end
end

"""
    vbmf_sparse!(Y::Array{Float64, 2}, params::vbmf_sparse_parameters, niter::Int, eps::Float64 = 1e-6, 
    est_var = false, full_cov::Bool = false, logdir = "", desc = "")

Computes variational bayes matrix factorization of Y = AB' + E. Independence of A and B is assumed. 
Estimate of variance sigma can be either a scalar or a vector - for row variance estimation. 
ARD property is imposed upon columns of B and also
upon vec(A) through gamma prior CB and CA and estimation of covariance. 
The prior model is following:
    
    p(Y|A,B) = N(Y|BA^T, 1/sigma*I) or p(Y|A,B) = N(Y|BA^T, inv(diag(sigma))) 
    p(vec(A^T)) = N(vec(A^T)|0, invCA), CA = diag(c_a)
    p(B_h) = MN(B|0, I, invCB), CB = diag(c1, ..., cH)
    p(CA) = G(C_A| alpha0, beta0)
    p(CB) = G(C_B| gamma0, delta0)

The params argument with initialized data is modified and contains the resulting estimates after the 
algorithm stops.
"""
function vbmf_sparse!(Y::Array{Float64, 2}, params::vbmf_sparse_parameters, niter::Int; eps::Float64 = 1e-6,
    diag_var::Bool = false, full_cov::Bool = false, logdir = "", desc = "", verb = false)
    priors = Dict()

    # create the log dictionary
    log = false
    if logdir !=""
        log = true
        logVar = create_log(params)
    end

    # choice of convergence control variable
    convergence_var = :BHat
    #convergence_var = :dY

    if convergence_var == :dY
        old = Y - params.YHat
    else
        old = copy(getfield(params, convergence_var))
    end
    d = eps + 1.0 # delta
    i = 1

    # run the loop for a given number of iterations
    while ((i <= niter) && (d > eps))
        updateA!(Y, params, full_cov = full_cov, diag_var = diag_var)
        updateB!(Y, params, diag_var = diag_var)
        
        updateCA!(params)
        updateCB!(params)

        updateSigma!(Y, params, diag_var = diag_var)

        if log
            update_log!(logVar, params)
        end

        # check convergence
        if convergence_var == :dY
            updateYHat!(params)
            d = delta(Y - params.YHat, old)
            old = Y - params.YHat
        else
            d = delta(getfield(params, convergence_var), old)
            old = copy(getfield(params, convergence_var))
        end
        i += 1
        #println(lowerBound(Y, params))
        #println(lowerBound2(Y, params))
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

    return d
end

"""
    vbmf_sparse(Y::Array{Float64, 2}, params_in::vbmf_sparse_parameters, niter::Int, eps::Float64 = 1e-6, 
    est_var = false, full_cov::Bool = false, logdir = "", desc = "")

Calls vbmf_sparse!() but copies the params_in argument so that it is not modified and can be reused.
"""
function vbmf_sparse(Y::Array{Float64, 2}, params_in::vbmf_sparse_parameters, niter::Int; eps::Float64 = 1e-6,
    diag_var::Bool = false, full_cov::Bool = false, logdir = "", desc = "", verb = false)
    # make a copy of input params
    params = copy(params_in)

    # run the algorithm
    d = vbmf_sparse!(Y, params, niter, eps = eps, diag_var = diag_var, full_cov = full_cov, 
        logdir = logdir, desc = desc, verb = verb)

    return params, d
end

"""
    lowerBound(Y::Array{Float64,2}, params::vbmf_sparse_parameters)

Compute the lower bound for logarithm of data distribution.
"""
function lowerBound(Y::Array{Float64,2}, params::vbmf_sparse_parameters)
    L = 0.0
    # E[lnp(Y|params)]
    L += - params.L*params.M/2*ln2pi + params.L*params.M/2*gammaELn(params.eta, params.zeta) 
    L += - params.sigmaHat/2*(params.trYTY - 2*traceXTY(params.BHat, Y*params.AHat)  
         + traceXTY(params.AHat'*params.AHat + params.SigmaA, params.BHat'*params.BHat + params.L*params.SigmaB))
    # E[lnp(vec(A'))]
    L += - params.M*params.H/2*ln2pi + 1/2*sum(map(gammaELn, params.alpha*ones(size(params.beta)), params.beta))
    L += - (1/2*params.CA'*(params.ATVecHat.^2 + params.diagSigmaATVec))[1]
    # E[lnp(B)]
    L += - params.L*params.H/2*ln2pi
    L += params.L/2*sum(map(gammaELn, params.gamma*ones(size(params.delta)), params.delta))
    L += - 1/2*traceXTY(diagm(params.CB), (params.BHat'*params.BHat + params.L*params.SigmaB))  
    # E[lnp(sigma)]
    L += params.eta0*log(params.zeta0) - lgamma(params.eta0) 
    L += (params.eta0 - 1)*gammaELn(params.eta, params.zeta) - params.zeta0*params.sigmaHat
    # E[lnp(CA)]
    L += params.H*params.M*(params.alpha0*log(params.beta0) - lgamma(params.alpha0))
    L += (params.alpha0 - 1)*sum(map(gammaELn, params.alpha*ones(size(params.beta)), params.beta))
    L += - params.beta0*sum(params.CA)
    # E[lnp(CB)]
    L += params.H*(params.gamma0*log(params.delta0) - lgamma(params.gamma0))
    L += (params.gamma0 - 1)*sum(map(gammaELn, params.gamma*ones(size(params.delta)), params.delta))
    L += - params.gamma0*sum(params.CB)

    # H(vec(A'))
    L += normalEntropy(params.diagSigmaATVec)
    # H(B)
    L += normalEntropy(kron(params.SigmaB,eye(params.L))) # this also causes -inf
    # H(sigma)
    L += gammaEntropy(params.eta, params.zeta)
    # H(CA)
    L += sum(map(gammaEntropy, params.alpha*ones(size(params.beta)), params.beta))
    # H(CB)
    L += sum(map(gammaEntropy, params.gamma*ones(size(params.delta)), params.delta))
    return L
end
