using Roots # for fzero fction

"""
   vbmf_dual_parameters

Compound type for vbmf_dual computation. Contains the following fields:\n 
    
    L::Int - number of rows of the original matrix 
    M::Int - number of columns of the original matrix
    MH::Int - columns*internal factorization dimension
    H::Int - internal product dimension, H = H0 + H1
    H0::Int - size of base of noncontaminated class
    H1::Int - size of base of contaminated class

    AHat::Array{Float64, 2} - mean value of A, size (M, H), A = [A0 A1]
    ATVecHat::Array{Float64, 1} - mean value of vec(A^T), size (MH,1)
    SigmaATVec::Array{Float64, 2} - covariance of vec(A^T), size (MH, MH)
    diagSigmaATVec::Array{Float64,1} - diagonal of covariance of vec(A^T), size (MH)
    invSigmaATVec::Array{Float64, 2} - inverse of covariance of vec(A^T), size (MH, MH)
    SigmaA::Array{Float64,2} - covariance of A matrix, size (H, H)
    A0Hat::Array{Float64, 2} - mean value of A0, size (M, H0)
    A1Hat::Array{Float64, 2} - mean value of A1, size (M, H1)

    BHat::Array{Float64, 2} - mean value of B, size (L, H)
    SigmaB::Array{Float64, 2} - covariance of B matrix, size (H, H)

    CA::Array{Float64, 1} - diagonal of the prior covariance of vec(A^T), size (MH)
    alpha::Array{Float64, 1} - shapes of CA gamma posterior, size (2,1)
    beta::Array{Float64,1} - scale of CA gamma posterior, size (MH, 1)
    CA0::Array{Float64, 1} - prior covariance of A0, size (MH0)
    alpha00::Float64 - shape of CA0 gamma prior
    beta00::Float64 - scale of CA0 gamma prior
    alpha0::Float64 - shape of CA0 gamma posterior
    beta0::::Array{Float64,1} - scale of CA0 gamma posterior, size (MH0, 1)    
    CA1::Array{Float64, 1} - prior covariance of A1, size (MH1)
    alpha01::Float64 - shape of CA1 gamma prior
    beta01::Float64 - scale of CA1 gamma prior
    alpha1::Float64 - shape of CA1 gamma posterior
    beta1::::Array{Float64,1} - scale of CA1 gamma posterior, size (MH1, 1)

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
type vbmf_dual_parameters
    L::Int
    M::Int
    MH::Int
    H::Int
    H0::Int
    H1::Int
    
    AHat::Array{Float64, 2}
    ATVecHat::Array{Float64, 1}
    SigmaATVec::Array{Float64, 2}
    diagSigmaATVec::Array{Float64,1}
    invSigmaATVec::Array{Float64, 2}
    SigmaA::Array{Float64,2}
    A0Hat::Array{Float64, 2}
    A1Hat::Array{Float64, 2}
    
    BHat::Array{Float64, 2}
    SigmaB::Array{Float64, 2}

    CA::Array{Float64, 1}
    alpha::Array{Float64, 1}
    beta::Array{Float64,1}
    CA0::Array{Float64, 1}
    alpha00::Float64
    beta00::Float64
    alpha0::Float64
    beta0::Array{Float64,1}
    CA1::Array{Float64, 1}
    alpha01::Float64
    beta01::Float64
    alpha1::Float64
    beta1::Array{Float64,1}

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
    
    vbmf_dual_parameters() = new()
end

"""
    vbmf_dual_init(Y::Array{Float64,2}, H::Int, H0::Int; ca::Float64 = 1.0, 
    alpha0::Float64 = 1e-10, beta0::Float64 = 1e-10, cb::Float64 = 1.0, 
    gamma0::Float64 = 1e-10, delta0::Float64 = 1e-10,
    sigma::Float64 = 1.0, eta0::Float64 = 1e-10, zeta0::Float64 = 1e-10)

Returns an initialized structure of type vbmf_dual_parameters.
"""
function vbmf_dual_init(Y::Array{Float64,2}, H::Int, H0::Int; ca::Float64 = 1.0, 
    alpha0::Float64 = 1e-10, beta0::Float64 = 1e-10, cb::Float64 = 1.0, 
    gamma0::Float64 = 1e-10, delta0::Float64 = 1e-10,
    sigma::Float64 = 1.0, eta0::Float64 = 1e-10, zeta0::Float64 = 1e-10)
    if H < H0
        error("H must be at least H0!")
    end

    params = vbmf_dual_parameters()
    L, M = size(Y)

    params.L, params.M = L, M 
    params.H = H
    params.MH = M*H
    params.H0 = H0
    H1 = H - H0 
    params.H1 = H1

    params.AHat = randn(M, H)
    params.ATVecHat = reshape(params.AHat', M*H)
    params.SigmaATVec = eye(M*H, M*H)
    params.diagSigmaATVec = ones(M*H)
    params.invSigmaATVec = eye(M*H, M*H)
    params.SigmaA = zeros(H, H)
    params.A0Hat = params.AHat[:,1:params.H0]
    params.A1Hat = params.AHat[:,(params.H0+1):end]

    params.BHat = randn(L, H)
    params.SigmaB = zeros(H, H)

    params.CA0 = ca*ones(M*H0)
    params.CA1 = ca*ones(M*H1)
    params.CA = Array{Float64, 1}() 
    for m in 1:params.M
        params.CA = cat(1, params.CA, params.CA0[((m-1)*params.H0+1):m*params.H0])
        params.CA = cat(1, params.CA, params.CA1[((m-1)*params.H1+1):m*params.H1])
    end    
    params.alpha00 = alpha0
    params.alpha01 = alpha0
    params.beta00 = beta0
    params.beta01 = beta0
    params.alpha0 = alpha0 + 0.5
    params.beta0 = beta0*ones(M*H0)
    params.alpha1 = alpha0 + 0.5
    params.beta1 = beta0*ones(M*H1)
    params.alpha = [params.alpha0, params.alpha1]
    params.beta = Array{Float64, 1}()
    for m in 1:params.M
        params.beta = cat(1, params.beta, params.beta0[((m-1)*params.H0+1):m*params.H0])
        params.beta = cat(1, params.beta, params.beta1[((m-1)*params.H1+1):m*params.H1])
    end    

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
   copy(params_in::vbmf_dual_parameters)

Copy function for vbmfa_dual_parameters. 
"""
function copy(params_in::vbmf_dual_parameters)
    params = vbmf_dual_parameters()
    
    for field in fieldnames(params_in)
        setfield!(params, field, copy(getfield(params_in, field)))
    end
    
    return params
end

"""
    updateA!(Y::Array{Float64,2}, params::vbmf_dual_parameters; full_cov::Bool, diag_var::Bool = false)

Updates mean and covariance of vec(A^T) and also of the A matrix. If full_cov is true, 
then inverse of full covariance matrix is computed, otherwise just the diagonal is estimated.
"""
function updateA!(Y::Array{Float64,2}, params::vbmf_dual_parameters; full_cov::Bool = false, diag_var::Bool = false)
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

    params.AHat = reshape(params.ATVecHat, params.H, params.M)'
    params.A0Hat = params.AHat[:,1:params.H0]
    params.A1Hat = params.AHat[:,(params.H0+1):end]
end

"""
    updateB!(Y::Array{Float64,2}, params::vbmf_dual_parameters, diag_var::Bool = false)

Updates mean and covariance of the B matrix.
"""
function updateB!(Y::Array{Float64,2}, params::vbmf_dual_parameters;  diag_var::Bool = false)
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
    updateYHat!(params::vbmf_dual_parameters)

Updates estimate of Y.
"""
function updateYHat!(params::vbmf_dual_parameters)
    params.YHat = params.BHat*params.AHat'
end

"""
    updateCA!(params::vbmf_dual_parameters)

Updates the estimate of CA.
"""
function updateCA!(params::vbmf_dual_parameters)
    # compute the updates
    params.alpha0 = params.alpha00 + 1/2
    params.alpha1 = params.alpha01 + 1/2

    diagSigmaA = reshape(params.diagSigmaATVec, params.H, params.M)
    diagSigmaA0Vec = diagSigmaA[1:params.H0,:]
    diagSigmaA1Vec = diagSigmaA[(params.H0+1):end,:]

    params.beta0 = params.beta00*ones(params.M*params.H0) 
    params.beta0 += 1/2*vec(params.A0Hat'.*params.A0Hat' + diagSigmaA0Vec)
    params.beta1 = params.beta01*ones(params.M*params.H1)
    params.beta1 += 1/2*vec(params.A1Hat'.*params.A1Hat' + diagSigmaA1Vec)

    params.CA0 = params.alpha0*ones(params.M*params.H0)./params.beta0
    params.CA1 = params.alpha1*ones(params.M*params.H1)./params.beta1

    # now just recombine the results back to the original arrays
    params.CA = Array{Float64, 1}() 
    for m in 1:params.M
        params.CA = cat(1, params.CA, params.CA0[((m-1)*params.H0+1):m*params.H0])
        params.CA = cat(1, params.CA, params.CA1[((m-1)*params.H1+1):m*params.H1])
    end    
    params.alpha = [params.alpha0, params.alpha1]
    params.beta = Array{Float64, 1}()
    for m in 1:params.M
        params.beta = cat(1, params.beta, params.beta0[((m-1)*params.H0+1):m*params.H0])
        params.beta = cat(1, params.beta, params.beta1[((m-1)*params.H1+1):m*params.H1])
    end   
end

"""
    updateCB!(params::vbmf_dual_parameters)

Updates the estimate of CB.
"""
function updateCB!(params::vbmf_dual_parameters)
    for h in 1:params.H
        params.delta[h] = params.delta0 + 1/2*(params.BHat[:,h]'*params.BHat[:,h])[1] + 1/2*params.SigmaB[h,h]
        params.CB[h] = params.gamma/params.delta[h]
    end
end

"""
    updateSigma!(Y::Array{Float64,2}, params::vbmf_dual_parameters, diag_var::Bool = false)

Updates estimate of the measurement variance.
"""
function updateSigma!(Y::Array{Float64,2}, params::vbmf_dual_parameters; diag_var::Bool = false)
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
    updateAlpha00!(params::vbmf_dual_parameters)

Estimate alpha00 by maximizing the lower bound.
"""
function updateAlpha00!(params::vbmf_dual_parameters)
    function fAlpha00(x)
        return params.M*params.H0*log(params.beta00) - params.M*params.H0*digamma(x) + sum(map(gammaELn, params.alpha0*ones(size(params.beta0)), params.beta0))
    end
    try
        root = fzero(fAlpha00, 1e-10, 1e10, ftol = 1e-5)
        params.alpha00 = root
    end
end

"""
    updateBeta00!(params::vbmf_dual_parameters)

Estimate beta00 by maximizing the lower bound.
"""
function updateBeta00!(params::vbmf_dual_parameters)
    params.beta00 = params.M*params.H0*params.alpha00/sum(params.CA0)
end

"""
    updateAlpha01!(params::vbmf_dual_parameters)

Estimate alpha01 by maximizing the lower bound.
"""
function updateAlpha01!(params::vbmf_dual_parameters)
    function fAlpha01(x)
        return params.M*params.H1*log(params.beta01) - params.M*params.H1*digamma(x) + sum(map(gammaELn, params.alpha1*ones(size(params.beta1)), params.beta1))
    end
    try
        root = fzero(fAlpha01, 1e-10, 1e10, ftol = 1e-5)
        params.alpha01 = root
    end
end

"""
    updateBeta01!(params::vbmf_dual_parameters)

Estimate beta01 by maximizing the lower bound.
"""
function updateBeta01!(params::vbmf_dual_parameters)
    params.beta01 = params.M*params.H1*params.alpha01/sum(params.CA1)
end

"""
    vbmf_dual!(Y::Array{Float64, 2}, params::vbmf_dual_parameters, niter::Int, eps::Float64 = 1e-6, 
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
function vbmf_dual!(Y::Array{Float64, 2}, params::vbmf_dual_parameters, niter::Int; eps::Float64 = 1e-6,
    diag_var::Bool = false, full_cov::Bool = false, logdir = "", desc = "", verb = false, est_priors = true,
    est_cb::Bool = true)
    priors = Dict()

    # create the log dictionary
    loging = false
    if logdir !=""
        loging = true
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
        if est_cb
            updateCB!(params)
        end

        updateSigma!(Y, params, diag_var = diag_var)

        if est_priors
            # estimate priors
            updateAlpha00!(params)
            updateAlpha01!(params)
            updateBeta00!(params)
            updateBeta01!(params)
        end
        
        if loging
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
    end    

    # finally, compute the estimate of Y
    updateYHat!(params)

    # convergence info
    if verb
        print("Factorization finished after ", i-1, " iterations, eps = ", d, "\n")
    end
    
    # save inputs and outputs
    if loging
        println("Saving outputs and inputs under ", logdir, "/")
        save_log(logVar, Y, priors, logdir, desc = desc)
    end

    return d
end

"""
    vbmf_dual(Y::Array{Float64, 2}, params_in::vbmf_dual_parameters, niter::Int, eps::Float64 = 1e-6, 
    est_var = false, full_cov::Bool = false, logdir = "", desc = "")

Calls vbmf_dual!() but copies the params_in argument so that it is not modified and can be reused.
"""
function vbmf_dual(Y::Array{Float64, 2}, params_in::vbmf_dual_parameters, niter::Int; eps::Float64 = 1e-6,
    diag_var::Bool = false, full_cov::Bool = false, logdir = "", desc = "", verb = false, est_priors = true, 
    est_cb::Bool = true)
    # make a copy of input params
    params = copy(params_in)

    # run the algorithm
    d = vbmf_dual!(Y, params, niter, eps = eps, diag_var = diag_var, full_cov = full_cov, 
        logdir = logdir, desc = desc, verb = verb, est_priors = est_priors, est_cb = est_cb)

    return params, d
end

"""
    lowerBound(Y::Array{Float64,2}, params::vbmf_dual_parameters)

Compute the lower bound for logarithm of data distribution. 
"""
function lowerBound(Y::Array{Float64,2}, params::vbmf_dual_parameters)
    L = 0.0
    # E[lnp(Y|params)]
    L += - params.L*params.M/2*ln2pi + params.L*params.M/2*gammaELn(params.eta, params.zeta) 
    L += - params.sigmaHat/2*(params.trYTY - 2*traceXTY(params.BHat, Y*params.AHat)  
         + traceXTY(params.AHat'*params.AHat + params.SigmaA, params.BHat'*params.BHat + params.L*params.SigmaB))
    # E[lnp(vec(A'))]
    L += - params.MH/2*ln2pi + 1/2*sum(map(gammaELn, params.alpha0*ones(size(params.beta0)), params.beta0))
    L += 1/2*sum(map(gammaELn, params.alpha1*ones(size(params.beta1)), params.beta1))
    L += - (1/2*params.CA'*(params.ATVecHat.^2 + params.diagSigmaATVec))[1]
    # E[lnp(B)]
    L += - params.L*params.H/2*ln2pi
    L += params.L/2*sum(map(gammaELn, params.gamma*ones(size(params.delta)), params.delta))
    L += - 1/2*traceXTY(diagm(params.CB), (params.BHat'*params.BHat + params.L*params.SigmaB))  
    # E[lnp(sigma)]
    L += params.eta0*log(params.zeta0) - lgamma(params.eta0) 
    L += (params.eta0 - 1)*gammaELn(params.eta, params.zeta) - params.zeta0*params.sigmaHat
    # E[lnp(CA0)]
    L += params.M*params.H0*(params.alpha00*log(params.beta00) - lgamma(params.alpha00))
    L += (params.alpha00 - 1)*sum(map(gammaELn, params.alpha0*ones(size(params.beta0)), params.beta0))
    L += - params.beta00*sum(params.CA0)
    # E[lnp(CA1)]
    L += params.M*params.H1*(params.alpha01*log(params.beta01) - lgamma(params.alpha01))
    L += (params.alpha01 - 1)*sum(map(gammaELn, params.alpha1*ones(size(params.beta1)), params.beta1))
    L += - params.beta01*sum(params.CA1)
    # E[lnp(CB)]
    L += params.H*(params.gamma0*log(params.delta0) - lgamma(params.gamma0))
    L += (params.gamma0 - 1)*sum(map(gammaELn, params.gamma*ones(size(params.delta)), params.delta))
    L += - params.gamma0*sum(params.CB)

    # H(vec(A'))
    L += normalEntropy(params.diagSigmaATVec)
    # H(B)
    L += normalEntropy(kron(params.SigmaB,eye(params.L)))
    # H(sigma)
    L += gammaEntropy(params.eta, params.zeta)
    # H(CA0)
    L += sum(map(gammaEntropy, params.alpha0*ones(size(params.beta0)), params.beta0))
    # H(CA1)
    L += sum(map(gammaEntropy, params.alpha1*ones(size(params.beta1)), params.beta1))
    # H(CB)
    L += sum(map(gammaEntropy, params.gamma*ones(size(params.delta)), params.delta))
    return L
end

"""
    lowerBoundTrimmed(Y::Array{Float64,2}, params_in::vbmf_dual_parameters, trim = 1e-1)

AHat with abs(AHat[i,j]) < trim and the corresponding elements of CA are not involved in the computation.
"""
function lowerBoundTrimmed(Y::Array{Float64,2}, params_in::vbmf_dual_parameters, trim = 1e-1)
    params = copy(params_in)

    trim_inds = (abs(params.ATVecHat) .> trim)
    params.ATVecHat = params.ATVecHat[trim_inds]
    params.MH = size(params.ATVecHat)[1]
    params.beta = params.beta[trim_inds]
    params.CA = params.CA[trim_inds]
    params.diagSigmaATVec = params.diagSigmaATVec[trim_inds]

    return lowerBound(Y, params)
end