include("util.jl")

"""
   vbmf_parameters

Compound type for vbmf computation. Contains the following fields:\n 
    
    L::Int - number of rows of the original matrix 
    M::Int - number of columns of the original matrix
    H::Int - internal product dimension
    AHat::Array{Float64, 2} - mean value of A, size (M, H)
    BHat::Array{Float64, 2} - mean value of B, size (L, H)
    SigmaAHat::Array{Float64, 2} - covariance of A, size (H, H)
    SigmaBHat::Array{Float64, 2} - covariance of B, size (H, H)
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
    AHat::Array{Float64, 2}
    BHat::Array{Float64, 2}
    SigmaAHat::Array{Float64, 2}
    SigmaBHat::Array{Float64, 2}
    CA::Array{Float64, 2}
    CB::Array{Float64, 2}
    invCA::Array{Float64, 2}
    invCB::Array{Float64, 2}
    sigma2::Float64
    YHat::Array{Float64, 2}
    
    vbmf_parameters() = new()
end

"""
    vbmf_init(L::Int, M::Int, H::Int, ca::Float64, cb::Float64, sigma::Float64)

Returns an initialized structure of type vbmf_parameters.
"""
function vbmf_init(L::Int, M::Int, H::Int, ca::Float64, cb::Float64, sigma2::Float64)
    params = vbmf_parameters()

    params.L = L
    params.M = M
    params.H = H
    params.AHat = randn(M, H)
    params.BHat = randn(L, H)
    params.SigmaAHat = zeros(H, H)
    params.SigmaBHat = zeros(H, H)
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
    params.SigmaAHat = params.sigma2*inv(params.BHat'*params.BHat + 
        params.L*params.SigmaBHat + params.sigma2*params.invCA)
    params.AHat =  Y'*params.BHat*params.SigmaAHat/params.sigma2
end

"""
    updateB!(Y::Array{Float64,2}, params::vbmf_parameters)

Updates mean and covariance of the B matrix.
"""
function updateB!(Y::Array{Float64,2}, params::vbmf_parameters)
    params.SigmaBHat = params.sigma2*inv(params.AHat'*params.AHat + 
        params.M*params.SigmaAHat + params.sigma2*params.invCB)
    params.BHat =  Y*params.AHat*params.SigmaBHat/params.sigma2
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
        params.CA[h,h] = norm2(params.AHat[:,h])/params.M + params.SigmaAHat[h,h]
    end
    params.invCA = inv(params.CA)
end

"""
    updateCB!(params::vbmf_parameters)

Updates the estimate of CB.
"""
function updateCB!(params::vbmf_parameters)
    for h in 1:params.H
        params.CB[h,h] = norm2(params.BHat[:,h])/params.L + params.SigmaBHat[h,h]
    end
    params.invCB = inv(params.CB)
end

"""
    updateSigma2!(Y::Array{Float64,2}, params::vbmf_parameters)

Updates estimate of sigma^2, the measurement noise.
"""
function updateSigma2!(Y::Array{Float64,2}, params::vbmf_parameters)
    params.sigma2 = (norm2(Y) - trace(2*Y'*params.BHat*params.AHat') + 
        trace((params.AHat'*params.AHat + params.M*params.SigmaAHat)*
            (params.BHat'*params.BHat + params.L*params.SigmaBHat)))/(params.L*params.M)
end

"""
    vbmf(Y::Array{Float64, 2}, params_in::vbmf_parameters, niter::Int, est_covs = false, est_var = false)

Computes variational bayes matrix factorization of Y = AB' + E. Independence of A and B is assumed. 
Estimation of prior covariance cA and cB and of variance sigma can be turned on and off.
The prior model is following:
    
    p(Y|A,B) = N(Y|BA^T, sigma^2*I)
    p(A) = N(A|0, C_A), C_A = diag(c_a)
    p(B) = N(B|0, C_B), C_B = diag(c_b)

"""
function vbmf(Y::Array{Float64, 2}, params_in::vbmf_parameters, niter::Int, est_covs::Bool = false, est_var::Bool = false)
    params = copy(params_in)

    for i in 1:niter
        updateA!(Y, params)
        updateB!(Y, params)

        if est_covs
            updateCA!(params)
            updateCB!(params)
        end

        if est_var
            updateSigma2!(Y, params)
        end
    end    

    return params
end