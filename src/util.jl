const ln2pi = log(2*pi);

"""
    norm2(x::Array{Float64,1})

Computes ||x||^2 more precisely than norm(x)^2.
"""
function norm2(x::Array{Float64,1})
    return sum(x.^2)
end

"""
    norm2(X::Array{Float64,2})

Computes the square of the Frobenius matrix norm ||X||^2_FRO more precisely than vecnorm(A,2)^2.
"""
function norm2(X::Array{Float64,2})
    return sum(X.^2)
end


"""
   delta(new, old) 

Computes delta for convergence control for a given variable.
"""
function delta(new, old)
    return norm(old - new)/norm(old)
end

"""
    scaleY(Y::Array{Float64,2})

Scales down a 2 dimensional array so it has approx. standard normal distribution.
"""
function scaleY(Y::Array{Float64,2})
    L, M = size(Y)
    mu = mean(Y,2);
    sigma = var(Y,2);

    # if there are NaN, then sigma is zero for a given row -> 
    # the scaled down row is also zero
    # but we treat this more economically by setting the denominator for a given row to one
    # also, deal with numerical zeroes
    den = sigma
    den[abs(den) .<= 1e-15] = 1.0
    den[den .== 0.0] = 1.0
    den = repmat(sqrt(den),1, M)
    nom = Y - repmat(mu, 1, M)
    nom[abs(nom) .<= 1e-8] = 0.0
    Y = nom./den
    return Y
end

"""
    scaleY(Y::Array{Float32,2})

Scales down a 2 dimensional array so it has approx. standard normal distribution.
"""
function scaleY(Y::Array{Float32,2})
    Y64 = convert(Array{Float64,2}, Y)
    Y64 = scaleY(Y64)
    return convert(Array{Float32,2}, Y64)
end

"""
    traceXTY(X, Y)

Computes tr(X^TY) more effectively than trace(X'*Y). 
"""
function traceXTY(X::Array{Float64,2}, Y::Array{Float64,2})
    return(sum(X.*Y))
end

"""
    normalEntropy(Sigma)

Computes the entropy of a multivariate normal distribution with covariance Sigma.
"""
function normalEntropy(Sigma::Array{Float64,2})
    m, n = size(Sigma)
    if n != m
        error("Sigma must be square!")
    end

    return m/2 + m/2*ln2pi + 1/2*log(det(Sigma))
end

"""
    normalEntropy(diagSigma)

Computes the entropy of a multivariate normal distribution with 
diagonal of covariance diagSigma.
"""
function normalEntropy(diagSigma::Array{Float64,1})
    n = size(diagSigma)[1]

    return n/2 + n/2*ln2pi + 1/2*sum(log(diagSigma))
end

"""
    gammaEntropy(a, b)

Computes the entropy of a gamma distribution with shape a and rate b.
"""
function gammaEntropy(a::Float64, b::Float64)
    return a + log(b) + lgamma(a) + (1-a)*digamma(a)
end

"""
    gammaELn(a, b)

Computes E[ln(x)] where x follows gamma distribution with shape a and rate b.
"""
function gammaELn(a::Float64, b::Float64)
    return digamma(a) - log(b)
end

