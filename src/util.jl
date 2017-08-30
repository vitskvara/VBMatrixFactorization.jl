
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
