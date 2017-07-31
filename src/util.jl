
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