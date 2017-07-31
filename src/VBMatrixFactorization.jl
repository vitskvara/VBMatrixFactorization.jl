#__precompile__()

module VBMatrixFactorization

import Base.copy
using JLD 

export loadLog
export vbmf_parameters, vbmf, vbmf_init

include("util.jl")
include("data_manip.jl")
include("vbmf.jl")

end # module
