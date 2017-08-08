#__precompile__()

module VBMatrixFactorization

import Base.copy
using JLD 

export load_log, extract_params
export vbmf_parameters, vbmf, vbmf_init
export vbmf_sparse_parameters, vbmf_sparse, vbmf_sparse_init

include("util.jl")
include("data_manip.jl")
include("vbmf.jl")
include("vbmf_sparse.jl")

end # module
