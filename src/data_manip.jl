"
    createLog(params)

Initializes the log dictionary for iteration progress data storage.
"
function create_log(params)
    logVar = Dict()

    # iterate through the params type and create apropriate fields
    for name in fieldnames(params)
        strname = string(name) # otherwise we would have symbols and would have
        # to call dict[:name] instead of dict["name"]

        # if some of the fields in params are not array, create an array from 
        # them so that future new values can be concatenated
        if supertype(typeof(getfield(params, name))) == AbstractFloat || 
            supertype(typeof(getfield(params, name))) == Signed
            logVar[strname] = [getfield(params, name)]
        else
            logVar[strname] = copy(getfield(params, name))
        end
    end

    return logVar
end

"
    updateLog!(logVar::Dict{Any,Any}, params)

Concatenates all arrays and floats from params to an existing log dictionary.
"
function update_log!(logVar::Dict{Any,Any}, params)
    # iterate through the params type and create apropriate fields
    for name in fieldnames(params)
        strname = string(name)

        # if some of the fields in params are not array, arrayize it
        if supertype(typeof(getfield(params, name))) == AbstractFloat || 
            supertype(typeof(getfield(params, name))) == Signed
            logVar[strname] = cat(1, logVar[strname], [getfield(params, name)])
        else
            logVar[strname] = cat(ndims(getfield(params, name))+1, logVar[strname], getfield(params, name))
        end
    end
end

"
    saveLog(logVar::Dict{Any, Any}, Y, params, priors, logdir::String; desc::String = "")

Saves the log dictionary and inputs to a .jld file specified in logdir.
Name of the log is date_time if unspecified, or desc.
"
function save_log(logVar::Dict{Any, Any}, Y, params, priors, logdir::String; desc::String = "")
    # if no description given, create it from current datetime
    if desc == ""
        desc = Dates.format(now(), "yyyymmdd_HHMMSS")
    end

    # create the dir
    dir = string(logdir, "/", desc)
    mkpath(dir)

    # save the data
    save(string(dir, "/log.jld"), logVar)
    save(string(dir, "/inputs.jld"), "Y", Y, "priors", priors, "params", params)
end

"
    loadLog(path::String) 

Saves the log dictionary to a .jld file specified in logdir.
Name of the log is date_time if unspecified, or desc.
"
function load_log(path::String)
    try
        logVar = load(string(path, "/log.jld"))
        inputs = load(string(path, "/inputs.jld"))
        return logVar, inputs["Y"], inputs["priors"], inputs["params"]
    catch
        ls = readdir(path)
        msg = "The specified folder does not contain any log files but it contains the following:\n"
        for item in ls
            msg = string(msg, item, "\n")
        end
        error(msg)       
        
        return
    end 
end

"
    extractParams!(logVar::Dict{Any, Any}, t::Integer, params)\n

Extracts a single slice with index t from the log dictionary variable given an initialized params type variable.
"
function extract_params!(logVar::Dict, t::Integer, params)
    for name in fieldnames(params)
        strname = string(name)

        # we need just one time slice - time is always the last array dimension
        # but since there are arrays with different number of dimensions
        # we must slice the array dynamically - we cant use [:,t] or [:,:,t]
        dims = size(logVar[strname])
        sliceSize = 1 # how large is one slice
        for d in dims[1:end-1]
            sliceSize*=d
        end

        # this uses linear indices to get the data, then reshapes the final array
        if length(dims) == 1
            setfield!(params, name, logVar[strname][t])
        else
            setfield!(params, name, reshape(logVar[strname][(t-1)*sliceSize+1:t*sliceSize], dims[1:end-1]))
        end    
    end

    return params 
end
