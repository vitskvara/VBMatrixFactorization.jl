using JLD
using PyPlot
include("../src/util.jl")
using StatsBase

"""
    getBag(data, field, id)

Extracts a field from a MIL data variable given an id of a bag.
"""
function getBag(data::Dict{String,Any}, field::String, id::Int)
    s = size(data[field])
    inds = (data["bagids"] .== id)
    if length(s) == 1
        return data[field][inds]
    else
        return data[field][:, inds]
    end
end

"""
    getY(data, id)

Extracts a Y matrix from a MIL data variable given the id of a bag.
"""
function getY(data::Dict{String,Any}, id::Int)
    res = getBag(data, "fMat", id)
    return convert(Array{Float64, 2}, res)
end

"""
    getLabel(data, id)

Extracts label of a bag with index id.
"""
function getLabel(data::Dict{String,Any}, id::Int)
    return getBag(data, "y", id)[1]
end

"""
    get_matrices(data, bag_ids)

For given bag indices, extracts negative and positive bags and returns them as a matrix.
"""
function get_matrices(data::Dict{String,Any}, bag_ids)
    Y0 = 0
    Y1 = 0
    n0 = 0
    n1 = 0
    inds0 = 0
    inds1 = 0
    for i in bag_ids
        label = getLabel(data,i);
        mat = getY(data, i);
        if label == 1
            n1 += 1
            if Y1 == 0
                Y1 = mat;
                inds1 = repmat([i], size(mat)[2]);
            else
                Y1 = cat(2, Y1, mat);
                inds1 = cat(1, inds1, repmat([i], size(mat)[2]));
            end
        else
            n0 += 1
            if Y0 == 0
                Y0 = mat;
                inds0 = repmat([i], size(mat)[2]);
            else
                Y0 = cat(2, Y0, mat)
                inds0 = cat(1, inds0, repmat([i], size(mat)[2]));

            end
        end        
    end
    
    # returns the concatenated negative and positive matrices, their numbers and indices of
    # the bags for slicing
    return Y0, Y1, n0, n1, inds0, inds1
end

"""
    train(data, bag_ids, solver, H, niter; verb::Bool = true)

Using data and a vector of bag ids, trains the classifier using a given solver.
The inner dimension of the factorization is H.

solver = basic - calls vbmf()
solver = sparse - calls vbmf_sparse()

"""
function train(data::Dict{String,Any}, bag_ids, solver::String, H::Int, niter::Int; eps::Float64 = 1e-6, 
    verb::Bool = true, diag_var::Bool = false)
    Y0, Y1, n0, n1, inds0, inds1 = get_matrices(data, bag_ids)

    if n0 == 0 || n1 ==0
        warn("One of the input matrices is empty, ending.")
        return 0, 0
    end

    if verb
        println("Extracted $n0 negative and $n1 positive bags.")
    end

    L, M0 = size(Y0)
    L, M1 = size(Y1)

    if solver == "basic"
        res0 = VBMatrixFactorization.vbmf_init(Y0, H)
        VBMatrixFactorization.vbmf!(Y0, res0, niter, eps = eps, est_covs = true, est_var = true, verb = verb)
        res1 = VBMatrixFactorization.vbmf_init(Y1, H)
        VBMatrixFactorization.vbmf!(Y1, res1, niter, eps = eps, est_covs = true, est_var = true, verb = verb)
    elseif solver == "sparse"
        # this is to decide whether to compute full covariance of A or just the diagonal
        if (H*M0 > 200) || (H*M1 > 200)
            full_cov = false
        else
            full_cov = true
        end
        res0 = VBMatrixFactorization.vbmf_sparse_init(Y0, H)
        VBMatrixFactorization.vbmf_sparse!(Y0, res0, niter, eps = eps, diag_var = diag_var, verb = verb, full_cov = full_cov)
        res1 = VBMatrixFactorization.vbmf_sparse_init(Y1, H)
        VBMatrixFactorization.vbmf_sparse!(Y1, res1, niter, eps = eps, diag_var = diag_var, verb = verb, full_cov = full_cov)
    else
        error("Unknown type of solver. Use 'basic' or 'sparse'.")
        return
    end

    return res0, res1
end

"""
    ols(Y::Array{Float64, 2}, B::Array{Float64, 2})

Solves Y = B*X + E for unknown X.
"""
function ols(Y::Array{Float64, 2}, B::Array{Float64, 2})
    return inv(B'*B)*B'*Y;
end

"""
    classify_one(res0, res1, Y)

Using training data res0 and res1, classifies the specimen Y.
"""
function classify(res0, res1, Y::Array{Float64, 2})
    # compute the ols estimate of YHat and choose the label
    # depending on the distance to the real Y matrix.
    B0 = res0.BHat
    YHat0 = B0*ols(Y, B0)
    err0 = norm(Y - YHat0)

    B1 = res1.BHat
    YHat1 = B1*ols(Y, B1)
    err1 = norm(Y - YHat1)

    if err0 > err1
        label = 1
    else
        label = 0
    end

    return label, err0, err1
end

"""
    test_one(res0, res1, Y, bag_id, data)

Using training data res0 and res1, test the classification of Y.
Returns one of the set {-1,0,1} = {false positive, match, false negative}.
"""
function test_one(res0, res1, bag_id::Int, data::Dict{String,Any})
    Y = getY(data, bag_id)

    label = getLabel(data, bag_id)

    est_label, err0, err1 = classify(res0, res1, Y)

    return label - est_label
end

"""
    test(res0, res1, data, bag_ids)

For given bag_ids, it tests them all against a traning dataset. Returns 
mean error rate, equal error rate and false positives and negatives count.
"""
function test_classification(res0, res1, data::Dict{String,Any}, bag_ids)
    n = size(bag_ids)[1]
    n0 = 0 # number of negative/positive bags tested
    n1 = 0

    fp = 0 # number of false positives
    fn = 0 # number of false negatives
    for id in bag_ids
        res = test_one(res0, res1, id, data)
        
        if res == 1
            fn += 1
        elseif res == -1
            fp += 1
        end

        # also we count number of positive and negative try bags
        label = getLabel(data, id)
        if label == 0
            n0 +=1
        else
            n1 += 1
        end
    end

    mer = (fp+fn)/n
    eer = (fp/n0 + fn/n1)/2

    return mer, eer, fp, fn, n0, n1
end

"""
    validate(p_known, data, niter, solver, H; eps = 1e-6)

For a dataset and a percentage of known labels, asses the classification.
"""
function validate(p_known::Float64, data::Dict{String,Any}, niter::Int, solver::String, H::Int; eps::Float64 = 1e-6, 
    verb::Bool = true, diag_var::Bool = false)
    nBags = data["bagids"][end]

    rand_inds = sample(1:nBags, nBags, replace = false);
    train_inds = rand_inds[1:Int(floor(p_known*nBags))];
    test_inds = rand_inds[Int(floor(p_known*nBags))+1:end];

    # training
    res0, res1 = train(data, train_inds, solver, H, niter, eps = eps, verb = verb, diag_var = diag_var);
    if res0 == 0
        return -1.0, -1.0, -1.0, -1.0, -1.0, -1.0
    end

    # validation
    mer, eer, fp, fn, n0, n1 = test_classification(res0, res1, data, test_inds)

    return mer, eer, fp, fn, n0, n1
end

"""
    validate_with_cvs(data::Dict{String,Any}, which::Int, niter::Int, solver::String, H::Int; eps::Float64 = 1e-6, 

For a dataset and a cv_index array index "which", asses the classification.
"""
function validate_with_cvs(data::Dict{String,Any}, which::Int, niter::Int, solver::String, H::Int; eps::Float64 = 1e-6, 
    verb::Bool = true, diag_var::Bool = false)
    nBags = data["bagids"][end]
    
    train_inds = data["cvindexes"][which];
    # select test indices
    test_inds = 1:nBags
    for i in 1:size(train_inds)[1]
        test_inds = test_inds[test_inds .!= train_inds[i]]
    end

    # training
    res0, res1 = train(data, train_inds, solver, H, niter, eps = eps, verb = verb, diag_var = diag_var);
    if res0 == 0
        return -1.0, -1.0, -1.0, -1.0, -1.0, -1.0
    end

    # validation
    mer, eer, fp, fn, n0, n1 = test_classification(res0, res1, data, test_inds)

    return mer, eer, fp, fn, n0, n1
end

"""
    validate_dataset(data::Dict{String,Any}, inputs::Dict{Any, Any}; verb::Bool = true)

Validates classification using vbmf on a whole MIL dataset using vector of 
percentages of known labels. Inputs contain the vector of  percentages of known
labels, number of iterations over percentages and in vbmf algorithm etc. 
Example of input:

inputs = Dict()
inputs["p_vec"] =  [0.01, 0.02, 0.05, 0.1, 0.33, 0.5, 0.75, 0.9]
inputs["nclass_iter"] = 1 # how many times is iterated over one p
inputs["niter"] = 200 # vbmf algorithm iterations
inputs["eps"] = 1e-3 # convergence criterion for vbmf
inputs["solver"] = "basic" # basic/sparse vbmf
inputs["H"] = 1 # inner dimension of factorization
inputs["scale_y"] = true # should Y be scaled to standard distribution?
inputs["use_cvs"] = true # should cv_indexes be used instead of p_vec?
inputs["diag_var"] = true # should homo- or heteroscedastic noise model be used?
"""
function validate_dataset(data::Dict{String,Any}, inputs::Dict{Any, Any}; verb::Bool = true)
    p_vec = inputs["p_vec"]
    nclass_iter = inputs["nclass_iter"]
    np = size(p_vec)[1]

    res_mat = Array{Any,2}(nclass_iter*np, 7) # matrix of resulting error numbers 

    # always compute the cv indexes partitioning first
    if inputs["use_cvs"]
        println("using cv_indexes")
        print("n = ")
        a,b = size(data["cvindexes"])
        ncvs = a*b
        cv_res_mat = Array{Any,2}(ncvs, 7) # matrix of resulting error numbers for cv_indexes

        for n in 1:ncvs
            print("$(n) ")
            cv_res_mat[n,1] = "cvs"
            try
                mer, eer, fp, fn, n0, n1 = validate_with_cvs(data, n, inputs["niter"], inputs["solver"], 
                    inputs["H"], eps = inputs["eps"], verb = verb, diag_var = inputs["diag_var"])
                cv_res_mat[n,2:end] = [mer, eer, fp, fn, n0, n1] 
            catch y 
                warn("Something went wrong during vbmf, no output produced.")
                println(y)
                cv_res_mat[n,2:end] = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0] 
            end         
        end
    end
    println("")

    # then the partitioning given by a percentage of known bags p
    for ip in 1:np
        p = p_vec[ip]
        println("p = $(p) ")
        print("n = ")    
        
        for n in 1:nclass_iter
            print("$(n) ")    
            res_mat[(ip-1)*nclass_iter+n,1] = p
            try
                mer, eer, fp, fn, n0, n1 = validate(p, data, inputs["niter"], inputs["solver"], 
                    inputs["H"], eps = inputs["eps"], verb = verb, diag_var = inputs["diag_var"])
                res_mat[(ip-1)*nclass_iter+n,2:end] = [mer, eer, fp, fn, n0, n1] 
            catch y 
                warn("Something went wrong during vbmf, no output produced.")
                println(y)
                res_mat[(ip-1)*nclass_iter+n,2:end] = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0] 
            end          
        end
    end
    println("")

    if inputs["use_cvs"]
        res = cat(1, cv_res_mat, res_mat)
    else
        res = res_mat
    end

    return res
end

"""
    validate_datasets(inputs::Dict{Any, Any}, file_inds::UnitRange{Int64}, input_path::String, output_path::String)

Wrapper for validate_dataset() that takes a whole folder of inputs, some settings and computes data for evaluation of the vbmf 
classification.
"""
function validate_datasets(inputs::Dict{Any, Any}, file_inds::UnitRange{Int64}, input_path::String, output_path::String; 
    verb::Bool = true)

    files = readdir(input_path)
    #if verb
    #    println("The directory $input_path contains the following files:")
    #    for file in files
    #        println(file)
    #    end
    #    println("")
    #end

    mkpath(output_path)

    # loop through all the files, train them using vbmf, then validate the classification using a testing dataset
    # then save the results

    for file in files[file_inds]
        dataset_name, suf = split(file, ".")
        if suf != "jld" # if the file is not a .jld file, move on
            continue
        end

        if verb
            println("Processing file $file...")
        end

        # load the data
        data = load(string(mil_path, "/", file));
        
        # now decide whether to ignore some rows
        Y = convert(Array{Float64, 2}, data["fMat"])
        sY = scaleY(Y)
        rowsums = sum(abs(sY), 2)
        L, M = size(Y)
        used_rows = 1:L
        used_rows = used_rows[rowsums .>= 1e-5]

        if verb
            println("Original problem size: $L rows, $(L-size(used_rows)[1]) rows not relevant and are not used.")
        end

        # now replace the original Y matrix with a new one
        if inputs["scale_y"]
            data["fMat"] = copy(sY[used_rows,:])
        else
            data["fMat"] = copy(Y[used_rows,:])
        end

        # perform testing of the classification on the dataset
        inputs["dataset_name"] = dataset_name
        res_mat = validate_dataset(data, inputs, verb = false)

        # save the outputs and inputs
        fname = string(dataset_name, "_", inputs["solver"], "_", inputs["H"], "_", inputs["nclass_iter"])
        save("$output_path/$fname.jld", "res_mat", res_mat, "inputs", inputs)

        if verb
            println("Done.")
            println()
        end
    end
end

"""
    warmup()

Serves to precompile the validate_datasets() function.
"""
function warmup(mil_path::String)
    print("Precompiling the validation function... ")
    inputs = Dict()
    inputs["p_vec"] =  [0.01] 
    inputs["nclass_iter"] = 1 # how many times should be bags randomly assigned and classification tested over one percentage of known labels
    inputs["niter"] = 1 # iterations for vbmf solver
    inputs["eps"] = 1e-3 # the convergence limit for vbmf
    inputs["solver"] = "basic" # basic/sparse for non/full ARD on A matrix in vbmf
    inputs["H"] = 1 # inner dimension of the factorization
    inputs["dataset_name"] = ""
    inputs["scale_y"] = true
    inputs["use_cvs"] = false
    inputs["diag_var"] = false

    output_path = "./warmup_garbage"
    file_inds = 1:1

    validate_datasets(inputs, file_inds, mil_path, output_path, verb = false)

    rm("output_path", force=true, recursive=true)
    print("done. \n")
end

"""
    table_summary(res_mat::Array{Float64,2}; verb::Bool = true)

Returns and possibly prints mean values of error rates for a result of the validate_datasets() function.
"""
function table_summary(class_res::Dict{String,Any}; verb::Bool = true)
    inputs = class_res["inputs"]
    res_mat = class_res["res_mat"]

    p_vec =inputs["p_vec"]
    np = size(p_vec)[1]
    ndiag = size(res_mat)[2]
    ncvs = 0

    mean_table = Array{Any,2}(np, ndiag) # matrix of resulting error numbers 

    # to be able to work with old results
    use_cvs = get(inputs, "use_cvs", false)

    # first the cv indexes statistics
    if use_cvs
        cvs_mean_table = Array{Any,2}(1, ndiag) # matrix of resulting error numbers 
        cv_res_mat = res_mat[res_mat[:,1] .== "cvs", :] # cv result matrix
        cvs_mean_table[1,1] = "cvs"

        cv_res_mat = cv_res_mat[cv_res_mat[:,2] .!= -1.0, :] # throw away lines with computation errors
        if size(cv_res_mat)[1] == 0
            cvs_mean_table[1,2:end] = repmat([-1.0], 1, ndiag-1)
        else
            for i in 2:ndiag
                cvs_mean_table[1,i] = mean(cv_res_mat[!isnan(convert(Array{Float64,1},cv_res_mat[:,i])),i])  # throw away nans
            end  
        end
    end

    for n in 1:np
        p = p_vec[n]
        p_mat = res_mat[res_mat[:,1] .== p, :] # extract just rows with current p-val
        p_mat = p_mat[p_mat[:,2] .!= -1.0, :] # throw away lines with computation errors
        if size(p_mat)[1] == 0
            p_mat = repmat([-1.0], 1, ndiag)
            p_mat[1] = p
        end
        for i in 1:ndiag
            mean_table[n,i] = mean(p_mat[!isnan(convert(Array{Float64,1},p_mat[:,i])),i])  # throw away nans
        end        
    end

    heteroscedastic = get(inputs, "diag_var", false)
    if heteroscedastic
        noise_model = "heteroscedastic"
    else
        noise_model = "homoscedastic"
    end

    if verb
        dataset_name = inputs["dataset_name"]
        H = inputs["H"]
        nclass_iter = inputs["nclass_iter"]
        method = inputs["solver"]
        print("\nMean classsification error, $method solver, dataset $dataset_name, H = $H, $nclass_iter samples, $noise_model noise: \n \n")
        print(" perc. of known labels | error rate | EER | false pos. | false neg. | neg. samples | pos. samples \n")
        print("------------------------------------------------------------------------------------------------------\n")
        if use_cvs
            @printf "        cvs                 %0.3f    %0.3f     %0.1f       %0.1f          %0.1f         %0.1f \n" cvs_mean_table[1,2] cvs_mean_table[1,3] cvs_mean_table[1,4] cvs_mean_table[1,5] cvs_mean_table[1,6] cvs_mean_table[1,7]
        end
        for n in 1:np
            @printf "        %0.2f                %0.3f    %0.3f     %0.1f       %0.1f          %0.1f         %0.1f \n" mean_table[n,1] mean_table[n,2] mean_table[n,3] mean_table[n,4] mean_table[n,5] mean_table[n,6] mean_table[n,7]
        end
    end

    if use_cvs
        res = cat(1, cvs_mean_table, mean_table)
    else
        res = mean_table
    end

    return res
end

"""
   plot_statistics(clas_res::Dict{String,Any}) 

Plots statistics of a validate_datasets() function result.
"""
function plot_statistics(class_res::Dict{String,Any}; verb::Bool = false, save_path::String = "")
    inputs = class_res["inputs"]
    res_mat = class_res["res_mat"]

    p_vec =inputs["p_vec"]
    np = size(p_vec)[1]
    ndiag = size(res_mat)[2]
    stat_names = ["mean error", "equal error rate", "false positives", "false negatives", "negative samples", "positive samples"]

    dataset_name = inputs["dataset_name"]
    H = inputs["H"]
    nclass_iter = inputs["nclass_iter"]
    method = inputs["solver"]
    mean_table = table_summary(class_res, verb = verb)

    # to be able to work with old results
    if isdefined(inputs["use_cvs"])
        use_cvs = true
    else
        use_cvs = false
    end

    # because of cv indexes results are in the same file
    if use_cvs
        cv_mean_table = mean_table[1, :]        
        mean_table = mean_table[2:end, :]
          
        if size(mean_table)[1] == 0
            # nothing to draw
            return
        end
    end
    mean_table = convert(Array{Float64,2}, mean_table)

    # plots
    ioff() # Interactive plotting OFF, necessary for inline plotting in IJulia
    fig = figure("vbmfa classification statistics",figsize=(10,15))
    suptitle("$dataset_name, $method solver, H = $H, $nclass_iter samples")
    subplots_adjust(hspace=0.3)

    # mean values of error rates
    subplot(411) # Create the 1st axis of a 3x1 array of axes
    #ax = gca()
    #ax[:set_yscale]("log") # Set the y axis to a logarithmic scale
    plot(1:np, mean_table[:,2], label = stat_names[1])
    plot(1:np, mean_table[:,3], label = stat_names[2])
    title("Mean error values")
    xlabel("percentage of known labels")
    ylabel("")
    xticks(1:np, p_vec)
    legend()

    # false negatives and positives
    subplot(412)
    plot(1:np, mean_table[:,4], label = stat_names[3])
    plot(1:np, mean_table[:,5], label = stat_names[4])
    plot(1:np, mean_table[:,6], label = stat_names[5])
    plot(1:np, mean_table[:,7], label = stat_names[6])
    xlabel("percentage of known labels")
    ylabel("")
    title("Identification statistics")
    xticks(1:np, p_vec)
    legend()   

    # boxplots
    subplot(413) # Create the 2nd axis of a 3x1 array of axes
    data = []
    for n in 1:np
        p = p_vec[n]
        curr_vec = res_mat[res_mat[:,1] .== p, 2] # extract just rows with current p-val
        curr_vec = curr_vec[curr_vec .!= -1.0] # throw away lines with computation errors
        curr_vec = curr_vec[!isnan(convert(Array{Float64,1},curr_vec))]
        push!(data, curr_vec)
    end
    boxplot(data)
    title("box plot of mean error rate")
    xlabel("percentage of known labels")
    xticks(1:np, p_vec)

    subplot(414) # Create the 2nd axis of a 3x1 arrax of axes
    data = []
    for n in 1:np
        p = p_vec[n]
        curr_vec = res_mat[res_mat[:,1] .== p, 3] # extract just rows with current p-val
        curr_vec = curr_vec[curr_vec .!= -1.0] # throw away lines with computation errors
        curr_vec = curr_vec[!isnan(convert(Array{Float64,1},curr_vec))]
        push!(data, curr_vec)
    end
    boxplot(data)
    title("box plot of equal error rate")
    xlabel("percentage of known labels")
    xticks(1:np, p_vec)

     fig[:canvas][:draw]() # Update the figure
     gcf() # Needed for IJulia to plot inline

     # save the figure
     if save_path != ""
        filename = string(save_path, "/$dataset_name", "_$method", "_$H", "_$nclass_iter.eps")
        savefig(filename, format="eps", dpi=1000);
        println("Saving the figure to $filename.")
     end
end
