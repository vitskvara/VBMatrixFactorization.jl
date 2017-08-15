"""
    getBag(data, field, id)

Extracts a field from a MIL data variable given an id of a bag.
"""
function getBag(data, field, id)
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
function getY(data, id)
    res = getBag(data, "fMat", id)
    return convert(Array{Float64, 2}, res)
end

"""
    getLabel(data, id)

Extracts label of a bag with index id.
"""
function getLabel(data, id)
    return getBag(data, "y", id)[1]
end

"""
    get_matrices(data, bag_ids)

For given bag indices, extracts negative and positive bags and returns them as a matrix.
"""
function get_matrices(data, bag_ids)
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
function train(data, bag_ids, solver, H, niter; eps::Float64 = 1e-6, verb::Bool = true)
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
        init0 = VBMatrixFactorization.vbmf_init(L, M0, H)
        res0 = VBMatrixFactorization.vbmf(Y0, init0, niter, eps = eps, est_covs = true, est_var = true, verb = verb)
        init1 = VBMatrixFactorization.vbmf_init(L, M1, H)
        res1 = VBMatrixFactorization.vbmf(Y1, init1, niter, eps = eps, est_covs = true, est_var = true, verb = verb)
    elseif solver == "sparse"
        init0 = VBMatrixFactorization.vbmf_sparse_init(L, M0, H)
        res0 = VBMatrixFactorization.vbmf_sparse(Y0, init0, niter, eps = eps, est_var = true, verb = verb)
        init1 = VBMatrixFactorization.vbmf_sparse_init(L, M1, H)
        res1 = VBMatrixFactorization.vbmf_sparse(Y1, init1, niter, eps = eps, est_var = true, verb = verb)
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
function test_one(res0, res1, bag_id, data)
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
function test(res0, res1, data, bag_ids)
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
function validate(p_known, data, niter, solver, H; eps = 1e-6, verb = true)
    nBags = data["bagids"][end]

    rand_inds = sample(1:nBags, nBags, replace = false);
    train_inds = rand_inds[1:Int(floor(p_known*nBags))];
    test_inds = rand_inds[Int(floor(p_known*nBags))+1:end];

    # training
    res0, res1 = train(data, train_inds, solver, H, niter, eps = eps, verb = verb);
    if res0 == 0
        return -1.0, -1.0, -1.0, -1.0, -1.0, -1.0
    end

    # validation
    mer, eer, fp, fn, n0, n1 = test(res0, res1, data, test_inds)

    return mer, eer, fp, fn, n0, n1
end

"""
    validate_dataset(data, inputs ;verb = true)

Validates classification using vbmf on a whole MIL dataset using vector of 
percentages of known labels. Inputs contain the vector of  percentages of known
labels, number of iterations over percentages and in vbmf algorithm etc. 
Example of input:

inputs = Dict()
inputs["p_vec"] =  [0.01, 0.02, 0.05, 0.1, 0.33, 0.5, 0.75, 0.9]
inputs["nclass_iter"] = 1
inputs["niter"] = 200
inputs["eps"] = 1e-3
inputs["solver"] = "basic"
inputs["H"] = 1
"""
function validate_dataset(data, inputs ;verb = true)
    p_vec = inputs["p_vec"]
    nclass_iter = inputs["nclass_iter"]
    np = size(p_vec)[1]

    res_mat = zeros(nclass_iter*np, 7) # matrix of resulting error numbers 
    for ip in 1:np
        p = p_vec[ip]
        for n in 1:nclass_iter
            try
                res_mat[(ip-1)*nclass_iter+n,1] = p
                mer, eer, fp, fn, n0, n1 = validate(p, data, inputs["niter"], inputs["solver"], 
                    inputs["H"], eps = inputs["eps"], verb = verb)
                res_mat[(ip-1)*nclass_iter+n,2:end] = [mer, eer, fp, fn, n0, n1] 
            catch y 
                warn("Something went wrong during vbmf, no output produced.")
                res_mat[(ip-1)*nclass_iter+n,1] = p
                res_mat[(ip-1)*nclass_iter+n,2:end] = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0] 
            end          
        end
    end

    return res_mat
end
