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

