function ensure!(d::Dict, key)
    if !(key ∈ keys(d))
        error("Key `$key` not found in dictionary")
    end
end

function ensure!(d::Dict, key, val)
    @assert get(d, key, val) == val
    d[key] = val
end

function suggest!(d::Dict, key, val)
    !isnothing(val) && get!(d, String(key), val)
end

function suggest!(d::Dict; kwargs...)
    for (k, v) in kwargs
        suggest!(d, k, v)
    end
end

function suggest_from!(d::Dict, k1, suggested_keys...)
    for k2 in suggested_keys
        suggest!(d, k1, get(d, k2, nothing))
        if !isnothing(get(d, k1, nothing))
            break
        end
    end
    @assert !isnothing(d[k1])
end

function getfirst(d, keys, default=nothing)
    for k in keys
        if haskey(d, k)
            return d[k]
        end
    end
    return default
end

