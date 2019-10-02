export CGaussian
export mean_var
export AbstractVar, DiagVar, ScalarVar, UnitVar

"""Abstract variance type"""
abstract type AbstractVar end

"""Diagonal variance represented as a vector"""
struct DiagVar <: AbstractVar end

"""Scalar variance represented as a one-element vector"""
struct ScalarVar <: AbstractVar end

"""Unit variance represented by a vector of ones"""
struct UnitVar <: AbstractVar end

"""
    CGaussian{T,AbstractVar}

Conditional Gaussian that maps an input of `zlength` to its mean of `xlength`.
The mapping must output dimensions appropriate for the chosen variance type

# Arguments
- `xlength::Int`: length of mean
- `zlength::Int`: length of condition
- `mapping`: maps condition z to μ=mapping(z) (e.g. a Flux Chain)
- `AbstractVar`: one of the variance types: DiagVar, ScalarVar, UnitVar

# Example
```julia-repl
julia> p = CGaussian(3, 2, Dense(2, 3))
CGaussian{Float64,UnitVar}(xlength=3, zlength=2, mapping=Dense(2, 3))

julia> mean_var(p, ones(2))
([-0.339991, -0.061213, -0.769473] (tracked), [1.0, 1.0, 1.0])

julia> rand(p, ones(2))
Tracked 3×1 Array{Float64,2}:
 0.1829532154926673
 0.1235498922955946
 0.0767166501426535
```
"""
struct CGaussian{T,V<:AbstractVar} <: AbstractCGaussian{T}
    xlength::Int
    zlength::Int
    mapping
end

function CGaussian(xlength::Int, zlength::Int, mapping::Function, T=Float32)
    V = detect_mapping_variant(mapping, T, xlength, zlength)
    CGaussian{T,V}(xlength, zlength, mapping)
end

function CGaussian(xlength::Int, zlength::Int, mapping)
    T = eltype(first(params(mapping)).data)
    V = detect_mapping_variant(mapping, xlength, zlength)
    CGaussian{T,V}(xlength, zlength, mapping)
end

Flux.@treelike CGaussian

function mean_var(p::CGaussian{T}, z::AbstractArray) where T
    ex = p.mapping(z)
    return ex[1:p.xlength,:], softplus_safe.(ex[p.xlength+1:end,:], T)
end

function mean_var(p::CGaussian{T,UnitVar}, z::AbstractArray) where T
    μ = p.mapping(z)
    σ2 = ones(T, xlength(p))
    σ2 = isa(μ, Array) ? σ2 : σ2 |> gpu
    return μ, σ2
end

function Base.show(io::IO, p::CGaussian{T,V}) where T where V
    e = repr(p.mapping)
    e = sizeof(e)>50 ? "($(e[1:47])...)" : e
    msg = "CGaussian{$T,$V}(xlength=$(p.xlength), zlength=$(p.zlength), mapping=$e)"
    print(io, msg)
end
