import MultivariateStats: RegressionModel, fit, predict

using Statistics
using MultivariateStats
using MultivariateStats.LinearAlgebra

struct RRR{T<:Real} <: RegressionModel
    xmean::Vector{T}
    ymean::Vector{T}
    B::Matrix{T}
    r::Integer
    λ::Number
end


function fit(::Type{RRR}, X::AbstractMatrix{T}, Y::AbstractMatrix{T};
             r::Int=min(size(X, 1), size(Y, 1)), λ=0.1) where T<:Real
    @debug "λ = $λ, r = $r"
    if r < 1 || r > min(size(X, 1), size(Y, 1))
        throw(ArgumentError("Rank should be between 1 and `min(size(X, 1), size(Y, 1))`"))
    end

    # Center the data
    x̄ = mean(X, dims=2)
    ȳ = mean(Y, dims=2)
    X = X .- x̄
    Y = Y .- ȳ

    B = ridge(X', Y', λ, bias=false, dims=1)
    Ŷₗ = B'*X

    Svd = svd(Ŷₗ)
    n = size(Ŷₗ, 2)
    v = Svd.S
    U = Svd.U
    for i = 1:length(v)
        @inbounds v[i] = abs2(v[i]) / (n-1)
    end
    ord = sortperm(v; rev=true)
    Pᵣ = U[:, ord[1:r]] # A projection based on the PCA of the estimate

    # Produce the reduced rank solution
    B̂ = B*Pᵣ*Pᵣ'

    return RRR(vec(x̄), vec(ȳ), B̂, r, λ)
end

function predict(M::RRR, X::AbstractMatrix{T}) where T<:Real
    X = X .- M.xmean
    Ŷ = M.B'*X .+ M.ymean
end

function show(io::IO, M::RRR)
    xi = length(M.xmean)
    yi = length(M.ymean)
    print(io, "RRR (xindim = $xi, yindim = $yi, rank = $(M.r), λ = $(M.λ))")
end
