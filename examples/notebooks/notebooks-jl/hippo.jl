using LinearAlgebra;
using SpecialFunctions;
using ControlSystems;


# Translated Legendre
"""
    sys = legt_ss([T::Type{<:AbstractFloat}=Float64,] dim::Integer[, theta::Real=1])

Create an orthogonal state-space model `sys::StateSpace{Continuous, T}` derived from the Translated Legendre measure.
"""
function legt_ss end

function legt_ss(T::Type{<:AbstractFloat}, dim::Integer, theta::Real=1)
    A, B = setprecision(precision(T) * 2) do
        B = sqrt.(big.(1:2:dim+dim))
        A = -flipsign.(convert(Matrix{T}, (inv(big(theta)) .* B * B')), ((-one(Int8)) .^ triu([1:dim;] .- [1:dim;]')))
        return A, convert(Vector{T}, B)
    end

    legt = ss(A, B, Matrix(I, dim, dim), 0)
    return legt
end

legt_ss(dim::Integer, theta::Real=1) = legt_ss(Float64, dim, theta)


# Translated Laguerre
"""
    lagt_ss([T::Type{<:AbstractFloat}=Float64,] dim::Integer[[, alpha::Real=0], beta::Real=1])

Create an orthogonal state-space model `sys::StateSpace{Continuous, T}` derived from the Translated Laguerre measure.
"""
function lagt_ss end

function lagt_ss(T::Type{<:AbstractFloat}, dim::Integer, alpha::Real=0, beta::Real=1)
    A = -tril(ones(dim, dim), -1) - ((1 + beta) / 2) * I
    B = Vector{Float64}(1:dim)
    B = gamma.((B .+ alpha)) ./ (gamma.(B) .* gamma(alpha + 1))

    L = exp.(0.5 .* (loggamma.((B .+ beta)) .- loggamma.(B)))
    A = inv.(L) .* A .* L
    B = inv.(L) .* B .* exp.(-0.5 .* loggamma.((B .+ beta))) .* beta^((1 - alpha) / 2)

    A = T.(A)
    B = T.(B)

    return ss(A, B, Matrix(I, dim, dim), 0)
end

lagt_ss(T::Type{<:AbstractFloat}, dim::Integer, beta::Real=1) = lagt_ss(T, dim, 0, beta)
lagt_ss(T::Type{<:AbstractFloat}, dim::Integer) = lagt_ss(T, dim, 1)
lagt_ss(dim::Integer, alpha::Real=0, beta::Real=1) = lagt_ss(Float64, dim, alpha, beta)
lagt_ss(dim::Integer, beta::Real=1) = lagt_ss(dim, 0, beta)
#lagt_ss(dim::Integer) = lagt_ss(dim, 1)


# Scaled Legendre
"""
    legs_ss([T::Type{<:AbstractFloat},] dim::Integer)

Create an orthogonal state-space model `sys::StateSpace{Continuous, T}` derived from the Scaled Legendre measure.
"""
function legs_ss end

function legs_ss(T::Type{<:AbstractFloat}, dim::Integer)
    q = 0:dim-1
    r = 2 .* q .+ 1
    M = -(tril(ones(Int64, dim) * r') - Diagonal(q))
    V = sqrt(Diagonal(r))
    A = T.(V * M * inv(V))
    B = T.(Diagonal(V))

    legs = ss(A, B, Matrix(I, dim, dim), 0)
    return legs
end


# Translated Fourier
"""
    fout_ss([T::Type{<:AbstractFloat},] dim::Integer[, theta::Real=1])

Create an orthogonal state-space model `sys::StateSpace{Continuous, T}` derived from the Translated Fourier measure.
"""
function fout_ss end

function fout_ss(T::Type{<:AbstractFloat}, dim::Integer, theta::Real=1)

    # Construct the normal component of the system/state transition matrix
    d = pi * [zeros(Int64, div(dim, 2)) [0:2:dim-1;]]'[2:dim] * theta
    normal = Tridiagonal(d, zeros(dim), -d) # Use a Tridiagonal view to prevent creating a dense matrix

    # Construct the low-rank component of the system/state transition matrix
    low_rank = zeros(dim)
    low_rank[1:2:dim] .= 2.0
    low_rank[1] = √2

    # Constuct A from normal and low-rank
    A = T.(normal .- low_rank .* low_rank' .* theta)
    A[1, 1] = T(-2.0 * theta) # Fix (√2.0)² ≠ 2.0

    # Copy B from A for efficency and numerics
    B = -copy(A[1, :])

    return ss(A, B, Matrix(I, dim, dim), 0)
end

fout_ss(T::Type{<:AbstractFloat}, dim::Integer) = fout_ss(T, dim, 1)
fout_ss(dim::Integer, theta::Real=1) = fout_ss(Float64, dim, theta)
#fout_ss(dim::Integer) = fout_ss(Float64, dim)


# Diagonal Fourier
"""
    foud_ss([T::Type{<:AbstractFloat},] dim::Integer)
"""
function foud_ss end

function foud_ss(T::Type{<:AbstractFloat}, dim::Integer)
    d = T.(pi * ([zeros(div(dim, 2)) [0:2:dim-1;]]'[:])[2:dim])
    A = Tridiagonal(d, fill(T(-1 // 2), dim,), -d)

    B = zeros(T, dim)
    B[1:2:dim] .= T(√2)
    B[1] = one(T)

    fout = ss(A, B, Matrix(I, dim, dim), 0)
    return fout
end

foud_ss(dim::Integer) = foud_ss(Float64, dim)

function chebyshev_polys(order)
    polys = zeros(Int64, order, order)
    polys[begin, begin] = 1
    polys[begin+1, begin+1] = 1

    T_previous = @view polys[begin, :]
    T_current = @view polys[begin+1, :]

    for T_next in eachrow(@view polys[begin+2:end, :])
        circshift!(T_next, T_current, (1,))
        T_next .*= 2
        T_next .-= T_previous

        T_previous = T_current
        T_current = T_next
    end

    return polys
end

polys = chebyshev_polys(16)

d_polys = zeros(eltype(polys), size(polys));
d_polys[:, begin:end-1] .= @view polys[:, begin+1:end];
d_polys .*= transpose((1:size(d_polys, 1)))

function legendre_polys!(polys::AbstractMatrix{<:Real})
    polys[begin, begin] = 1
    polys[begin+1, begin+1] = 1

    T_previous = @view polys[begin, :]
    T_current = @view polys[begin+1, :]

    for (n, T_next) in enumerate(eachrow(@view polys[begin+2:end, :]))
        circshift!(T_next, T_current, (1,))
        T_next .*= 2 .* n .+ 1
        T_next .-= n .* T_previous
        T_next ./= n .+ 1

        T_previous = T_current
        T_current = T_next
    end
end

function legendre_polys(T::Type{<:Real}, order::Integer)
    polys = zeros(T, (order, order))
    legendre_polys!(polys)
    return polys
end

A = legendre_polys(Rational{Int32}, 16);
display(A * (transpose([-1, -1 // 2, 0, 1 // 2, 1]) .^ (Base.OneTo(size(A, 1)) .- 1)));