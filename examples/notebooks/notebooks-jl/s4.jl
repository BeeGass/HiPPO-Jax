include("hippo.jl")
using SpecialMatrices
using FFTW

FAST_DIAGONALIZE = true

function s4_kernel_legs(sys, t, dt)
    dim = size(sys.B,1)

    low_rank = sqrt.(0.5:dim-0.5);
    (normal = triu(low_rank .* low_rank'); 
     normal .-= normal'; 
     normal -= UniformScaling(0.5));

    if FAST_DIAGONALIZE
        (S_real = ones(dim)*sum(Diagonal(normal))/dim; 
         (S_imag, V) = eigen(Hermitian(normal * -1im)); 
         S = S_real + S_imag * 1im);
    else
        S, V = eigen(normal);
    end

    Λ = Diagonal(S);

    B = V'*sys.B
    low_rank = V'*low_rank

    sys_d = c2d(ss(Λ - low_rank*low_rank', sys.B, sys.C * V, 0), dt, :tustin);

    L=convert(Int64,div(t,dt,RoundNearest));
    Ω = exp.((-2im*pi)*(0:inv(L):(L-1)/L));

    g = 2/∆ .* ((1 .- Ω) ./ (1 .+ Ω))
    c = 2 ./ (1 .+ Ω)
    M = Cauchy(g, -S)

    C = (I - sys_d.A^L)' * transpose(sys_d.C)

    r00 = M * (conj(C) .* B)
    r01 = M * (conj(C) .* low_rank)
    r10 = M * (conj(low_rank) .* B)
    r11 = inv.(1 .+ (M * (conj(low_rank) .* low_rank)))

    return -real(ifft(c .* (r00 .- r01 .* r11 .* r10)))
end