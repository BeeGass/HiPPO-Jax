include("hippo.jl")
using Random;
using ProgressMeter;
using LinearAlgebra;
using FFTW;
using Zygote;
using Printf;
#using OrdinaryDiffEq
using NNlib;
using Plots;

n = 64;
delay = 1;
θ = 2

function lssl_kernel(sys, t, dt)
    sys_d = c2d(sys,dt,:tustin);
    L=convert(Int64,div(t,dt,RoundNearest));
    z=exp.(complex(0,-2*pi).*(0:1//L:(L-1)//L));
    #return real(ifft([([conj(sys_d.C)] .* ([I] .- ([sys_d.A^L] .* z.^L)) .* inv.([I] .- ([sys_d.A] .* z)) .* [sys_d.B])...;;],(2,)))
    return real(ifft(reduce(vcat,[conj(sys_d.C),] .* ([I,] .- ([sys_d.A^L,] .* z.^L)) .* inv.([I,] .- ([sys_d.A,] .* z)) .* [sys_d.B,])))
end

#function diffeq_kernel(sys, t, dt)
#    L = convert(Int64,div(t,dt,RoundNearest));
#    x0 = sys.B[:]
#    P = ss(sys.A, zero(sys.B), sys.C, 0)
#    reference(x,t) = [zero(Float64)]
#    s = Simulator(P, reference)
#    sol = solve(s, x0, (0.0,t), alg=BS3(), tstops=1//L:t//L:t, adaptive=true)
#    return s.y(sol, 1//L:t//L:t)[:] .* t//L
#end

function train_delay(rng, sys, L, steps)
    function forward(sys, θ, x)
        kernel = lssl_kernel(sys, θ, θ//size(x,1))[:]
        y = reshape(NNlib.conv(reshape(x, (:, 1, 1)), reshape(kernel, (:, 1, 1)); pad=length(kernel)),:)
        return y[1:size(x,1)]
    end

    function loss(ŷ, y)
        sum((ŷ .- y).^2)
    end

    function eval_loss(sys, θ, x, y)
        ŷ = forward(sys, θ, x)
        l = loss(ŷ, y)
        return l
    end

    seq = zeros(Float64, (L*θ*3÷2,))
    C = sys.C
    
    for i in 1:steps
        x = @view(seq[end-L*θ+1:end])
        y = @view(seq[begin:begin+L*θ-1])
        randn!(rng, x)
        #C -= Zygote.gradient(C -> sum(forward(ss(sys.A,sys.B,C,0), θ, x)), C)[1]
        C .-=  Zygote.gradient(C -> eval_loss(ss(sys.A,sys.B,C,0), θ, x, y), C)[1] .* 0.01
        #return grad
    end
    ss(sys.A,sys.B,C,0)
end

sys = legt_ss(n,θ);
C = transpose(ones(size(sys.C)[1])) 

rng = Xoshiro(1)
sys_delay = ss(sys.A,sys.B,C*0.1,0)
display(plot(lssl_kernel(sys_delay, θ, 1//256); discord...))
sys_delay = train_delay(rng, sys_delay, 32, 200)
display(plot(lssl_kernel(sys_delay, θ, 1//256); discord...))
sys_delay = train_delay(rng, sys_delay, 64, 200)
display(plot(lssl_kernel(sys_delay, θ, 1//256); discord...))
sys_delay = train_delay(rng, sys_delay, 128, 200)
display(plot(lssl_kernel(sys_delay, θ, 1//256); discord...))
sys_delay = train_delay(rng, sys_delay, 256, 200)
display(plot(lssl_kernel(sys_delay, θ, 1//256); discord...))
#sys_delay = train_delay(sys_delay, 16, 16)
#sys_delay = train_delay(sys_delay, 32, 32)
#sys_delay = train_delay(sys_delay, 64, 64)
#sys_delay = train_delay(sys_delay, 128, 128)
#
#plot(lssl_kernel(sys_delay, θ, 1//256))


#y = sol.u
#y, t, x = impulse(sys, tfinal, alg=RK4(), abstol=1e-12, tstops=t);

#display(plot(1//L:1//L:tfinal, s.y(sol, 1//L:1//L:tfinal)[:], xlabel="Time", xlims=(0, tfinal), ylimit=(-2, 5), title="HiPPO-LegS Unit Impulse Response\n(OrdinaryDiffEq)", lab=nothing));