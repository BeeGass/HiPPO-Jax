include("hippo.jl")
using Flux;
using Random;


dt             = inv(128);
sys            = legs_ss(128)
sys_d = c2d(ss(sys.A,sys.B,ones(size(sys.C)[1])',0),dt);


x0 = zeros(size(sys.A)[1]);
x = x0;

function ss_cell(x, u)
    x = sys_d.A*x + sys_d.B*u;
    y = sys_d.C*x + sys_d.D*u;
    return x, y
end

ss_model = Flux.Recur(ss_cell, x);

function loss(u)
    ŷ = [ss_model(ui)[1] for ui in u];
    #y = vcat(zeros(div(length(u),2)),[input[1:div(length(input),2)]...;]);
    y = [ui[1] for ui in u];
    return Flux.Losses.mse(ŷ,y);
end


for i=1:1000
    rng = MersenneTwister(i);
    input = [[i;] for i in randn(rng, 256)];
    grad = gradient(() -> loss(input), Flux.Params([sys_d]));
    sys_d.C .-= grad[sys_d][3];
    if i % 50 == 0
        println(sys_d.C)
    end
end
