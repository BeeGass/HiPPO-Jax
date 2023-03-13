function lssl_kernel(sys, t, dt)
    sys_d = c2d(sys, dt, :tustin)
    L = convert(Int64, div(t, dt, RoundNearest))
    z = exp.((-2im * pi) * (0:inv(L):(L-1)/L))
    return real(ifft([([conj(sys_d.C)] .* ([I] .- ([sys_d.A^L] .* z .^ L)) .* inv.([I] .- ([sys_d.A] .* z)) .* [sys_d.B])...;;], (2,)))
end