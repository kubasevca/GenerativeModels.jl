export FluxDSSDecoder

"""
    FluxDSSDecoder{M}(slength::Int, tlength::Int, dt::Real,
                      model::M, observe::Function)

Uses a Flux `model` as discrete state-space model and solves it for the given time span.
The solver can be conveniently called by (dec::FluxODEDecoder)(z), which
assumes that all parameters of the neural ODE and its initial conditions are
passed in as one long vector i.e.: z = vcat(Flux.destructure(dec.model)[1], u0).
Can use any Flux model as neural ODE. The adjoint is computed via ForwardDiff.

# Arguments
* `slength`: length of the ODE state
* `tlength`: number of ODE solution samples
* `dt`: time step with which ODE is sampled
* `model`: Flux model
* `observe`: Observation operator. Function that receives ODESolution and
  outputs the observation. Default: observe(sol) = reshape(hcat(sol.u...),:)
* `zlength`: length(model) + slength
* `restructure`: function that maps vector of ODE params to `model`
"""
mutable struct FluxDSSDecoder
    slength::Int
    timesteps::Vector
    model
    observe::Function
    zlength::Int
end

# TODO: FluxODEDecoder{M} fails during training because forward diff wants to
#       stick dual number in there...
function FluxDSSDecoder(slength::Int, tlength::Int, dt::T,
                        model, observe::Function, zlength::Int) where T
    timesteps = range(T(0), step=dt, length=tlength)
    # ps, restructure = Flux.destructure(model)
    # zlength = length(ps) + slength
    FluxDSSDecoder(slength, timesteps, model, observe, zlength)
end

function FluxDSSDecoder(slength::Int, tlength::Int, dt::Real, model)
    observe(sol) = reshape(hcat(sol.u...), :)
    FluxDSSDecoder(slength, tlength, dt, model, observe)
end

function (dec::FluxDSSDecoder)(z::AbstractVector, observe::Function)
    dec(z,observe,0)
end    

function (dec::FluxDSSDecoder)(z::AbstractVector, observe::Function, ii::Int)
    @assert length(z) == dec.zlength
    ps = z[1:end-dec.slength]
    u0 = z[end-dec.slength+1:end]

   # dec.model = dec.restructure(ps)
   # z = vcat(Flux.destructure(dec.model)[1], u0)
   # tspan = (dec.timesteps[1], dec.timesteps[end])
   # dudt_(u::AbstractVector, ps, t) = dec.model(u)
   # prob = ODEProblem(dudt_, u0, tspan, ps)
   # sol = solve(prob, Tsit5(), saveat=dec.timesteps)


    #u0[u0 .< 20.] .= 20.
    z = vcat(ps, u0)

    # discrete
    # tspan = (Int32(1), Int32(length(dec.timesteps)))
    # function dss_sim(u::AbstractVector, ps, t)
    #     dec.model(ps,u)
    # end
    # prob = DiscreteProblem(dss_sim,u0,tspan,ps) 
    # sol = solve(prob,FunctionMap())
    # #sol = solve(prob,FunctionMap(scale_by_time = true))
    # observe(sol)

    # ode
    sol = dec.model(ps,u0,dec.timesteps,ii)
    observe(sol)
end

# by default call with stored observe function
(dec::FluxDSSDecoder)(z::AbstractVector) = dec(z, dec.observe)

# by default call with stored observe function and with ii for the proper indexing of input vector in function dec.model
(dec::FluxDSSDecoder)(z::AbstractVector,ii::Int) = dec(z, dec.observe, ii)

# Use loop to get batched reconstructions so that jacobian and @adjoint work...
(dec::FluxDSSDecoder)(Z::AbstractMatrix) = hcat([dec(Z[:,ii],ii) for ii in 1:size(Z,2)]...)

ddec(dec::FluxDSSDecoder, z::AbstractVector) = ForwardDiff.jacobian(dec, z)

ddec(dec::FluxDSSDecoder, z::AbstractVector, ii::Int) = ForwardDiff.jacobian(z -> dec(z,ii), z)

@adjoint function (dec::FluxDSSDecoder)(z::AbstractVector)
    (dec(z), Δ -> (J=(Δ'*ddec(dec, z))'; (nothing,J)))
end

@adjoint function (dec::FluxDSSDecoder)(z::AbstractVector,ii::Int)
    (dec(z,ii), Δ -> (J=(Δ'*ddec(dec, z, ii))'; (nothing,J,ii)))
end


# ddec(dec::FluxDSSDecoder, z::AbstractVector) = ForwardDiff.jacobian(dec, z)

# function ddec(dec::FluxDSSDecoder, z::AbstractVector, ii::Int)
#     actual_ii = copy(ii) 
#     dec_ii(x::AbstractVector) = dec(x,actual_ii)
#     ForwardDiff.jacobian(dec_ii, z)
# end

# @adjoint function (dec::FluxDSSDecoder)(z::AbstractVector)
#     (dec(z,), Δ -> (J=(Δ'*ddec(dec, z))'; (nothing,J)))
# end

# @adjoint function (dec::FluxDSSDecoder)(z::AbstractVector,ii::Int)
#     actual_ii = copy(ii) 
#     dec_ii(x::AbstractVector) = dec(x,actual_ii)
#     (dec_ii(z), Δ -> (J=(Δ'*ddec(dec, z, ii))'; (nothing,J)))
# end
