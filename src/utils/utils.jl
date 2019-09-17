export train!

# for over/underflow in logs
"""
	softplus_safe(x,T=Float32)

A softplus with small additive constant for safe operations.
"""
softplus_safe(x,T=Float32) = softplus(x) .+ T(1e-6)

"""
    function layer_builder(d,k,o,n,ftype,lastlayer = "",ltype = "Dense")

Create a chain with `n` layers of with `k` neurons with activation function `ftype`.
Input and output dimension is `d` / `o`.
If lastlayer is not specified, all layers use the same function.
If lastlayer is "linear", then the last layer is forced to be Dense with linear activation.

It is also possible to specify dimensions in a vector.
"""
layer_builder(k::Vector{Int},l::Vector,f::Vector) = Flux.Chain(map(i -> i[1](i[3],i[4],i[2]),zip(l,f,k[1:end-1],k[2:end]))...)

layer_builder(d::Int,k::Int,o::Int,n::Int, args...) =
    layer_builder(vcat(d,fill(k,n-1)...,o), args...)

function layer_builder(ks::Vector{Int},ftype::String,lastlayer::String = "",ltype::String = "Dense")
    ftype = (ftype == "linear") ? "identity" : ftype
    ls = Array{Any}(fill(eval(:($(Symbol(ltype)))),length(ks)-1))
    fs = Array{Any}(fill(eval(:($(Symbol(ftype)))),length(ks)-1))
    if !isempty(lastlayer)
        fs[end] = (lastlayer == "linear") ? identity : eval(:($(Symbol(lastlayer))))
        ls[end] = (lastlayer == "linear") ? Dense : ls[end]
    end
    layer_builder(ks,ls,fs)
end

"""
    ae_layer_builder(lsize, activation, layer)

Construct encoder/decoder consisting of `length(lsize)-1` layers of type `layer` with 
`activation` in between. Last activation is always `identity`. Width of layers is defined
by the vector `lsize`.
"""
ae_layer_builder(lsize::Vector, activation, layer)=  
    layer_builder(lsize, 
        Array{Any}(fill(layer, size(lsize,1)-1)), 
        Array{Any}([fill(activation, size(lsize,1)-2); identity]))

### training ###
"""
    update(model, optimiser)

Update model parameters using optimiser.
"""
function update!(model, optimiser)
    for p in params(model)
        Δ = Flux.Optimise.apply!(optimiser, p.data, p.grad)
        p.data .-= Δ
        p.grad .= 0
    end
end

"""
    loss_back_update!(model, data, loss, opt)

Basic training step - computation of the loss, backpropagation of gradients and optimisation 
of weights. The loss and opt arguments can be arrays/lists/tuples.
"""
function loss_back_update!(model, data, loss, opt)
    l = loss(data)
    Flux.Tracker.back!(l)
    update!(model, opt)
end 

"""
    train!(model, data, loss, optimiser, callback; [usegpu, memory_efficient])

Train the model. Function callback(model, batch, loss, opt) is 
called every iteration - use it to store or print training progress, stop training etc. 
"""
function train!(model, data, loss, optimiser, callback; 
    usegpu = false, memory_efficient = false)
    for _data in data
        try
            if usegpu
             _data = _data |> gpu
            end
            loss_back_update!(model, _data, loss, optimiser)
            # now call the callback function
            # can be an object so it can store some values between individual calls
            callback(model, _data, loss, optimiser)
        catch e
            # setup a special kind of exception for known cases with a break
            rethrow(e)
        end
        if memory_efficient
            # this might be important for conv nets running on gpu
            # large nets might still train but slowly
            GC.gc();
        end
    end
end