export GAN
export generator_loss, discriminator_loss

"""
	GAN{T}([prior::AbstractPDF,] generator::AbstractCPDF, discriminator::AbstractCPDF)

The Generative Adversarial Network.

# Example
Create a GAN with standard normal prior with:
```julia-repl
julia> gen = CGaussian{Float32,UnitVar}(4,2,Dense(2,4))
CGaussian{Float32,UnitVar}(xlength=4, zlength=2, mapping=Dense(2, 4))

julia> disc = CGaussian{Float32,UnitVar}(1,4,Chain(Dense(4,1), x->Flux.σ.(x)))
CGaussian{Float32,UnitVar}(xlength=1, zlength=4, mapping=(Chain(Dense(4, 1), getfield(Main, Symbol("##3#4...))

julia> gan = GAN(gen, disc)
GAN{Float32}:
 prior   = (Gaussian{Float32}(μ=2-element Array{Float32,1}, σ2=2-element Arra...)
 generator = CGaussian{Float32,UnitVar}(xlength=4, zlength=2, mapping=Dense(2, 4))
 discriminator = (CGaussian{Float32,UnitVar}(xlength=1, zlength=4, mapping=(Chain(Den...)
```
"""
struct GAN{T} <: AbstractGAN{T}
	prior::AbstractPDF
	generator::AbstractCPDF
	discriminator::AbstractCPDF

	function GAN{T}(p::AbstractPDF{T}, g::AbstractCPDF{T}, d::AbstractCPDF{T}) where T
        (xlength(d) != 1) ? error("Discriminator output must be scalar.") : nothing
        if xlength(g) == zlength(d) 
            new(p, g, d)
        else
            error("Generator and discriminator dimensions do not fit.")
        end
    end
end

Flux.@treelike GAN

function GAN(g::CGaussian{T}, d::CGaussian{T}) where T
    zlen = zlength(g)
    prior = Gaussian(zeros(T, zlen), ones(T, zlen))
    GAN{T}(prior, g, d)
end

"""
	generator_loss(m::GAN, z::AbstractArray)
	generator_loss(m::GAN, batchsize::Int)

Loss of the GAN generator. The input is either the random code `z` or `batchsize` 
of samples to generate from the model prior and compute the loss from.
"""
generator_loss(m::GAN{T}, z::AbstractArray) where T = generator_loss(T,freeze(m.discriminator.mapping)(mean(m.generator,z)))
generator_loss(m::GAN{T}, batchsize::Int) where T = generator_loss(m, rand(m.prior, batchsize))


"""
	discriminator_loss(m::GAN, x::AbstractArray[, z::AbstractArray])

Loss of the GAN discriminator given a batch of training samples `x` and latent prior samples `z`.
If z is not given, it is automatically generated from the model prior.
"""
discriminator_loss(m::GAN{T}, x::AbstractArray, z::AbstractArray) where T = discriminator_loss(T, mean(m.discriminator,x), mean(m.discriminator, freeze(m.generator.mapping)(z)))
discriminator_loss(m::GAN{T}, x::AbstractArray) where T = discriminator_loss(m, x, rand(m.prior, size(x,2)))

function Base.show(io::IO, m::AbstractGAN{T}) where T
    p = repr(m.prior)
    p = sizeof(p)>70 ? "($(p[1:70-3])...)" : p
    g = repr(m.generator)
    g = sizeof(g)>70 ? "($(g[1:70-3])...)" : g
    d = repr(m.discriminator)
    d = sizeof(d)>70 ? "($(d[1:70-3])...)" : d
    msg = """$(typeof(m)):
     prior   = $(p)
     generator = $(g)
     discriminator = $(d)
    """
    print(io, msg)
end

