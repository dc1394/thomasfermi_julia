module Load2_module
    using Interpolations

    struct Load2_module_param
        LAMBDA::Float64
        THRESHOLD::Float64
        K::Float64
    end

    mutable struct Load2_module_variables
        x::Array{Float64}
        yitp::Union{Interpolations.GriddedInterpolation{Float64, 1, Float64, Interpolations.Gridded{Interpolations.Linear}, Tuple{Vector{Float64}}}, Nothing}
    end
end