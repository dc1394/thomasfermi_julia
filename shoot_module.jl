module Shoot_module
    include("load2_module.jl")
    using .Load2_module

    mutable struct Shoot_module_variables
        dx::Float64
        eps::Float64
        grid_num::Int64
        usethread::Bool
        v0::Float64
        vmax::Float64
        xmin::Float64
        xf::Float64
        xmax::Float64
        load2_param::Load2_module.Load2_module_param
        load2_val::Load2_module.Load2_module_variables
    end
end