module Data_module
    const CHEMICAL_NUMBER = "chemical.number"

    const EPS_DEFAULT = 1.0E-15

    const GAUSS_LEGENDRE_INTEG_DEFAULT = 2

    const GRID_NUM_DEFAULT = 20000
    
    const ITERATION_CRITERION_DEFAULT = 1.0E-13

    const ITERATION_MAXITER_DEFAULT = 1000

    const ITERATION_MIXING_WEIGHT_DEFAULT = 0.1

    const XMAX_DEFAULT = 100.0

    const XMIN_DEFAULT = 1.0E-5

    const X_MATCHING_POINT_DEFAULT = 11.0

    mutable struct Data_val
        usethread::Bool
        eps::Union{Float64, Nothing}
        gauss_legendre_integ_num::Union{Int32, Nothing}
        grid_num::Union{Int32, Nothing}
        iteration_criterion::Union{Float64, Nothing}
        iteration_maxiter::Union{Int32, Nothing}
        iteration_mixing_weight::Union{Float64, Nothing}
        xmax::Union{Float64, Nothing}
        xmin::Union{Float64, Nothing}
        x_matching_point::Union{Float64, Nothing}
        Z::Union{Float64, Nothing}
    end
end