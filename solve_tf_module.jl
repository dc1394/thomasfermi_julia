module Solve_TF_module
    using LinearAlgebra
        
    struct Solve_TF_param
        ELE_TOTAL::Int64
        INTEGTABLENUM::Int64
        NODE_TOTAL::Int64
        X_MAX::Float64
        X_MIN::Float64
        Y0::Float64
        YMAX::Float64
    end

    mutable struct Solve_TF_variables
        length::Array{Float64, 1}
        mat_A_ele::Array{Float64, 3}
        mat_A_glo::SymTridiagonal{Float64,Array{Float64,1}}
        node_num_seg::Array{Int64, 2}
        node_x_ele::Array{Float64, 2}
        node_x_glo::Array{Float64, 1}
        ug::Array{Float64, 1}
        vec_b_ele::Array{Float64, 2}
        vec_b_glo::Array{Float64, 1}
        w::Array{Float64, 1}
        x::Array{Float64, 1}
    end
end