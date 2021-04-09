module Readinputfile_module
    include("data_module.jl")
    using .Data_module

    mutable struct Readinputfile_variables
        data::Data_module.Data_param
        lines::Array{String, 1}
        lineindex::UInt64
    end
end