include("comlineoption.jl")
include("iteration.jl")
include("saveresult.jl")
using .Comlineoption
using .Iteration
using .Saveresult

function main(args)
    opt = Comlineoption.construct(args)
    data, solve_tf_param, solve_tf_val, yarray = Iteration.construct(opt)
    xarray = Iteration.iteration(data, solve_tf_param, solve_tf_val, yarray)
    Saveresult.saveresult(data, xarray, yarray)
end

@time main(ARGS)