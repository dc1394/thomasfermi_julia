include("comlineoption.jl")
include("iteration.jl")
using .Comlineoption
using .Iteration

function main(args)
    opt = Comlineoption.construct(args)
    solve_tf_param, solve_tf_val, yarray = Iteration.construct(opt)
    Iteration.iteration(solve_tf_param, solve_tf_val, yarray)
end

main(ARGS)