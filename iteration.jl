module Iteration
    include("readinputfile.jl")
    include("shoot.jl")
    include("solve_tf.jl")
    using .Readinputfile
    using .Shoot
    using .Solve_TF
    using Printf

    function construct(opt)
        rif_val = Readinputfile.construct(opt.inpname, opt.usethread)
        data = Readinputfile.readfile(rif_val)
    
        dx = data.xmax / float(data.grid_num)

        shoot_val = Shoot.construct(dx, data)
        xarray, yarray = Shoot.shootf(data.xmin, data.xmax, data.matching_point, shoot_val)
        solve_tf_param, solve_tf_val = Solve_TF.construct(data, xarray, yarray)
        return solve_tf_param, solve_tf_val, yarray
    end

    getNormRD(newarray, oldarray) = let
        tmp = newarray .- oldarray
        tmp .*= tmp

        return sqrt(sum(tmp))
    end

    function iteration(solve_tf_param, solve_tf_val, yarray)
        for i in 0:1000
            newyarray = Solve_TF.solvetf!(i, solve_tf_param, solve_tf_val, yarray)
            @printf("NormRD = %.14f\n", getNormRD(newyarray, yarray))
            yarray = simple_mixing(newyarray, yarray)
        end
    end

    simple_mixing(newarray, oldarray) = let
        return 0.05 .* newarray .+ 0.95 .* oldarray
    end

end
