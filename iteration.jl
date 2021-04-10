module Iteration
    include("readinputfile.jl")
    include("shoot.jl")
    include("solve_tf.jl")
    using .Readinputfile
    using .Shoot
    using .Solve_TF
    using Printf

    function construct(opt)
        rif_val = Readinputfile.construct(opt.inpname)
        data = Readinputfile.readfile(rif_val)
    
        dx = data.xmax / float(data.grid_num)

        shoot_val = Shoot.construct(dx, data)
        xarray, yarray = Shoot.shootf!(data.xmin, data.xmax, data.matching_point, shoot_val)
        solve_tf_param, solve_tf_val = Solve_TF.construct!(data, xarray, yarray)
        return data, solve_tf_param, solve_tf_val, yarray
    end

    getNormRD(newarray, oldarray) = let
        tmp = newarray .- oldarray
        tmp .*= tmp

        return sqrt(sum(tmp))
    end

    function iteration(data, solve_tf_param, solve_tf_val, yarray)
        normrd = 1.0
        for iter in 0:data.iteration_maxiter - 1        
            newyarray = Solve_TF.solvetf!(iter, solve_tf_param, solve_tf_val, yarray)
            normrd = getNormRD(newyarray, yarray)
            if normrd <= data.iteration_criterion
                break
            end
            @printf("反復回数: %d回, NormRD = %.14f\n", iter + 1, normrd)
            yarray = simple_mixing(data, newyarray, yarray)
        end

        @printf("計算が終了しました。")

        return solve_tf_val.node_x_glo
    end

    simple_mixing(data, newarray, oldarray) = let
        return data.iteration_mixing_weight .* newarray .+ (1.0 - data.iteration_mixing_weight) .* oldarray
    end
end
