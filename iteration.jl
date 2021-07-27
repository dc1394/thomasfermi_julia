module Iteration
    include("readinputfile.jl")
    include("shoot.jl")
    include("solve_tf.jl")
    using Printf
    using .Readinputfile
    using .Shoot
    using .Solve_TF

    function construct(opt)
        rif_val = Readinputfile.construct(opt.inpname, opt.usethread)
        data = Readinputfile.readfile(rif_val)
    
        dx = data.xmax / float(data.grid_num)

        shoot_val = Shoot.construct(dx, data)
        xarray, yarray = Shoot.shootf!(shoot_val)
        solve_tf_param, solve_tf_val = Solve_TF.construct(data, xarray, yarray)
        return data, solve_tf_param, solve_tf_val, yarray
    end

    getNormRD(newarray, oldarray, x) = let
        residual = abs.(newarray .- oldarray)
    
        sum = 0.0
        max = length(residual) - 2
        dx = x[2] - x[1]
        
        # Simpsonの法則で数値積分する
        @inbounds @simd for i = 1:2:max
            f0 = sqrt(x[i]) * (residual[i]) ^ 1.5
            f1 = sqrt(x[i + 1]) * (residual[i + 1]) ^ 1.5 
            f2 = sqrt(x[i + 2]) * (residual[i + 2]) ^ 1.5
            
            sum += (f0 + 4.0 * f1 + f2)
        end
        
        return sum * dx / (12.0 * pi)
    end

    function iteration!(data, solve_tf_param, solve_tf_val, yarray)
        convergence_flag = false
        
        for iter in 0:data.iteration_maxiter - 1
            newyarray = Solve_TF.solvetf!(iter, solve_tf_param, solve_tf_val, yarray)
            normrd = getNormRD(newyarray, yarray, solve_tf_val.node_x_glo)
            
            if normrd <= data.iteration_criterion
                convergence_flag = true
                break
            end
            
            @printf("Iteration # %d: NormRD = %.15f\n", iter + 1, normrd)
            yarray = simple_mixing(data, newyarray, yarray)
        end

        if convergence_flag
            @printf("計算が収束しました．")
        else
            @printf("計算が収束しませんでした．プログラムを終了します．\n")
            exit(1)
        end

        return solve_tf_val.node_x_glo
    end

    simple_mixing(data, newarray, oldarray) = let
        return data.iteration_mixing_weight .* newarray .+ (1.0 - data.iteration_mixing_weight) .* oldarray
    end
end
