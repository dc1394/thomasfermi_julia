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

        shoot_val = Shoot.construct(dx, data.eps, data.xmax, data)
        xarray, yarray = Shoot.shootf(data.xmin, data.xmax, data.matching_point, shoot_val)
        solve_tf_param, solve_tf_val = Solve_TF.construct(data, yarray)
        return solve_tf_param, solve_tf_val, xarray, yarray
    end

    function iteration(solve_tf_param, solve_tf_val, xarray, yarray)
        #for i in 0:1000
            newyarray = Solve_TF.solvepoisson!(0, solve_tf_param, solve_tf_val, xarray, yarray)
            #@printf("NormRD = %.14f\n", NormRD(newyarray, yarray))
            #yarray = newyarray
        #end
    end            

    function NormRD(newarray, oldarray)
        len = length(newarray)
        @printf("%d %d\n", len, length(oldarray))
        tmp = newarray .- oldarray
        normrd = 0.0
        for i in 1:len
            normrd += tmp[i] * tmp[i]
        end

        return normrd
    end
end
