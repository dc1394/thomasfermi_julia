module Shoot
    include("load2.jl")
    include("load2_module.jl")
    include("shoot_module.jl")
    using DifferentialEquations
    using .Load2
    using .Shoot_module

    const DELV = 1.0E-7
    const NVAR = 2
    const VMIN = -1.588071022611375

    function construct(dx, data)
        load2_param, load2_val = Load2.construct()
        shoot_val = Shoot_module.Shoot_module_variables(
            dx,
            data.eps,
            data.grid_num,
            data.usethread,
            VMIN,
            Load2.make_vmax(data.xmax, load2_param, load2_val),
            data.xmin,
            data.matching_point,
            data.xmax,
            Shoot_module.Load2_module.Load2_module_param(load2_param.LAMBDA, load2_param.THRESHOLD, load2_param.K),
            Shoot_module.Load2_module.Load2_module_variables(load2_val.x, load2_val.yitp))
    
        return shoot_val
    end

    createresult(xarray1, xarray2, yarray1, yarray2) = let
        xarray = append!(xarray1[2:end], xarray2[2:end])
        yarray1[end] = (yarray1[end] + yarray2[1]) / 2.0
        yarray = append!(yarray1[2:end], yarray2[2:end])

        xar = [0.0]
        yar = [1.0]
        return append!(xar, xarray), append!(yar, yarray)
    end

    function funcx1!(dfdv, f1, f_vector, shoot_val)
        sav = shoot_val.vmin
        shoot_val.vmin += DELV

        u0 = load1(shoot_val.xmin, shoot_val.vmin)
        tspan = (shoot_val.xmin, shoot_val.xf)
        prob = ODEProblem(f_vector, u0, tspan)
        sol = solve(prob, VCABM(), abstol = shoot_val.eps, reltol = shoot_val.eps)
        f = sol.u[end]
        
        # NVAR個の合致条件にある偏微分を数値的に計算
        for i in 1:NVAR
            dfdv[i, 1] = (f[i] - f1[i]) / DELV
        end

        # 境界におけるパラメータを格納
        shoot_val.vmin = sav
    end

    function funcx2!(dfdv, f2, f_vector, shoot_val)
        sav = shoot_val.vmax
        shoot_val.vmax += DELV

        u0 = Load2.load2(shoot_val.xmax, shoot_val.vmax, shoot_val.load2_param, shoot_val.load2_val)
        tspan = (shoot_val.xmax, shoot_val.xf)
        prob = ODEProblem(f_vector, u0, tspan)
        sol = solve(prob, VCABM(), abstol = shoot_val.eps, reltol = shoot_val.eps)
        f = sol.u[end]
        
        for i in 1:NVAR
            dfdv[i, 2] = (f2[i] - f[i]) / DELV
        end

        shoot_val.vmax = sav
    end

    load1(xmin, vmin) = let
        y = Vector{Float64}(undef, 2)

        # y[1] = 1.0 + vmin[0] * xmin + 4.0 / 3.0 * xmin * sqrt(xmin) + 0.4 * vmin[0] * xmin * xmin * sqrt(xmin) + 1.0 / 3.0 * xmin * xmin * xmin
        y[1] = (((1.0 / 3.0 * xmin + 0.4 * vmin * sqrt(xmin)) * xmin) + 4.0 / 3.0 * sqrt(xmin) + vmin) * xmin + 1.0
        # y[2] = vmin[0] + 2.0 * sqrt(xmin) + vmin[0] * xmin * sqrt(xmin) + xmin * xmin + 0.15 * vmin[0] * xmin * xmin * sqrt(xmin)
        y[2] = ((0.15 * vmin * sqrt(xmin) + 1.0) * xmin + vmin * sqrt(xmin)) * xmin + 2.0 * sqrt(xmin) + vmin

        return y
    end

    function shootf!(shoot_val) 
        f_vector(u, p, t) = [u[2], u[1] * sqrt(u[1] / t)]

        # 最良の仮の値v1_でx1からxfまで解いていく
        u0 = load1(shoot_val.xmin, shoot_val.vmin)
        tspan = (shoot_val.xmin, shoot_val.xf)
        prob = ODEProblem(f_vector, u0, tspan)
        sol = solve(prob, VCABM(), abstol = shoot_val.eps, reltol = shoot_val.eps)
        f1 = sol.u[end]
        
        # 最良の仮の値v2_でx2からxfまで解いていく
        u0 = Load2.load2(shoot_val.xmax, shoot_val.vmax, shoot_val.load2_param, shoot_val.load2_val)
        tspan = (shoot_val.xmax, shoot_val.xf)
        prob = ODEProblem(f_vector, u0, tspan)
        sol = solve(prob, VCABM(), abstol = shoot_val.eps, reltol = shoot_val.eps)
        f2 = sol.u[end]
        
        dfdv = Matrix{Float64}(undef, NVAR, NVAR)

        if shoot_val.usethread
            # x1で用いる境界条件を変える
            fa = Threads.@spawn funcx1!(dfdv, f1, f_vector, shoot_val)

            # 次にx2で用いる境界条件を変える
            fb = Threads.@spawn funcx2!(dfdv, f2, f_vector, shoot_val)
        
            fetch(fa)
            fetch(fb)
        else
            # x1で用いる境界条件を変える
            funcx1!(dfdv, f1, f_vector, shoot_val)

            # 次にx2で用いる境界条件を変える
            funcx2!(dfdv, f2, f_vector, shoot_val)
        end
        
        f = f1 .- f2
        ff = -1.0 .* f

        solf = dfdv \ ff
        
        # x1の境界でのパラメータ値の増分
        shoot_val.vmin += solf[1]

        # x2の境界でのパラメータ値の増分
        shoot_val.vmax += solf[2]

        size1 = floor(Int, shoot_val.xf / shoot_val.dx) + 1
        size2 = floor(Int, (shoot_val.xmax - shoot_val.xf) / shoot_val.dx) + 1
        
        xarray1 = Vector{Float64}(undef, size1)
        yarray1 = Vector{Float64}(undef, size1)
        xarray2 = Vector{Float64}(undef, size2)
        yarray2 = Vector{Float64}(undef, size2)

        if shoot_val.usethread
            fa = Threads.@spawn solveode_xmintoxf(f_vector, shoot_val)
            fb = Threads.@spawn solveode_xmaxtoxf(f_vector, shoot_val)
            xarray1, yarray1 = fetch(fa)
            xarray2, yarray2 = fetch(fb)
        else
            xarray1, yarray1 = solveode_xmintoxf(f_vector, shoot_val)
            xarray2, yarray2 = solveode_xmaxtoxf(f_vector, shoot_val)
        end

        return createresult(xarray1, xarray2, yarray1, yarray2)
    end

    solveode_xmaxtoxf(f_vector, shoot_val) = let
        # 得られた条件でx2...xfまで微分方程式を解く
        u0 = Load2.load2(shoot_val.xmax, shoot_val.vmax, shoot_val.load2_param, shoot_val.load2_val)
        tspan = (shoot_val.xmax, shoot_val.xf)
        prob = ODEProblem(f_vector, u0, tspan)
        sol = solve(prob, VCABM(), abstol = shoot_val.eps, reltol = shoot_val.eps, saveat = shoot_val.dx)
        a = vcat(sol.u...)
        yarray = deleteat!(a, 2:2:length(a))   # shoot_val.xmax...xfの結果を得る

        return reverse(sol.t), reverse(yarray)
    end

    solveode_xmintoxf(f_vector, shoot_val) = let
        # 得られた条件でx1...dxまで微分方程式を解く
        u0 = load1(shoot_val.xmin, shoot_val.vmin)
        tspan = (shoot_val.xmin, shoot_val.dx)
        prob = ODEProblem(f_vector, u0, tspan)
        sol = solve(prob, VCABM(), abstol = shoot_val.eps, reltol = shoot_val.eps)
        y = sol.u[end]
        tarray = sol.t[1:1]

        # 得られた条件でdx...xfまで微分方程式を解く
        u0 = y
        tspan = (shoot_val.dx, shoot_val.xf)
        prob = ODEProblem(f_vector, u0, tspan)
        sol = solve(prob, VCABM(), abstol = shoot_val.eps, reltol = shoot_val.eps, saveat = shoot_val.dx)
        a = vcat(sol.u...)
        f = deleteat!(a, 2:2:length(a))
        
        y = load1(shoot_val.xmin, shoot_val.vmin)
        yarray = append!(y[1:1], f)
        tarray = append!(tarray, sol.t)

        return tarray, yarray
    end
end
