module Shoot
    include("load2.jl")
    include("load2_module.jl")
    include("readinputfile.jl")
    include("shoot_module.jl")
    using DifferentialEquations
    using LinearAlgebra
    using MKL
    using .Load2
    using .Shoot_module

    const DELV = 1.0E-7
    const NVAR = 2
    const VMIN = -1.588076779

    function construct(dx, data)
        load2_param, load2_val = Load2.construct()
        shoot_val = Shoot_module.Shoot_module_variables(
            dx,
            data.eps,
            data.grid_num,
            VMIN,
            Load2.make_vmax(data.xmax, load2_param, load2_val),
            data.xmin,
            data.matching_point,
            data.xmax,
            Shoot_module.Load2_module.Load2_module_param(load2_param.LAMBDA, load2_param.THRESHOLD, load2_param.K),
            Shoot_module.Load2_module.Load2_module_variables(load2_val.x, load2_val.yitp))
    
        return shoot_val
    end

    createresult(res1, res2, xarray1, xarray2, shoot_val) = let
        xarray = append!(xarray1[2:end], xarray2[2:end])
        res1[end] = (res1[end] + res2[1]) / 2.0
        yarray = append!(res1[2:end], res2[2:end])

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
        sol = solve(prob, abstol = shoot_val.eps, reltol = shoot_val.eps)
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
        sol = solve(prob, abstol = shoot_val.eps, reltol = shoot_val.eps)
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

    function shootf!(xmin, xmax, xf, shoot_val) 
        f_vector(u, p, t) = [u[2], u[1] * sqrt(u[1] / t)]

        # 最良の仮の値v1_でx1からxfまで解いていく
        u0 = load1(shoot_val.xmin, shoot_val.vmin)
        tspan = (shoot_val.xmin, shoot_val.xf)
        prob = ODEProblem(f_vector, u0, tspan)
        sol = solve(prob, abstol = shoot_val.eps, reltol = shoot_val.eps)
        f1 = sol.u[end]
        
        # 最良の仮の値v2_でx2からxfまで解いていく
        u0 = Load2.load2(shoot_val.xmax, shoot_val.vmax, shoot_val.load2_param, shoot_val.load2_val)
        tspan = (shoot_val.xmax, shoot_val.xf)
        prob = ODEProblem(f_vector, u0, tspan)
        sol = solve(prob, abstol = shoot_val.eps, reltol = shoot_val.eps)
        f2 = sol.u[end]
        
        dfdv = Matrix{Float64}(undef, NVAR, NVAR)

        # x1で用いる境界条件を変える
        fa = @async funcx1!(dfdv, f1, f_vector, shoot_val)

        # 次にx2で用いる境界条件を変える
        fb = @async funcx2!(dfdv, f2, f_vector, shoot_val)

        wait(fa)
        wait(fb)
        
        f = f1 .- f2
        ff = -1.0 .* f

        solf = dfdv \ ff
        
        # x1の境界でのパラメータ値の増分
        shoot_val.vmin += solf[1]

        # x2の境界でのパラメータ値の増分
        shoot_val.vmax += solf[2]

        res1, xarray1 = solveode_xmintoxf(f_vector, shoot_val)

        res2, xarray2 = solveode_xmaxtoxf(f_vector, shoot_val)

        return createresult(res1, res2, xarray1, xarray2, shoot_val)
    end

    solveode_xmaxtoxf(f_vector, shoot_val) = let
        # 得られた条件でx2...xfまで微分方程式を解く
        u0 = Load2.load2(shoot_val.xmax, shoot_val.vmax, shoot_val.load2_param, shoot_val.load2_val)
        tspan = (shoot_val.xmax, shoot_val.xf)
        prob = ODEProblem(f_vector, u0, tspan)
        sol = solve(prob, abstol = shoot_val.eps, reltol = shoot_val.eps, saveat = shoot_val.dx)
        a = vcat(sol.u...)
        res = deleteat!(a, 2:2:length(a))   # shoot_val.xmax...xfの結果を得る

        return reverse(res), reverse(sol.t)
    end

    solveode_xmintoxf(f_vector, shoot_val) = let
        # 得られた条件でx1...dxまで微分方程式を解く
        u0 = load1(shoot_val.xmin, shoot_val.vmin)
        tspan = (shoot_val.xmin, shoot_val.dx)
        prob = ODEProblem(f_vector, u0, tspan)
        sol = solve(prob, abstol = shoot_val.eps, reltol = shoot_val.eps)
        y = sol.u[end]
        tarray = sol.t[1:1]

        # 得られた条件でdx...xfまで微分方程式を解く
        u0 = y
        tspan = (shoot_val.dx, shoot_val.xf)
        prob = ODEProblem(f_vector, u0, tspan)
        sol = solve(prob, abstol = shoot_val.eps, reltol = shoot_val.eps, saveat = shoot_val.dx)
        a = vcat(sol.u...)
        f = deleteat!(a, 2:2:length(a))
        
        y = collect(load1(shoot_val.xmin, shoot_val.vmin))
        res = append!(y[1:1], f)
        tarray = append!(tarray, sol.t)

        return res, tarray
    end
end
