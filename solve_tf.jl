module Solve_TF
    include("gausslegendre.jl")
    include("solve_tf_module.jl")
    using LinearAlgebra
    using Match
    using MKL
    using .GaussLegendre
    using .Solve_TF_module

    const BETA0 = 1000.0

    beta(x, xarray, beta_array) = let
        klo = 1
        max = length(xarray)
        khi = max

        # 表の中の正しい位置を二分探索で求める
        @inbounds while khi - klo > 1
            k = (khi + klo) >> 1

            if xarray[k] > x
                khi = k        
            else 
                klo = k
            end
        end

        # yvec_[i] = f(xvec_[i]), yvec_[i + 1] = f(xvec_[i + 1])の二点を通る直線を代入
        return (beta_array[khi] - beta_array[klo]) / (xarray[khi] - xarray[klo]) * (x - xarray[klo]) + beta_array[klo]
    end

    function boundary_conditions!(solve_tf_val, solve_tf_param)
        a = solve_tf_param.Y0
        solve_tf_val.vec_b_glo[1] = a
        solve_tf_val.vec_b_glo[2] -= a * solve_tf_val.tmp[1]
        
        b = solve_tf_param.YMAX
        solve_tf_val.vec_b_glo[solve_tf_param.NODE_TOTAL] = b
        solve_tf_val.vec_b_glo[solve_tf_param.NODE_TOTAL - 1] -= b * solve_tf_val.tmp[2]
    end

    function construct(data, xarray, yarray)
        solve_tf_param = Solve_TF_module.Solve_TF_param(
            data.grid_num,
            data.gauss_legendre_integ_num,
            data.grid_num + 1,
            data.usethread,
            data.xmax,
            data.xmin,
            yarray[1],
            yarray[end])

        solve_tf_val = Solve_TF_module.Solve_TF_variables(
            Array{Float64}(undef, solve_tf_param.ELE_TOTAL),
            SymTridiagonal(Array{Float64}(undef, solve_tf_param.NODE_TOTAL), Array{Float64}(undef, solve_tf_param.NODE_TOTAL - 1)),
            Array{Int64, 2}(undef, solve_tf_param.ELE_TOTAL, 2),
            Array{Float64, 2}(undef, solve_tf_param.ELE_TOTAL, 2),
            xarray,
            Array{Float64}(undef, 2),
            Array{Float64}(undef, solve_tf_param.NODE_TOTAL),
            Array{Float64}(undef, solve_tf_param.ELE_TOTAL, 2),
            Array{Float64}(undef, solve_tf_param.NODE_TOTAL),
            Array{Float64}(undef, solve_tf_param.INTEGTABLENUM),
            Array{Float64}(undef, solve_tf_param.INTEGTABLENUM))
        
        solve_tf_val.x, solve_tf_val.w = GaussLegendre.gausslegendre(solve_tf_param.INTEGTABLENUM)

        make_data!(solve_tf_param, solve_tf_val)
        make_global_matrix!(solve_tf_param, solve_tf_val)

        return solve_tf_param, solve_tf_val
    end
    
    make_beta(solve_tf_val, yarray) = let
        beta_array = Vector{Float64}(undef, length(solve_tf_val.node_x_glo))
        beta_array[1] = BETA0
        for i in 2:length(solve_tf_val.node_x_glo)
            beta_array[i] = yarray[i] * sqrt(yarray[i] / solve_tf_val.node_x_glo[i])
        end

        return beta_array
    end

    function make_data!(solve_tf_param, solve_tf_val)
        @inbounds for e = 1:solve_tf_param.ELE_TOTAL
            solve_tf_val.node_num_seg[e, 1] = e
            solve_tf_val.node_num_seg[e, 2] = e + 1
        end
            
        @inbounds for e = 1:solve_tf_param.ELE_TOTAL
            for i = 1:2
                solve_tf_val.node_x_ele[e, i] = solve_tf_val.node_x_glo[solve_tf_val.node_num_seg[e, i]]
            end
        end

        # 各線分要素の長さを計算
        @inbounds for e = 1:solve_tf_param.ELE_TOTAL
            solve_tf_val.length[e] = abs(solve_tf_val.node_x_ele[e, 2] - solve_tf_val.node_x_ele[e, 1])
        end
    end

    function make_element_vector!(solve_tf_param, solve_tf_val, yarray)
        beta_array = make_beta(solve_tf_val, yarray)
        
        if solve_tf_param.USETHREAD
            # Local節点ベクトルの各成分を計算
            Threads.@threads for e = 1:solve_tf_param.ELE_TOTAL
                for i = 1:2
                    solve_tf_val.vec_b_ele[e, i] =
                        @match i begin
                            1 => GaussLegendre.gl_integ(x -> -beta(x, solve_tf_val.node_x_glo, beta_array) * (solve_tf_val.node_x_ele[e, 2] - x) / solve_tf_val.length[e],
                                                        solve_tf_val.node_x_ele[e, 1],
                                                        solve_tf_val.node_x_ele[e, 2],
                                                        solve_tf_val)
                        
                            2 => GaussLegendre.gl_integ(x -> -beta(x, solve_tf_val.node_x_glo, beta_array) * (x - solve_tf_val.node_x_ele[e, 1]) / solve_tf_val.length[e],
                                                        solve_tf_val.node_x_ele[e, 1],
                                                        solve_tf_val.node_x_ele[e, 2],
                                                        solve_tf_val)
                    
                            _ => 0.0
                        end
                end
            end
        else
            # Local節点ベクトルの各成分を計算
            @inbounds for e = 1:solve_tf_param.ELE_TOTAL
                for i = 1:2
                    solve_tf_val.vec_b_ele[e, i] =
                        @match i begin
                            1 => GaussLegendre.gl_integ(x -> -beta(x, solve_tf_val.node_x_glo, beta_array) * (solve_tf_val.node_x_ele[e, 2] - x) / solve_tf_val.length[e],
                                                        solve_tf_val.node_x_ele[e, 1],
                                                        solve_tf_val.node_x_ele[e, 2],
                                                        solve_tf_val)
                        
                            2 => GaussLegendre.gl_integ(x -> -beta(x, solve_tf_val.node_x_glo, beta_array) * (x - solve_tf_val.node_x_ele[e, 1]) / solve_tf_val.length[e],
                                                        solve_tf_val.node_x_ele[e, 1],
                                                        solve_tf_val.node_x_ele[e, 2],
                                                        solve_tf_val)
                    
                            _ => 0.0
                        end
                end
            end
        end
    end

    function make_global_matrix!(solve_tf_param, solve_tf_val)
        # 要素行列
        mat_A_ele = Array{Float64}(undef, solve_tf_param.ELE_TOTAL, 2, 2)

        # 要素行列の各成分を計算
        @inbounds for e = 1:solve_tf_param.ELE_TOTAL
            for i = 1:2
                for j = 1:2
                    mat_A_ele[e, i, j] = (-1) ^ i * (-1) ^ j / solve_tf_val.length[e]
                end
            end
        end

        tmp_dv = zeros(solve_tf_param.NODE_TOTAL)
        tmp_ev = zeros(solve_tf_param.NODE_TOTAL - 1)

        # 全体行列を生成
        @inbounds for e = 1:solve_tf_param.ELE_TOTAL
            for i = 1:2
                for j = 1:2
                    if solve_tf_val.node_num_seg[e, i] == solve_tf_val.node_num_seg[e, j]
                        tmp_dv[solve_tf_val.node_num_seg[e, i]] += mat_A_ele[e, i, j]
                    elseif solve_tf_val.node_num_seg[e, i] + 1 == solve_tf_val.node_num_seg[e, j]
                        tmp_ev[solve_tf_val.node_num_seg[e, i]] += mat_A_ele[e, i, j]
                    end
                end
            end
        end

        # 全体ベクトルの境界条件処理のために保管しておく
        solve_tf_val.tmp[1] = tmp_ev[1] 
        solve_tf_val.tmp[2] = tmp_ev[solve_tf_param.NODE_TOTAL - 1]

        # 全体行列の境界条件処理
        tmp_dv[1] = 1.0
        tmp_ev[1] = 0.0
        tmp_dv[solve_tf_param.NODE_TOTAL] = 1.0
        tmp_ev[solve_tf_param.NODE_TOTAL - 1] = 0.0

        solve_tf_val.mat_A_glo = SymTridiagonal(tmp_dv, tmp_ev)
    end

    function make_global_vector!(solve_tf_param, solve_tf_val)
        solve_tf_val.vec_b_glo = zeros(solve_tf_param.NODE_TOTAL)

        # 全体行列と全体ベクトルを生成
        @inbounds for e = 1:solve_tf_param.ELE_TOTAL
            for i = 1:2
                solve_tf_val.vec_b_glo[solve_tf_val.node_num_seg[e, i]] += solve_tf_val.vec_b_ele[e, i]
            end
        end
    end
    
    function solvetf!(solve_tf_param, solve_tf_val, yarray)
        # Local節点ベクトルを生成
        make_element_vector!(solve_tf_param, solve_tf_val, yarray)
                
        # 全体ベクトルを生成
        make_global_vector!(solve_tf_param, solve_tf_val)

        # 境界条件処理
        boundary_conditions!(solve_tf_val, solve_tf_param)

        # 連立方程式を解く
        solve_tf_val.ug = solve_tf_val.mat_A_glo \ solve_tf_val.vec_b_glo

        return solve_tf_val.ug
    end
end