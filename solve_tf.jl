module Solve_TF
    include("gausslegendre.jl")
    include("solve_tf_module.jl")
    using LinearAlgebra
    using Match
    using MKL
    using Interpolations
    using .GaussLegendre
    using .Solve_TF_module

    const BETA0 = 1000.0

    function construct(data, xarray, yarray)
        solve_tf_param = Solve_TF_module.Solve_TF_param(data.grid_num, data.gauss_legendre_integ_num, data.grid_num + 1, data.xmax, data.xmin, yarray[1], yarray[end])
        solve_tf_val = Solve_TF_module.Solve_TF_variables(
            Array{Float64}(undef, solve_tf_param.ELE_TOTAL),
            Array{Float64}(undef, solve_tf_param.ELE_TOTAL, 2, 2),
            SymTridiagonal(Array{Float64}(undef, solve_tf_param.NODE_TOTAL), Array{Float64}(undef, solve_tf_param.NODE_TOTAL - 1)),
            Array{Int64, 2}(undef, solve_tf_param.ELE_TOTAL, 2),
            Array{Float64, 2}(undef, solve_tf_param.ELE_TOTAL, 2),
            xarray,
            Array{Float64}(undef, solve_tf_param.NODE_TOTAL),
            Array{Float64}(undef, solve_tf_param.ELE_TOTAL, 2),
            Array{Float64}(undef, solve_tf_param.NODE_TOTAL),
            Array{Float64}(undef, solve_tf_param.INTEGTABLENUM),
            Array{Float64}(undef, solve_tf_param.INTEGTABLENUM))
        
        solve_tf_val.x, solve_tf_val.w = GaussLegendre.gausslegendre(solve_tf_param.INTEGTABLENUM)

        return solve_tf_param, solve_tf_val
    end

    function solvetf!(iter, solve_tf_param, solve_tf_val, yarray)
        if iter == 0
            # データの生成
            make_data!(solve_tf_param, solve_tf_val)
        end

        make_element_matrix_and_vector!(solve_tf_param, solve_tf_val, yarray)
                
        # 全体行列と全体ベクトルを生成
        tmp_dv, tmp_ev = make_global_matrix_and_vector!(solve_tf_param, solve_tf_val)

        # 境界条件処理
        boundary_conditions!(solve_tf_val, solve_tf_param, tmp_dv, tmp_ev)

        # 連立方程式を解く
        solve_tf_val.ug = solve_tf_val.mat_A_glo \ solve_tf_val.vec_b_glo

        return solve_tf_val.ug
    end
    
    function boundary_conditions!(solve_tf_val, solve_tf_param, tmp_dv, tmp_ev)
        a = solve_tf_param.Y0
        tmp_dv[1] = 1.0
        solve_tf_val.vec_b_glo[1] = a
        solve_tf_val.vec_b_glo[2] -= a * tmp_ev[1]
        tmp_ev[1] = 0.0
    
        b = solve_tf_param.YMAX
        tmp_dv[solve_tf_param.NODE_TOTAL] = 1.0
        solve_tf_val.vec_b_glo[solve_tf_param.NODE_TOTAL] = b
        solve_tf_val.vec_b_glo[solve_tf_param.NODE_TOTAL - 1] -= b * tmp_ev[solve_tf_param.NODE_TOTAL - 1]
        tmp_ev[solve_tf_param.NODE_TOTAL - 1] = 0.0

        solve_tf_val.mat_A_glo = SymTridiagonal(tmp_dv, tmp_ev)
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

    function make_element_matrix_and_vector!(solve_tf_param, solve_tf_val, yarray)
        beta_array = make_beta(solve_tf_val, yarray)
        yitp = interpolate((solve_tf_val.node_x_glo,), beta_array, Gridded(Linear()))

        # 要素行列とLocal節点ベクトルの各成分を計算
        Threads.@threads for e = 1:solve_tf_param.ELE_TOTAL
            for i = 1:2
                for j = 1:2
                    solve_tf_val.mat_A_ele[e, i, j] = (-1) ^ i * (-1) ^ j / solve_tf_val.length[e]
                end

                solve_tf_val.vec_b_ele[e, i] =
                    @match i begin
                        1 => GaussLegendre.gl_integ(x -> -yitp(x) * (solve_tf_val.node_x_ele[e, 2] - x) / solve_tf_val.length[e],
                                                    solve_tf_val.node_x_ele[e, 1],
                                                    solve_tf_val.node_x_ele[e, 2],
                                                    solve_tf_val)
                        
                        2 => GaussLegendre.gl_integ(x -> -yitp(x) * (x - solve_tf_val.node_x_ele[e, 1]) / solve_tf_val.length[e],
                                                    solve_tf_val.node_x_ele[e, 1],
                                                    solve_tf_val.node_x_ele[e, 2],
                                                    solve_tf_val)
                    
                        _ => 0.0
                    end
            end
        end
    end

    function make_global_matrix_and_vector!(solve_tf_param, solve_tf_val)
        tmp_dv = zeros(solve_tf_param.NODE_TOTAL)
        tmp_ev = zeros(solve_tf_param.NODE_TOTAL - 1)

        solve_tf_val.vec_b_glo = zeros(solve_tf_param.NODE_TOTAL)

        # 全体行列と全体ベクトルを生成
        for e = 1:solve_tf_param.ELE_TOTAL
            for i = 1:2
                for j = 1:2
                    if solve_tf_val.node_num_seg[e, i] == solve_tf_val.node_num_seg[e, j]
                        tmp_dv[solve_tf_val.node_num_seg[e, i]] += solve_tf_val.mat_A_ele[e, i, j]
                    elseif solve_tf_val.node_num_seg[e, i] + 1 == solve_tf_val.node_num_seg[e, j]
                        tmp_ev[solve_tf_val.node_num_seg[e, i]] += solve_tf_val.mat_A_ele[e, i, j]
                    end
                end
                
                solve_tf_val.vec_b_glo[solve_tf_val.node_num_seg[e, i]] += solve_tf_val.vec_b_ele[e, i]
            end
        end

        return tmp_dv, tmp_ev
    end
end
