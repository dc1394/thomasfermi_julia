module Solve_TF
    include("gausslegendre.jl")
    include("solve_tf_module.jl")
    using LinearAlgebra
    using Match
    using Interpolations
    using .GaussLegendre
    using .Solve_TF_module
    using Printf

    function construct(data, yarray)
        solve_tf_param = Solve_TF_module.Solve_TF_param(data.grid_num + 2, data.gauss_legendre_integ, data.grid_num + 3, data.xmax, data.xmin, yarray[1], yarray[end])
        solve_tf_val = Solve_TF_module.Solve_TF_variables(
            Array{Float64}(undef, solve_tf_param.ELE_TOTAL),
            Array{Float64}(undef, solve_tf_param.ELE_TOTAL, 2, 2),
            SymTridiagonal(Array{Float64}(undef, solve_tf_param.NODE_TOTAL), Array{Float64}(undef, solve_tf_param.NODE_TOTAL - 1)),
            Array{Int64, 2}(undef, solve_tf_param.ELE_TOTAL, 2),
            Array{Float64, 2}(undef, solve_tf_param.ELE_TOTAL, 2),
            Array{Float64}(undef, solve_tf_param.NODE_TOTAL),
            Array{Float64}(undef, solve_tf_param.NODE_TOTAL),
            Array{Float64}(undef, solve_tf_param.ELE_TOTAL, 2),
            Array{Float64}(undef, solve_tf_param.NODE_TOTAL),
            Array{Float64}(undef, solve_tf_param.INTEGTABLENUM),
            Array{Float64}(undef, solve_tf_param.INTEGTABLENUM))
        
        solve_tf_val.x, solve_tf_val.w = GaussLegendre.gausslegendre(solve_tf_param.INTEGTABLENUM)

        return solve_tf_param, solve_tf_val
    end

    phi(solve_tf_param, solve_tf_val, r) = let
        klo = 1
        max = solve_tf_param.NODE_TOTAL
        khi = max

        # 表の中の正しい位置を二分探索で求める
        @inbounds while khi - klo > 1
            k = (khi + klo) >> 1

            if solve_tf_val.node_x_glo[k] > r
                khi = k        
            else 
                klo = k
            end
        end

        # yvec_[i] = f(xvec_[i]), yvec_[i + 1] = f(xvec_[i + 1])の二点を通る直線を代入
        return (solve_tf_val.phi[khi] - solve_tf_val.phi[klo]) / (solve_tf_val.node_x_glo[khi] - solve_tf_val.node_x_glo[klo]) * (r - solve_tf_val.node_x_glo[klo]) + solve_tf_val.phi[klo]
    end

    function solvepoisson!(iter, solve_tf_param, solve_tf_val, xarray, yarray)
        if iter == 0
            # データの生成
            make_data!(solve_tf_param, solve_tf_val, xarray)
        end

        make_element_matrix_and_vector_first(solve_tf_param, solve_tf_val, xarray, yarray)
                
        # 全体行列と全体ベクトルを生成
        tmp_dv, tmp_ev = make_global_matrix_and_vector(solve_tf_param, solve_tf_val)

        # 境界条件処理
        boundary_conditions!(solve_tf_val, solve_tf_param, tmp_dv, tmp_ev)

        # 連立方程式を解く
        solve_tf_val.ug = solve_tf_val.mat_A_glo \ solve_tf_val.vec_b_glo

        # openしてブロック内で書き込み
        open( "res.txt", "w" ) do fp
            for item in solve_tf_val.ug
                write( fp, @sprintf("%.14f\n", item))
            end
        end
        return solve_tf_val.ug
    end
    
    function boundary_conditions!(solve_tf_val, solve_tf_param, tmp_dv, tmp_ev)
        a = 0.5 * 1.0E-10 #solve_tf_param.Y0
        tmp_dv[1] = 1.0
        solve_tf_val.vec_b_glo[1] = a
        solve_tf_val.vec_b_glo[2] -= a * tmp_ev[1]
        tmp_ev[1] = 0.0
    
        b = 5000.0#solve_tf_param.YMAX
        tmp_dv[solve_tf_param.NODE_TOTAL] = 1.0
        solve_tf_val.vec_b_glo[solve_tf_param.NODE_TOTAL] = b
        solve_tf_val.vec_b_glo[solve_tf_param.NODE_TOTAL - 1] -= b * tmp_ev[solve_tf_param.NODE_TOTAL - 1]
        tmp_ev[solve_tf_param.NODE_TOTAL - 1] = 0.0

        open("tmp_dv.txt", "w" ) do fp
            for item in tmp_dv
                println(fp, @sprintf("%.14f", item))
            end
        end

        solve_tf_val.mat_A_glo = SymTridiagonal(tmp_dv, tmp_ev)
    end

    function make_beta(xarray, yarray)
        len = length(xarray)
        beta_array = Array{Float64}(undef, len)

        for i in 1:len
            beta_array[i] = yarray[i] * sqrt(yarray[i] / xarray[i])
        end

        return beta_array
    end

    function make_data!(solve_tf_param, solve_tf_val, xarray)
        # Global節点のx座標を定義(X_MIN～X_MAX）
        solve_tf_val.node_x_glo = xarray

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

    function make_element_matrix_and_vector(solve_tf_param, solve_tf_val)
        # 要素行列とLocal節点ベクトルの各成分を計算
        for e = 1:solve_tf_param.ELE_TOTAL
            for i = 1:2
                for j = 1:2
                    solve_tf_val.mat_A_ele[e, i, j] = (-1) ^ i * (-1) ^ j / solve_tf_val.length[e]
                end

                solve_tf_val.vec_b_ele[e, i] =
                    @match i begin
                        1 => GaussLegendre.gl_integ(r -> r * rho(solve_tf_param, solve_tf_val, r) * (solve_tf_val.node_x_ele[e, 2] - r) / solve_tf_val.length[e],
                                                     solve_tf_val.node_x_ele[e, 1],
                                                     solve_tf_val.node_x_ele[e, 2],
                                                     solve_tf_val)
                        
                        2 => GaussLegendre.gl_integ(r -> r * rho(solve_tf_param, solve_tf_val, r) * (r - solve_tf_val.node_x_ele[e, 1]) / solve_tf_val.length[e],
                                                     solve_tf_val.node_x_ele[e, 1],
                                                     solve_tf_val.node_x_ele[e, 2],
                                                     solve_tf_val)
                    
                        _ => 0.0
                    end
            end
        end
    end

    function make_element_matrix_and_vector_first(solve_tf_param, solve_tf_val, xarray, yarray)
        beta_array = make_beta(xarray, yarray)
        yitp = interpolate((xarray,), beta_array, Gridded(Constant()))

        # 要素行列とLocal節点ベクトルの各成分を計算
        for e = 1:solve_tf_param.ELE_TOTAL
            for i = 1:2
                for j = 1:2
                    solve_tf_val.mat_A_ele[e, i, j] = (-1) ^ i * (-1) ^ j / solve_tf_val.length[e]
                end

                solve_tf_val.vec_b_ele[e, i] =
                    @match i begin
                        1 => GaussLegendre.gl_integ(x -> (solve_tf_val.node_x_ele[e, 2] - x) / solve_tf_val.length[e],
                                                     solve_tf_val.node_x_ele[e, 1],
                                                     solve_tf_val.node_x_ele[e, 2],
                                                     solve_tf_val)
                        
                        2 => GaussLegendre.gl_integ(x -> (x - solve_tf_val.node_x_ele[e, 1]) / solve_tf_val.length[e],
                                                     solve_tf_val.node_x_ele[e, 1],
                                                     solve_tf_val.node_x_ele[e, 2],
                                                     solve_tf_val)
                    
                        _ => 0.0
                    end
            end
        end
    end

    function make_global_matrix_and_vector(solve_tf_param, solve_tf_val)
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

    rho(solve_tf_param, solve_tf_val, r) = let
        tmp = phi(solve_tf_param, solve_tf_val, r)
        return tmp * tmp
    end
end
