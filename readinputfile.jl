module Readinputfile
    include("data_module.jl")
    include("readinputfile_module.jl")
    using Base.Unicode
    using Printf
    using .Readinputfile_module

    function construct(inputfilename)
        lines = nothing
        try
            lines = readlines(inputfilename)
        catch e
            @printf "%s。プログラムを終了します。\n" sprint(showerror, e)
            exit(-1)
        end

        data = Readinputfile_module.Data_module.Data_val(
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing)

        return Readinputfile_module.Readinputfile_variables(data, lines, 1)
    end

    errormessage(s) = let
        @printf "インプットファイルに%sの行が見つかりませんでした。\n" s
    end

    errormessage(line, s1, s2) = let
        @printf "インプットファイルの[%s]の行が正しくありません。\n" s1
        @printf "%d行目, 未知のトークン:%s\n" line s2
    end

    gettokens(rif_val, word) = let
        line = rif_val.lines[rif_val.lineindex]

        # 読み込んだ行が空、あるいはコメント行でないなら
        if !isempty(line) && line[1] != '#'
            # トークン分割
            tokens = map(line -> lowercase(line), split(line))
            w = lowercase(word)
            
            if tokens[1] != w
                errormessage(rif_val.lineindex, word, tokens[1])
                return -1, nothing
            end
            
            return 0, tokens
        else
            return 1, nothing
        end
    end

    function readchemical_number!(rif_val)
        # 原子の種類を読み込む
        chemnum = readdata!(rif_val, Data_module.CHEMICAL_NUMBER)
        if chemnum == nothing
            throw(ErrorException("インプットファイルが異常です"))
        end

        rif_val.data.Z = parse(Float64, chemnum)
    end

    function readdata!(rif_val, word)
        while true
            ret_val, tokens = gettokens(rif_val, word)

            if ret_val == -1
                return nothing
            elseif ret_val == 0
                # 読み込んだトークンの数がもし2個以外だったら
                if length(tokens) != 2
                    @printf "インプットファイル%d行目の、[%s]の行が正しくありません。\n" rif_val.lineindex word
                    return nothing
                end

                rif_val.lineindex += 1
                return tokens[2]
            end

            rif_val.lineindex += 1
        end
    end

    function readdata!(default_value, rif_val, word)
        # グリッドを読み込む
        while true
            ret_val, tokens = gettokens(rif_val, word)

            if ret_val == -1
                return nothing
            elseif ret_val == 0
                rif_val.lineindex += 1
                   
                len = length(tokens)
                # 読み込んだトークンの数をはかる
                if len == 1
                    # デフォルト値を返す
                    return default_value
                elseif len == 2 
                    if tokens[2] == "default"
                        # デフォルト値を返す
                        return default_value
                    else
                        val = nothing
                        if typeof(default_value) != String
                            val = tryparse(typeof(default_value), tokens[2])
                        else
                            val = tokens[2]
                        end

                        if val != nothing
                            return val
                        else
                            errormessage(rif_val.lineindex - 1, word, tokens[2])
                            return nothing
                        end
                    end
                else
                    str = tokens[2]

                    if str == "default" || str[1] == '#'
                        return default_value
                    elseif tokens[3][1] != '#'
                        errormessage(rif_val.lineindex - 1, word, tokens[3]);
                        return nothing
                    else
                        val = nothing
                        if typeof(default_value) != String
                            val = tryparse(typeof(default_value), tokens[2])
                        else
                            val = tokens[2]
                        end
                        
                        if val != nothing
                            return val
                        else
                            errormessage(rif_val.lineindex - 1, word, str)
                            return nothing
                        end
                    end
                end
            end

            rif_val.lineindex += 1
        end
    end

    function readdataauto!(rif_val, word)
        while true
            ret_val, tokens = gettokens(rif_val, word)

            if ret_val == -1
                return nothing
            elseif ret_val == 0
                rif_val.lineindex += 1

                # 読み込んだトークンの数をはかる
                len = length(tokens)
                
                if len == 1
                    return nothing
                elseif len == 2
                    return (tokens[2] == "default" || tokens[2] == "auto") ? "" : tokens[2]
                else
                    if tokens[2] == "default" || tokens[2] == "auto" || tokens[2][0] == '#'
                        # デフォルト値を返す
                        return ""
                    elseif tokens[3][0] != '#'
                        errormessage(rif_val.lineindex - 1, word, tokens[3])

                        # エラー
                        return nothing
                    end

                    return tokens[2]
                end
            end

            rif_val.lineindex += 1
        end
    end

    function readfile(rif_val)
        # 対象の原子番号を読み込む
        readchemical_number!(rif_val)
                
        # グリッドの最小値を読み込む
        rif_val.data.xmin = readvalue!(Data_module.XMIN_DEFAULT, rif_val, "grid.xmin")

        # グリッドの最大値を読み込む
        rif_val.data.xmax = readvalue!(Data_module.XMAX_DEFAULT, rif_val, "grid.xmax")
 
        # グリッドのサイズを読み込む
        rif_val.data.grid_num = readvalue!(Data_module.GRID_NUM_DEFAULT, rif_val, "grid.num")
 
        # 許容誤差を読み込む
        rif_val.data.eps = readvalue!(Data_module.EPS_DEFAULT, rif_val, "eps")
 
        # マッチングポイントを読み込む
        rif_val.data.matching_point = readvalue!(Data_module.MATCHING_POINT_DEFAULT, rif_val, "matching.point")

        # 積分に使うGauss-Legendreの分点を読み込む
        rif_val.data.gauss_legendre_integ_num  = readvalue!(Data_module.GAUSS_LEGENDRE_INTEG_DEFAULT, rif_val, "gauss.legendre.integ")

        # SCFの最大ループ回数を読み込む
        rif_val.data.iteration_maxiter = readvalue!(Data_module.ITERATION_MAXITER_DEFAULT, rif_val, "iteration.maxIter")

        # SCFの一次混合の重みを読み込む
        readiterationmixingweight!(rif_val)

        # SCFの収束判定条件の値を読み込む
        rif_val.data.iteration_criterion = readvalue!(Data_module.ITERATION_CRITERION_DEFAULT, rif_val, "iteration.criterion")

        return rif_val.data
    end
 
    function readiterationmixingweight!(rif_val)
        rif_val.data.iteration_mixing_weight = readvalue!(Data_module.ITERATION_MIXING_WEIGHT_DEFAULT, rif_val, "iteration.Mixing.Weight")
        if rif_val.data.iteration_mixing_weight <= 0.0 || rif_val.data.iteration_mixing_weight > 1.0
            @printf "インプットファイルの[scf.Mixing.Weight]の行が正しくありません。\n"
            throw(ErrorException("インプットファイルが異常です"))
        end
    end
 
    function readvalue!(default_value, rif_val, word)
        val = readdata!(default_value, rif_val, word)
        if val != nothing
            return val
        else
            throw(ErrorException("インプットファイルが異常です"))
        end
    end
end
