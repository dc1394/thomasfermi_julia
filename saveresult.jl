module Saveresult
    include("saveresult_module.jl")
    using Printf
    using .Saveresult_module

    const rho_csv_filename = "rho.csv"
    const rhoTilde_csv_filename = "rhoTilde.csv"
    const y_csv_filename = "y.csv"

    function saveresult(data, xarray, yarray)
        alpha = (128.0 / (9.0 * pi ^ 2) * data.Z) ^ (1.0 / 3.0)

        sr_param = Saveresult_module.Saveresult_param(
            alpha,
            32.0 / (9.0 * pi ^ 3) * data.Z * data.Z,
            xarray[2] - xarray[1],
            get_s(xarray, yarray),
            collect(0.0:xarray[2] - xarray[1]:floor(xarray[end] / alpha)),
            xarray,
            yarray,
            data.Z)

        @printf(" Energy = %.15f (Hartree)\n", makeenergy(sr_param))

        saverho(sr_param)
        saverhoTilde(sr_param)
        savey(sr_param)

        @printf("電子密度を%sと%sに，y(x)を%sに書き込みました．\n", rho_csv_filename, rhoTilde_csv_filename, y_csv_filename)
    end

    exactrho(r, sr_param) = let
        return 4.0 * r * r * sr_param.Z ^ 3 * exp(-2.0 * sr_param.Z * r)
    end

    exactrhoTilde(r, sr_param) = let
        return 4.0 * sr_param.Z ^ 3 * exp(-2.0 * sr_param.Z * r)
    end

    makeenergy(sr_param) = let        
        y0_prime = (sr_param.yarray[2] - sr_param.yarray[1]) / (sr_param.xarray[2] - sr_param.xarray[1])
        
        return 3.0 / 7.0 * sr_param.alpha * sr_param.Z ^ (7.0 / 3.0) * y0_prime
    end

    rho(r, sr_param) = let
        return sr_param.s * sr_param.b * (1.0 / sr_param.alpha) ^ 2 * sqrt(r) * y(sr_param, r) * sqrt(y(sr_param, r))
    end

    rhoTilde(r, sr_param) = let
        return sr_param.s * sr_param.b * (y(sr_param, r) / r) * sqrt(y(sr_param, r) / r)
    end

    saverho(sr_param) = let
        open(rho_csv_filename, "w" ) do fp
            for r in sr_param.rarray
                write(fp, @sprintf("%.15f, %.15f, %.15f\n", r, rho(sr_param.alpha * r, sr_param), exactrho(r, sr_param)))
            end
        end
    end

    saverhoTilde(sr_param) = let
        open(rhoTilde_csv_filename, "w" ) do fp
            for r in sr_param.rarray
                write(fp, @sprintf("%.15f, %.15f, %.15f\n", r, rhoTilde(sr_param.alpha * r, sr_param), exactrhoTilde(r, sr_param)))
            end
        end
    end
        
    savey(sr_param) = let
        open(y_csv_filename, "w" ) do fp
            for x in sr_param.xarray
                write(fp, @sprintf("%.15f, %.15f\n", x, y(sr_param, x)))
            end
        end
    end

    y(sr_param, x) = let
        klo = 1
        max = length(sr_param.xarray)
        khi = max

        # 表の中の正しい位置を二分探索で求める
        @inbounds while khi - klo > 1
            k = (khi + klo) >> 1

            if sr_param.xarray[k] > x
                khi = k        
            else 
                klo = k
            end
        end

        # yvec_[i] = f(xvec_[i]), yvec_[i + 1] = f(xvec_[i + 1])の二点を通る直線を代入
        return (sr_param.yarray[khi] - sr_param.yarray[klo]) / (sr_param.xarray[khi] - sr_param.xarray[klo]) * (x - sr_param.xarray[klo]) + sr_param.yarray[klo]
    end

    get_s(xarray, yarray) = let
        sum = 0.0
        len = length(xarray)
        max = len - 2

        # Simpsonの公式によって数値積分する
        @inbounds @simd for i = 1:2:max
            f0 = yarray[i] * sqrt(yarray[i]) * sqrt(xarray[i])
            f1 = yarray[i + 1] * sqrt(yarray[i + 1]) * sqrt(xarray[i + 1])
            f2 = yarray[i + 2] * sqrt(yarray[i + 2]) * sqrt(xarray[i + 2])
        
            sum += (f0 + 4.0 * f1 + f2)
        end

        sum *= (xarray[2] - xarray[1]) / 3.0
    
        return 4.0 * pi / sum
    end
end