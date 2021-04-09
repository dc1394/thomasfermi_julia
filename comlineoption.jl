module Comlineoption
    include("comlineoption_module.jl")
    using ArgParse
    using Printf
    using .Comlineoption_module

    const DEFINPNAME = "input.inp"

    function construct(args)
        parsed_args = parse_commandline(args)
        opt = Comlineoption_module.comlineoption_param(parsed_args["inputfile"], parsed_args["usethread"])

        return opt
    end

    function parse_commandline(args)
        s = ArgParseSettings()

        @add_arg_table! s begin
            "--inputfile", "-I"
                help = "インプットファイル名を指定します"
                arg_type = String
                default = DEFINPNAME
            "--usethread", "-T"
                help = "スレッド並列を行うかどうかを指定します"
                arg_type = Bool
                default = true
        end

        return parse_args(args, s)
    end
end