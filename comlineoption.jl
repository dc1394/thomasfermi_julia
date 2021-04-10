module Comlineoption
    include("comlineoption_module.jl")
    using ArgParse
    using .Comlineoption_module

    const DEFINPNAME = "input.inp"

    function construct(args)
        parsed_args = parse_commandline(args)
        opt = Comlineoption_module.comlineoption_param(parsed_args["inputfile"])

        return opt
    end

    function parse_commandline(args)
        s = ArgParseSettings()

        @add_arg_table! s begin
            "--inputfile", "-I"
                help = "インプットファイル名を指定します"
                arg_type = String
                default = DEFINPNAME
        end

        return parse_args(args, s)
    end
end