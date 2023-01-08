using Documenter, AutoBZ

Documenter.HTML(
    mathengine = MathJax3(Dict(
        :loader => Dict("load" => ["[tex]/physics"]),
        :tex => Dict(
            "inlineMath" => [["\$","\$"], ["\\(","\\)"]],
            "tags" => "ams",
            "packages" => ["base", "ams", "autoload", "physics"],
        ),
    )),
)

makedocs(
    sitename="AutoBZ.jl",
    modules=[AutoBZ, AutoBZ.Applications],
    pages = [
        "Home" => "index.md",
        "Core" => [
            "pages/core/adaptive_integration.md",
            "pages/core/equispace_integration.md",
            "pages/core/integration_limits.md",
            ],
        "Manual" => [
            "pages/man/fourier.md",
            "pages/man/integrands.md",
            "pages/man/interfaces.md",
            "pages/man/self_energy.md",
        ],
        "Jobs" => "pages/jobs.md",
        "AdaptChebInterp" => "pages/adaptinterp.md",
        "EquiBaryInterp" => "pages/equiinterp.md",
        "Demos" => "pages/demo.md",
        "Workflow" => "pages/workflow.md",
    ],
)

deploydocs(
    repo = "github.com/lxvm/AutoBZ.jl.git",
)