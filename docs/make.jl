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
    modules=[AutoBZ],
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "pages/man/adaptive_integration.md",
            "pages/man/equispace_integration.md",
            "pages/man/integration_limits.md",
            "pages/man/fourier.md",
            "pages/man/integrands.md",
            "pages/man/integrators.md",
            ],
        "Jobs" => [
            "pages/app/jobs.md",
            "pages/app/self_energy.md",
            "pages/app/integrands.md",
            "pages/app/interfaces.md",
            "pages/app/fourier3d.md",
        ],
        "AdaptChebInterp" => "pages/adaptinterp.md",
        "EquiBaryInterp" => "pages/equiinterp.md",
        "Demos" => "pages/demo.md",
        "Workflow" => "pages/workflow.md",
    ],
)

deploydocs(
    repo = "github.com/lxvm/AutoBZ.jl.git",
)