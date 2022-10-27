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
        "Manual" => [
            "pages/man/adaptive_integration.md",
            "pages/man/equispace_integration.md",
            "pages/man/integration_limits.md",
            ],
        "Applications" => [
            "pages/app/fourier.md",
            "pages/app/integrands.md",
            "pages/app/interfaces.md",
        ],
        "Demos" => "pages/demo.md",
    ],
)

deploydocs(
    repo = "github.com/lxvm/AutoBZ.jl.git",
)