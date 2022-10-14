using Documenter, AutoBZ

makedocs(
    sitename="AutoBZ.jl",
    modules=[AutoBZ, AutoBZ.Applications],
    pages = [
        "Home" => "pages/index.md",
        "Manual" => [
            "pages/man/adaptive_integration.md",
            "pages/man/equispace_integration.md",
            "pages/man/integration_limits.md",
            ],
        "Applications" => [
            "pages/app/integrands.md",
            "pages/app/interfaces.md",
        ],
        "Demos" => "pages/demo.md",
    ],
)

deploydocs(
    repo = "github.com/lxvm/AutoBZ.jl.git",
)