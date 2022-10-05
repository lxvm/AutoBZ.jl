using Documenter, AutoBZ

makedocs(
    sitename="AutoBZ.jl",
    modules=[AutoBZ, AutoBZ.Applications],
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "man/adaptive_integration.md",
            "man/equispace_integration.md",
            "man/integration_limits.md",
        ],
        "Applications" => [
            "app/integrands.md",
        ],
    ],
)

deploydocs(
    repo = "github.com/lxvm/AutoBZ.jl.git",
)