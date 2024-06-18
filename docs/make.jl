push!(LOAD_PATH, "../src")
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
        "Getting started" => "pages/workflow.md",
        "Tutorials" => [
            "pages/demo/dos.md",
            "pages/demo/density.md",
            "pages/demo/oc.md",
        ],
        "Manual" => [
            "pages/man/fourier.md",
            "pages/app/self_energy.md",
            "pages/app/integrands.md",
            "pages/app/interfaces.md",
            "pages/man/internal.md",
        ],
    ],
)

deploydocs(
    repo = "github.com/lxvm/AutoBZ.jl.git",
)
