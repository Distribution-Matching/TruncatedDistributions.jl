using Documenter, TruncatedDistributions

DocMeta.setdocmeta!(TruncatedDistributions, :DocTestSetup,
                    :(using TruncatedDistributions); recursive = true)

makedocs(
    sitename = "TruncatedDistributions.jl",
    modules  = [TruncatedDistributions],
    authors  = "Yoni Nazarathy and contributors",
    repo     = "https://github.com/Distribution-Matching/TruncatedDistributions.jl/blob/{commit}{path}#{line}",
    format   = Documenter.HTML(
        prettyurls   = get(ENV, "CI", "false") == "true",
        canonical    = "https://Distribution-Matching.github.io/TruncatedDistributions.jl",
        assets       = String[],
    ),
    pages    = [
        "Home"                => "index.md",
        "Quick start"         => "quickstart.md",
        "Moment matching"     => "fitting.md",
        "Internals"           => "internals.md",
        "API reference"       => "api.md",
    ],
    warnonly = [:missing_docs],
)

deploydocs(
    repo      = "github.com/Distribution-Matching/TruncatedDistributions.jl.git",
    devbranch = "master",
    push_preview = true,
)
