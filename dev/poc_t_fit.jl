using TruncatedDistributions, Distributions, LinearAlgebra, Random, Printf, Optim
import TruncatedDistributions: make_param_vec_from_μ_Σ
set_loss_gamma!(1.0)
rng = MersenneTwister(11)
function problem(n, semi)
  A=randn(rng,n,n); Σ=A*A'/n+0.5I; μ=0.4randn(rng,n); a=-1 .- rand(rng,n); b=1 .+ rand(rng,n); semi && (b[1]=Inf)
  μ, Σ, a, b
end
println("== 1. Theorem-2 gradient vs central FD (hcubature backend)")
set_kr_base_backend!(:hcubature)
for n in (2,3), ν in (4.5, 10.0), semi in (false,true)
  μ, Σ, a, b = problem(n, semi)
  μ̂ = μ .+ 0.1; Σ̂ = 0.8 .* Σ
  p = make_param_vec_from_μ_Σ(μ .- 0.05, 1.1 .* Σ)
  G = similar(p); vector_fg_true_loss_t(nothing, G, p, ν, a, b, μ̂, Σ̂)
  h=1e-5; Gfd=[(vector_moment_loss_t(p+h*e,ν,a,b,μ̂,Σ̂)-vector_moment_loss_t(p-h*e,ν,a,b,μ̂,Σ̂))/2h for e in eachcol(Matrix(1.0I,length(p),length(p)))]
  @printf("n=%d ν=%4.1f semi=%-5s  rel err %.1e\n", n, ν, semi, norm(G-Gfd)/norm(Gfd)); flush(stdout)
end
println("\n== 2. Recover (μ*, Σ*) from truncated moments of t_ν(μ*, Σ*)")
for backend in (:hcubature, :mvnormalcdf), n in (2,3), ν in (4.5, 10.0), semi in (false,true)
  (backend == :mvnormalcdf && n == 2) && continue
  set_kr_base_backend!(:hcubature)
  μs, Σs, a, b = problem(n, semi)
  dt = TruncatedMvT(ν, μs, Σs, a, b; max_moment_levels=2); μ̂ = mean(dt); Σ̂ = Matrix(cov(dt))
  set_kr_base_backend!(backend); set_kr_base_rng!(MersenneTwister(0))
  μf, Σf, info = fit_mvt(ν, μ̂, Σ̂, a, b; ftarget=1e-10)
  # baseline: same loss, L-BFGS with finite-difference gradient
  t = @elapsed r = optimize(p->vector_moment_loss_t(p,ν,a,b,μ̂,Σ̂), make_param_vec_from_μ_Σ(μ̂, (ν-2)/ν .* Σ̂), LBFGS(),
                            Optim.Options(iterations=200, time_limit=60.0, callback=s->s.f_x<1e-10))
  @printf("%-11s n=%d ν=%4.1f semi=%-5s | Thm2: L=%.1e it=%3d %6.2fs ‖Δμ‖=%.1e ‖ΔΣ‖=%.1e | FD: L=%.1e it=%3d %6.2fs\n",
          backend, n, ν, semi, info.loss, info.iterations, info.time, norm(μf-μs), norm(Σf-Σs), r.minimum, r.iterations, t); flush(stdout)
end
