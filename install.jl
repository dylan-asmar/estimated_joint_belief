using Pkg

Pkg.activate(".")
Pkg.instantiate()
Pkg.develop(path="./MultiAgentPOMDPProblems")
Pkg.build()
