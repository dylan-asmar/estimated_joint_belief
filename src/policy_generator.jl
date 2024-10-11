using POMDPs
using POMDPTools
using MultiAgentPOMDPProblems
using SARSOP
using DiscreteValueIteration
using Printf
# import NativeSARSOP
using JLD2
using Dates
using CSV
using DataFrames

include("problems.jl")

# PROBLEMS_TO_RUN = [
#     (:tiger, 60.0),
#     (:tiger_3, 60.0),
#     (:tiger_4, 60.0),
#     (:broadcast, 120.0),
#     (:broadcast_wp, 120.0),
#     (:broadcast_3, 120.0),
#     (:broadcast_3_wp_low, 120.0),
#     (:stochastic_mars, 60.0),
#     (:stochastic_mars_uni_init, 120.0),
#     (:stochastic_mars_big_uni, 300.0),
#     (:box_push, 120.0),
#     (:box_push_obs_05, 300.0),
#     (:joint_meet_2x2, 120.0),
#     (:joint_meet_2x2_13, 120.0),
#     (:joint_meet_3x3, 240.0),
#     (:joint_meet_3x3_wp_uni_init, 300.0),
#     (:joint_meet_big_wp_uni_both, 1200.0),
#     (:joint_meet_big_wp_uni_ls_03, 1800.0),
#     (:wireless, 600.0),
#     (:wireless_wp, 600.0)
# ]

PROBLEMS_TO_RUN = [
    (:joint_meet_3x3_wp_uni_init, 300.0),
    (:joint_meet_3_3x3_wp_uni_init, 8000.0),
]

function solve_and_save_problem(problem_symbol::Symbol, max_time::Float64)

    joint_problem = get_problem(problem_symbol, 0)
    agent_problems = [get_problem(problem_symbol, ii) for ii in 1:joint_problem.num_agents]

    @info "Solving MPOMDP for $problem_symbol..."
    
    sarsop_solver = SARSOPSolver(; timeout=max_time)
    
    joint_policy = solve(sarsop_solver, joint_problem)
    joint_policy_value = value(joint_policy, initialstate(joint_problem))

    vi_solver = SparseValueIterationSolver(max_iterations=1000, belres=1e-5, verbose=false);
    joint_mdp_policy = solve(vi_solver, UnderlyingMDP(joint_problem));
    joint_mdp_policy_value = 0.0;
    for (si, p) in weighted_iterator(initialstate(joint_problem))
        joint_mdp_policy_value += p * value(joint_mdp_policy, si)
    end

    @info "Solving individual policies... "
    sarsop_solver = SARSOPSolver(; timeout=max_time)
    agent_policies = Vector{AlphaVectorPolicy}(undef, joint_problem.num_agents)
    indiv_policy_values = Vector{Float64}(undef, joint_problem.num_agents)
    for (ii, ap) in enumerate(agent_problems)
        agent_policies[ii] = solve(sarsop_solver, ap)
        indiv_policy_values[ii] = value(agent_policies[ii], initialstate(ap))
    end

    file_name = joinpath("src", "policies", "$problem_symbol.jld2")
    @info "Saving $file_name"
    JLD2.@save file_name joint_problem agent_problems agent_policies joint_policy joint_mdp_policy
    
    
    # If CSV for policy values exist, append the results, otherwise create a new CSV
    csv_file = joinpath("src", "policies", "policy_values.csv")
    if isfile(csv_file)
        df = CSV.read(csv_file, DataFrame; types=Dict(:problem => String, :policy => String))
    else
        df = DataFrame(
            problem=String[],
            policy=String[],
            solve_time=Float64[],
            value=Float64[],
            date=Date[],
            time=Time[],
        )
    end
    
    push!(df, (problem=string(problem_symbol), policy="mmdp", solve_time=max_time, value=joint_mdp_policy_value, 
        date=Date(now()), time=Time(now())))
    push!(df, (problem=string(problem_symbol), policy="mpomdp", solve_time=max_time, value=joint_policy_value, 
        date=Date(now()), time=Time(now())))
    for ii in 1:joint_problem.num_agents
        push!(df, (problem=string(problem_symbol), policy="pomdp_$ii", solve_time=max_time, value=indiv_policy_values[ii], 
            date=Date(now()), time=Time(now())))
    end
    
    CSV.write(csv_file, df)
    
    @printf("%10s : %10.4f\n", "MMDP", joint_mdp_policy_value)
    @printf("%10s : %10.4f\n", "MPOMDP", joint_policy_value)
    for ii in 1:joint_problem.num_agents
        @printf("%10s : %10.4f\n", "POMDP $ii", indiv_policy_values[ii])
    end
    
    @info "Complete!"
end

# problem_symbol, max_time = PROBLEMS_TO_RUN[8]
# solve_and_save_problem(problem_symbol, max_time)

for (problem_symbol, max_time) in PROBLEMS_TO_RUN
    solve_and_save_problem(problem_symbol, max_time)
end
