using POMDPs
using POMDPTools
using MultiAgentPOMDPProblems
using ProgressMeter
using Printf
using JLD2
using CSV
using DataFrames
using DiscreteValueIteration

include("problems.jl")
include("suggested_action_policies.jl")
include("simulate.jl")

function load_policy(problem_symbol::Symbol)
    # Load joint_problem, agent_problems, joint_policy, agent_policies, joint_mdp_policy
    load_path = joinpath("src", "policies", "$problem_symbol.jld2")
    loaded_data = JLD2.load(load_path)
    joint_problem = loaded_data["joint_problem"]
    agent_problems = loaded_data["agent_problems"]
    joint_policy = loaded_data["joint_policy"]
    agent_policies = loaded_data["agent_policies"]
    joint_mdp_policy = loaded_data["joint_mdp_policy"]

    return joint_problem, agent_problems, joint_policy, agent_policies, joint_mdp_policy
end

function get_controller(control_option::Symbol, joint_problem, joint_policy, agent_problems, agent_policies)
    if control_option == :mpomdp
        control = JointPolicy(joint_problem, joint_policy)
    elseif control_option == :pomdp_1
        control = SinglePolicy(agent_problems[1], 1, agent_policies[1])
    elseif control_option == :pomdp_2
        control = SinglePolicy(agent_problems[2], 2, agent_policies[2])
    elseif control_option == :independent
        control = Independent(agent_problems, agent_policies)
    elseif control_option == :conflate_joint
        control = ConflateJoint(joint_problem, joint_policy, agent_problems, agent_policies)
    elseif control_option == :conflate_alpha
        control = Conflation(joint_problem, joint_policy, agent_problems, agent_policies;
            prune_option=:alpha,
            joint_belief_delta=1e-5,
            single_belief_delta=1e-5
        )
    elseif control_option == :conflate_action
        control = Conflation(joint_problem, joint_policy, agent_problems, agent_policies;
            prune_option=:action,
            joint_belief_delta=1e-5,
            single_belief_delta=1e-5
        )
    else
        throw(ArgumentError("Invalid control option: $control_option"))
    end
end

function print_policy_values(problem_symbol::Symbol)
    p = get_problem(problem_symbol, 1)
    num_agents = p.num_agents
    
    # Load the CSV file
    csv_file = joinpath("src", "policies", "policy_values.csv")
    df = CSV.read(csv_file, DataFrame)

    # Filter for the most recent entries for the given problem symbol
    most_recent_entries = df[df[:, :problem] .== string(problem_symbol), :]

    # Sort by date and time to get the most recent ones first
    most_recent_entries = sort(most_recent_entries, [:date, :time], rev=true)

    # Get the most recent entry for each policy
    unique_policies = unique(most_recent_entries[:, :policy])
    most_recent_policy_entries = DataFrame()

    for policy in unique_policies
        policy_entries = most_recent_entries[most_recent_entries[:, :policy] .== policy, :]
        most_recent_policy_entry = first(policy_entries, 1)
        append!(most_recent_policy_entries, most_recent_policy_entry)
    end

    # Create a dictionary mapping policy to value for the most recent entries
    policy_to_value = Dict(most_recent_policy_entries[:, :policy] .=> most_recent_policy_entries[:, :value])

    @printf("%-10s : %10.4f\n", "MMDP", policy_to_value["mmdp"])
    @printf("%-10s : %10.4f\n", "MPOMDP", policy_to_value["mpomdp"])
    for ii in 1:num_agents
        pomdp_str = "pomdp_$ii"
        @printf("%-10s : %10.4f\n", uppercase(pomdp_str), policy_to_value[pomdp_str])
    end
end

function run_simulation_for_policy(problem_symbol::Symbol, control_option::Symbol)
    
    joint_problem, agent_problems, joint_policy, agent_policies, joint_mdp_policy = load_policy(problem_symbol)

    print_policy_values(problem_symbol)

    cum_discounted_reward = 0.0
    cum_reward = 0.0
    num_runs = 2000
    discounted_reward = zeros(Float64, num_runs)
    reward = zeros(Float64, num_runs)

    @showprogress Threads.@threads for run_idx in 1:num_runs
        seed = run_idx
        s0 = rand(MersenneTwister(seed), initialstate(joint_problem))
        
        control = get_controller(control_option, joint_problem, joint_policy, agent_problems, agent_policies)

        results = run_simulation(
            joint_problem, s0, control;
            seed=seed,
            text_output=false, 
            max_steps=50,
            show_plots=false,
            joint_control=nothing
        )
        discounted_reward[seed] = results.cum_discounted_rew
        reward[seed] = results.cum_reward
    end

    ave_cum_discounted_reward = mean(discounted_reward)
    std_cum_discounted_reward = std(discounted_reward)
    std_err_discounted_reward = 1.96 * std_cum_discounted_reward / sqrt(num_runs)
    ave_cum_reward = mean(reward)
    std_cum_reward = std(reward)
    std_err_reward = 1.96 * std_cum_reward / sqrt(num_runs)

    @printf("Control Option: %s\n", control_option)
    @printf("%-40s : %10.4f ± %4.3f\n", "Average Cumulative Discounted Reward", ave_cum_discounted_reward, std_err_discounted_reward)
    @printf("%-40s : %10.4f ± %4.3f\n", "Average Cumulative Reward", ave_cum_reward, std_err_reward)

end


sims_to_run = [:tiger, :tiger_3, :tiger_4, 
                :broadcast, :broadcast_3_wp_low, 
                :joint_meet_2x2, :joint_meet_2x2_13, 
                :joint_meet_3x3, :joint_meet_3x3_wp_uni_init, :joint_meet_3_3x3_wp_uni_init, 
                :joint_meet_big_wp_uni_lr, 
                :box_push, 
                :wireless, :wireless_wp, 
                :stochastic_mars, :stochastic_mars_uni_init, :stochastic_mars_3_uni_init,
                :stochastic_mars_big_uni]

@printf(" INDEPENDENT\n")
for sim in sims_to_run
    @printf(" *** %s ***\n", sim)
    run_simulation_for_policy(sim, :independent)
end

@printf(" MPOMDP\n")
for sim in sims_to_run
    @printf(" *** %s ***\n", sim)
    run_simulation_for_policy(sim, :mpomdp)
end

@printf(" MPOMDP-C\n")
for sim in sims_to_run
    @printf(" *** %s ***\n", sim)
    run_simulation_for_policy(sim, :conflate_joint)
end

# seed = 3

# problem_symbol = :joint_meet_big_wp_uni_ls_03
# problem_symbol = :tiger
# control_option = :conflate_alpha

# joint_problem, agent_problems, joint_policy, agent_policies, joint_mdp_policy = load_policy(problem_symbol)

# print_policy_values(problem_symbol)

# # joint_control = get_controller(:mpomdp, joint_problem, joint_policy, agent_problems, agent_policies)
# joint_control = nothing
# control = get_controller(control_option, joint_problem, joint_policy, agent_problems, agent_policies)

# seed = 5

# s0 = rand(MersenneTwister(seed), initialstate(joint_problem))

# results = run_simulation(
#     joint_problem, s0, control;
#     seed=seed,
#     text_output=true, 
#     max_steps=10,
#     show_plots=true,
#     joint_control=joint_control
# )
