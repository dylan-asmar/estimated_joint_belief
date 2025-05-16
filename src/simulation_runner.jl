
using POMDPs
using POMDPTools
using MultiAgentPOMDPProblems
using ProgressMeter
using Printf
using JLD2

using DiscreteValueIteration # needed for loading the MMDP policy

# Used for handling results
using CSV
using DataFrames
using DataFrames
using Logging
using Dates

include("problems.jl")
include("suggested_action_policies.jl")
include("simulate.jl")
include("problem_and_policy_helpers.jl")

function run_simulation_for_policy(problem_symbol::Symbol, control_option::Symbol; 
    delta_single::Float64=1e-5, delta_joint::Float64=1e-5, max_beliefs::Int=1_000_000,
    num_runs::Int=2000, max_steps::Int=50, seed_offset::Int=0, print_to_console::Bool=false)
    
    joint_problem, agent_problems, joint_policy, agent_policies, _ = load_policy(problem_symbol)

    num_agents = joint_problem.num_agents
    
    print_policy_values(problem_symbol)

    discounted_reward = zeros(Float64, num_runs)
    reward = zeros(Float64, num_runs)
    max_num_beliefs = [zeros(Int, num_runs) for _ in 1:num_agents]
    ave_num_beliefs = [zeros(Float64, num_runs) for _ in 1:num_agents]

    @showprogress Threads.@threads for run_idx in 1:num_runs
        seed = run_idx + seed_offset
        try
            s0 = rand(MersenneTwister(seed), initialstate(joint_problem))
            
            control = get_controller(
                control_option, joint_problem, joint_policy, agent_problems, agent_policies;
                delta_single=delta_single,
                delta_joint=delta_joint,
                max_beliefs=max_beliefs
            )

            results = run_simulation(
                joint_problem, s0, control;
                seed=seed,
                text_output=false, 
                max_steps=max_steps,
                show_plots=false,
                joint_control=nothing
            )
            discounted_reward[run_idx] = results.cum_discounted_rew
            reward[run_idx] = results.cum_reward
            for jj in 1:(num_agents-1)
                max_num_beliefs[jj][run_idx] = maximum(results.num_beliefs[jj+1])
                ave_num_beliefs[jj][run_idx] = mean(results.num_beliefs[jj+1])
            end
        catch e
            println("Error with seed $seed")
            rethrow(e)
        end
    end

    ave_cum_discounted_reward = mean(discounted_reward)
    std_cum_discounted_reward = std(discounted_reward)
    ci_discounted_reward = 1.96 * std_cum_discounted_reward / sqrt(num_runs)
    ave_cum_reward = mean(reward)
    std_cum_reward = std(reward)
    ci_reward = 1.96 * std_cum_reward / sqrt(num_runs)

    max_max_num_beliefs = zeros(Float64, num_agents)
    ave_max_num_beliefs = zeros(Float64, num_agents)
    ci_max_num_beliefs = zeros(Float64, num_agents)
    ave_ave_num_beliefs = zeros(Float64, num_agents)
    ci_ave_num_beliefs = zeros(Float64, num_agents)
    max_beliefs_reached = zeros(Int, num_agents)
    for jj in 1:(num_agents-1)
        max_max_num_beliefs[jj] = maximum(max_num_beliefs[jj])
        ave_max_num_beliefs[jj] = mean(max_num_beliefs[jj])
        ci_max_num_beliefs[jj] = 1.96 * std(max_num_beliefs[jj]) / sqrt(num_runs)
        ave_ave_num_beliefs[jj] = mean(ave_num_beliefs[jj])
        ci_ave_num_beliefs[jj] = 1.96 * std(ave_num_beliefs[jj]) / sqrt(num_runs)
        
        # Count how many runs reached the max belief limit
        max_beliefs_reached[jj] = sum(max_num_beliefs[jj] .>= max_beliefs)
    end
    
    if print_to_console
        @printf("Problem Symbol: %s\n", problem_symbol)
        @printf("Control Option: %s\n", control_option)
        @printf("\tDelta Single: %1.1e\n", delta_single)
        @printf("\tDelta Joint: %1.1e\n", delta_joint)
        @printf("\tMax Beliefs: %d\n", max_beliefs)
        @printf("\tNum Runs: %d\n", num_runs)
        @printf("\tMax Steps: %d\n", max_steps)
        @printf("\tSeed Offset: %d\n", seed_offset)
        @printf("%-40s : %10.4f ± %4.3f\n", "Average Cumulative Discounted Reward", ave_cum_discounted_reward, ci_discounted_reward)
        @printf("%-40s : %10.4f ± %4.3f\n", "Average Cumulative Reward", ave_cum_reward, ci_reward)
        for jj in 1:(num_agents-1)
            @printf("%-40s : %10.4f ± %4.3f\n", "Average Max Num Beliefs $(jj+1)", ave_max_num_beliefs[jj], ci_max_num_beliefs[jj])
            @printf("%-40s : %10.4f ± %4.3f\n", "Average Ave Num Beliefs $(jj+1)", ave_ave_num_beliefs[jj], ci_ave_num_beliefs[jj])
            @printf("%-40s : %10.4f\n", "Max Beliefs Reached $(jj+1)", max_beliefs_reached[jj])
            @printf("%-40s : %10.2f %%\n", "Percent Reached Max $(jj+1)", max_beliefs_reached[jj] / num_runs * 100)
        end
    end

    # Output results as a df
    df_prob_data_cols = [:problem_symbol, :control_option, :delta_single, :delta_joint, :max_beliefs, :num_runs, :max_steps, :seed_offset]
    df_results_cols = [:ave_cum_discounted_reward, :ci_discounted_reward, :ave_cum_reward, :ci_reward]

    df_results_cols_per_agent = [
        [Symbol("ave_max_num_beliefs_$jj") for jj in 2:num_agents],
        [Symbol("ci_max_num_beliefs_$jj") for jj in 2:num_agents],
        [Symbol("ave_ave_num_beliefs_$jj") for jj in 2:num_agents],
        [Symbol("ci_ave_num_beliefs_$jj") for jj in 2:num_agents],
        [Symbol("max_beliefs_reached_$jj") for jj in 2:num_agents],
        [Symbol("percent_reached_max_$jj") for jj in 2:num_agents]
    ]
    df_results_cols_per_agent = vcat(df_results_cols_per_agent...)
    df_cols = [df_prob_data_cols; df_results_cols; df_results_cols_per_agent]

    results_prob_data = [problem_symbol, control_option, delta_single, delta_joint, max_beliefs, num_runs, max_steps, seed_offset]
    results_data = [ave_cum_discounted_reward, ci_discounted_reward, ave_cum_reward, ci_reward]
    ave_max_num_beliefs_list = [ave_max_num_beliefs[jj] for jj in 1:(num_agents-1)]
    ci_max_num_beliefs_list = [ci_max_num_beliefs[jj] for jj in 1:(num_agents-1)]
    ave_ave_num_beliefs_list = [ave_ave_num_beliefs[jj] for jj in 1:(num_agents-1)]
    ci_ave_num_beliefs_list = [ci_ave_num_beliefs[jj] for jj in 1:(num_agents-1)]
    max_beliefs_reached_list = [max_beliefs_reached[jj] for jj in 1:(num_agents-1)]
    percent_reached_max_list = [(max_beliefs_reached[jj] / num_runs) * 100 for jj in 1:(num_agents-1)]

    results_data_per_agent = [
        ave_max_num_beliefs_list;
        ci_max_num_beliefs_list;
        ave_ave_num_beliefs_list;
        ci_ave_num_beliefs_list;
        max_beliefs_reached_list;
        percent_reached_max_list
    ]

    results_row = vcat(results_prob_data, results_data, results_data_per_agent)
    
    df = DataFrame([[r] for r in results_row], df_cols)

    return df    
end


# global_logger(ConsoleLogger(Debug)) # Comment/uncomment as desired

sims_to_run = [ 
                :tiger, :tiger_3, :tiger_4, 
                :broadcast, :broadcast_dp_wp_3,                 
                :joint_meet_2x2, 
                :joint_meet_2x2_13, 
                :joint_meet_3x3,  
                :joint_meet_2x2_ui_wp,
                :box_push,
                :stochastic_mars,
                :stochastic_mars_ui,
                :joint_meet_3x3_ag_ui_wp,
                :stochastic_mars_ui_3,
                :stochastic_mars_5g_ui,
                :joint_meet_3x3_ag_ui_wp_3,
                :box_push_so,
                :wireless, :wireless_wp,
                :joint_meet_27_lr_ui_wp, 
]

controllers_to_run = [:independent, :mpomdp, :pomdp_1, :conflate_joint, :conflate_alpha, :conflate_action]

num_runs = 2000


existing_results = joinpath("src", "results", "results_2024-10-15_16-01.csv")
if isfile(existing_results)
    df = CSV.read(existing_results, DataFrame)
else
    df = DataFrame()
end

results_fn = joinpath("src", "results", "results_$(Dates.format(now(), "yyyy-mm-dd_HH-MM")).csv")

for sim in sims_to_run
    for ctrl in controllers_to_run
        if sim == :wireless || sim == :wireless_wp
            max_steps = 450
        else
            max_steps = 50
        end
        df_temp = run_simulation_for_policy(sim, ctrl; max_beliefs=200, num_runs=num_runs, max_steps=max_steps)
        append!(df, df_temp; cols=:union)
        CSV.write(results_fn, df)
    end
end
