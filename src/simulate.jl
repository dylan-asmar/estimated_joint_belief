using POMDPs
using POMDPTools
import SARSOP
import NativeSARSOP

using Random
using Plots
using StatsBase
using JLD2
using Printf

# using JuMP
# using Gurobi
using Distances

using LinearAlgebra

using DataFrames
using ProgressMeter

using MultiAgentPOMDPProblems
    
include("suggested_action_policies.jl")

mutable struct SimulateResults
    step_count::Int
    cum_reward::Float64
    cum_discounted_rew::Float64
end

function Base.show(io::IO, results::SimulateResults)
    println(io, "SimulateResults")
    println(io, "\tStep Count                   : $(results.step_count)")
    println(io, "\tCumulative Reward            : $(results.cum_reward)")
    println(io, "\tCumulative Discounted Reward : $(results.cum_discounted_rew)")
end

function run_simulation(
    problem::POMDP{S, A, O}, # The problem used to drive the simulation
    init_state::S,
    control::MultiAgentControlStrategy;
    max_steps::Int=35,
    seed::Int=42,
    show_plots::Bool=false,
    text_output::Bool=false,
    joint_control::Union{Nothing, SinglePolicy, JointPolicy}=nothing
) where {S, A, O}
    rng = MersenneTwister(seed)
    
    num_agents = problem.num_agents
    γ = discount(problem)
    
    # Initialize the results struct
    results = SimulateResults(0, 0.0, 0.0)
    
    s = deepcopy(init_state)
    
    stop_simulating = false
    while !stop_simulating
        results.step_count += 1
        
        # Get actions based on the control strategy
        act, info = action_info(control)
        
        if !isnothing(joint_control)
            joint_a, _ = action_info(joint_control)
        end
        
        # Generate the next state, observation, and reward (simulate the problem)
        (sp, o, r) = @gen(:sp, :o, :r)(problem, s, act, rng)
        
        # Any text info output here for simulation inspection/debugging/etc.
        if text_output
            #TODO: Text output options here. Maybe use the info dict for this?
            @info "Step: $(results.step_count)" s act[1] act[2] sp o[1] o[2] r
        end
        
        # Update the results struct
        results.cum_reward += r
        results.cum_discounted_rew += r * γ^(results.step_count-1)
        
        # Plotting options to visualize the simulation and policy decisions
        if show_plots
            if !isnothing(joint_control)
                plot_step = (s=s, a=act, joint_a=joint_a, joint_b=joint_control.belief)
            else
                plot_step = (s=s, a=act)
            end
            plt = POMDPTools.render(control, plot_step)
            display(plt)
        end
        
        # Update the beliefs. This is different than the normal POMDPs.jl process because
        # we need to update the beliefs differently based on the control strategy.
        update_belief!(control, act, o)
        
        if !isnothing(joint_control)
            update_belief!(joint_control, act, o)
        end
        
        # Update the state and check if the simulation should continue
        s = sp
        
        if isterminal(problem, s) || results.step_count >= max_steps
            stop_simulating = true
        end
    end
    
    return results
end
