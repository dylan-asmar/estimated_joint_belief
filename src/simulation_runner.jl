using POMDPs
using POMDPTools
using MultiAgentPOMDPProblems
using LinearAlgebra
using SARSOP
import NativeSARSOP

using POMCPOW

using DiscreteValueIteration

using ProgressMeter

using Printf

include("suggested_action_policies.jl")
include("simulate_test.jl")

num_agents = 2

#* Checked!
# problem_type = MultiTigerPOMDP
# kwparams = (num_agents=num_agents,)

#* Checked!
# problem_type = BroadcastChannelPOMDP
# kwparams = (num_agents=num_agents, buffer_fill_prob=[0.9, 0.1])
# kwparams = (num_agents=num_agents, buffer_fill_prob=[0.9, 0.1], send_penalty=-0.2)
# kwparams = (num_agents=num_agents, buffer_fill_prob=[0.2, 0.4, 0.4], send_penalty=-0.2)

#* Checked, but still differs for upper bound when compared to "best" found Dec-POMDP
problem_type = WirelessPOMDP
kwparams = (
    num_agents=num_agents,
    idle_to_packet_prob=0.0470,
    packet_to_idle_prob=0.0741,
    discount_factor=0.99,
)
# kwparams = (
#     num_agents=num_agents,
#     idle_to_packet_prob=0.0470,
#     packet_to_idle_prob=0.0741,
#     discount_factor=0.99,
#     send_penalty=-0.2
# )


#* Checked!
# problem_type = StochasticMarsPOMDP
# kwparams = (
#     num_agents=num_agents, 
#     map_str="ds\nsd",
#     init_state=StochasticMarsState((1, 1), falses(4)),
# )
# kwparams = (
#     num_agents=num_agents, 
#     map_str="ds\nsd\ndx"
# )

#* Checked!
# problem_type = BoxPushPOMDP
# kwparams = (map_option=1,)
# kwparams = (map_option=1, observation_prob=0.7)

#* Checked, differs from "best" based on different initial state (same column vs corners)
#* The 3x3 seems to check out
# problem_type = JointMeetPOMDP
# kwparams = (
#     num_agents=num_agents,
#     map_str="oo\noo", 
#     observation_option=:boundaries_lr, 
#     init_state=JointMeetState((2, 3))
# )
# kwparams = (
#     num_agents=num_agents,
#     map_str="ooo\nooo\nooo", 
#     observation_option=:boundaries_both, 
#     init_state=JointMeetState((3, 7)),
#     meet_reward_locations=[1, 9]
# )
# kwparams = (
#     num_agents=num_agents,
#     map_str="ooo\nooo\nooo",
#     observation_option=:boundaries_both,
#     wall_penalty=-1.0, #* Emphasize individual vs joint policies!
# )
# kwparams = (
#     num_agents=num_agents,
#     map_str="""oxoooxo
#                oxoooxo
#                ooooooo
#                oxoooxo
#                oxoooxo""",
#     observation_option=:boundaries_both,
#     wall_penalty=-0.1, #* Emphasize individual vs joint policies!
# )


joint_problem = problem_type(; observation_agent=0, kwparams...);
agent_problems = [problem_type(; observation_agent=ii, kwparams...) for ii in 1:num_agents];


sarsop_solver = SARSOPSolver(; timeout=120.0);
joint_policy = solve(sarsop_solver, joint_problem);
joint_policy_value = value(joint_policy, initialstate(joint_problem));

vi_solver = SparseValueIterationSolver(max_iterations=1000, belres=1e-5, verbose=false);
joint_mdp_policy = solve(vi_solver, UnderlyingMDP(joint_problem));
joint_mdp_policy_value = 0.0;
for (si, p) in weighted_iterator(initialstate(joint_problem))
    val = value(joint_mdp_policy, si)
    joint_mdp_policy_value += p * value(joint_mdp_policy, si)
end

# pomcpow_solver = POMCPOWSolver(
#     # criterion=MaxUCB(20.0),
#     max_depth=20,
#     tree_queries=5000,
#     # max_time=10.0,
#     # exploration_constant=100.0,
#     # k_action=10.0,
#     # alpha_action=0.0,
#     # k_observation=10.0,
#     # alpha_observation=0.0,
# )
# joint_policy = solve(pomcpow_solver, joint_problem)


sarsop_solver = SARSOPSolver(; timeout=120.0)
agent_policies = Vector{AlphaVectorPolicy}(undef, num_agents)

for (ii, ap) in enumerate(agent_problems)
    agent_policies[ii] = solve(sarsop_solver, ap)
end

individual_policy_value = value(agent_policies[1], initialstate(joint_problem))

@printf("Joint MDP:     %10.4f\n", joint_mdp_policy_value)
@printf("Joint Policy:  %10.4f\n", joint_policy_value)
@printf("Individual:    %10.4f\n", individual_policy_value)

# joint_control = SinglePolicy(joint_problem, 0, joint_policy);
joint_control = nothing
# control = JointPolicy(joint_problem, joint_policy);

# control = SinglePolicy(joint_problem, 0, joint_policy);
control = SinglePolicy(agent_problems[1], 1, agent_policies[1]);
# control = Independent(agent_problems, agent_policies);

# control = ConflateJoint(joint_problem, joint_policy, agent_problems, agent_policies);

control = Conflation(joint_problem, joint_policy, agent_problems, agent_policies;
    prune_option=:action,
    joint_belief_delta=2*eps(Float64),
    single_belief_delta=2*eps(Float64)
);

# s0 = rand(initialstate(joint_problem))

# results = run_simulation(
#     joint_problem, s0, control;
#     seed=1,
#     text_output=false, 
#     max_steps=20,
#     show_plots=false,
#     joint_control=joint_control
# )

cum_discounted_reward = 0.0
cum_reward = 0.0
num_runs = 50
@showprogress for seed in 1:num_runs
    s0 = rand(initialstate(joint_problem))
    joint_control = nothing
    control = JointPolicy(joint_problem, joint_policy);

    results = run_simulation(
        joint_problem, s0, control;
        seed=seed,
        text_output=false, 
        max_steps=45,
        show_plots=false,
        joint_control=joint_control
    )
    cum_discounted_reward += results.cum_discounted_rew
    cum_reward += results.cum_reward
end

ave_cum_discounted_reward = cum_discounted_reward / num_runs
# ave_reward_per_step = cum_reward / num_runs / 45 # 45 steps

println("Average Cumulative Discounted Reward: $ave_cum_discounted_reward")
# println("Average Reward Per Step: $ave_reward_per_step")
