using POMDPs
using MultiAgentPOMDPProblems

PARAM_OPTIONS = Dict(
    :tiger => (MultiTigerPOMDP, (num_agents=2,)),
    :tiger_3 => (MultiTigerPOMDP, (num_agents=3,)),
    :tiger_4 => (MultiTigerPOMDP, (num_agents=4,)), 
    
    :broadcast => (BroadcastChannelPOMDP, (num_agents=2,
        buffer_fill_prob=[0.9, 0.1])),
    :broadcast_wp => (BroadcastChannelPOMDP, (num_agents=2,
        buffer_fill_prob=[0.9, 0.1],
        send_penalty=-0.2)),
    :broadcast_3 => (BroadcastChannelPOMDP, (num_agents=3,
        buffer_fill_prob=[0.8, 0.1, 0.1])),
    :broadcast_3_wp_low => (BroadcastChannelPOMDP, (num_agents=3,
        buffer_fill_prob=[0.2, 0.4, 0.4],
        send_penalty=-0.2)),
    
    :wireless => (WirelessPOMDP, (num_agents=2,
        idle_to_packet_prob=0.0470,
        packet_to_idle_prob=0.0741,
        discount_factor=0.99)),
    :wireless_wp => (WirelessPOMDP, (num_agents=2,
        idle_to_packet_prob=0.0470,
        packet_to_idle_prob=0.0741,
        discount_factor=0.99,
        send_penalty=-0.2)),
    
    :stochastic_mars => (StochasticMarsPOMDP, (num_agents=2,
        map_str="ds\nsd",
        init_state=StochasticMarsState((1, 1), falses(4)))),
    :stochastic_mars_uni_init => (StochasticMarsPOMDP, (num_agents=2,
        map_str="ds\nsd")),
    :stochastic_mars_big_uni => (StochasticMarsPOMDP, (num_agents=2,
        map_str="dss\nsdx")),
    :stochastic_mars_3_uni_init => (StochasticMarsPOMDP, (num_agents=3,
        map_str="ds\nsd")),
    
    :box_push => (BoxPushPOMDP, (num_agents=2,
        map_option=1)),
    :box_push_obs_05 => (BoxPushPOMDP, (num_agents=2,
        map_option=1,
        observation_prob=0.5)),
    :box_push_2 => (BoxPushPOMDP, (num_agents=2,
        map_option=2)),
    
    :joint_meet_2x2 => (JointMeetPOMDP, (num_agents=2,
        map_str="oo\noo",
        observation_option=:boundaries_lr,
        init_state=JointMeetState((2, 3)))),
    :joint_meet_2x2_13 => (JointMeetPOMDP, (num_agents=2,
        map_str="oo\noo",
        observation_option=:boundaries_lr,
        init_state=JointMeetState((1, 3)))),
    :joint_meet_3x3 => (JointMeetPOMDP, (num_agents=2,
        map_str="ooo\nooo\nooo",
        observation_option=:boundaries_both,
        init_state=JointMeetState((3, 7)),
        meet_reward_locations=[1, 9])),
    :joint_meet_3x3_wp_uni_init => (JointMeetPOMDP, (num_agents=2,
        map_str="ooo\nooo\nooo",
        observation_option=:boundaries_both,
        wall_penalty=-0.1)),
    :joint_meet_big_wp_uni_both => (JointMeetPOMDP, (num_agents=2,
        map_str="""oxoooxo
                   oxoooxo
                   ooooooo
                   oxoooxo
                   oxoooxo""",
        observation_option=:boundaries_both,
        wall_penalty=-0.1)),
    :joint_meet_big_wp_uni_ls_03 => (JointMeetPOMDP, (num_agents=2,
        map_str="""oxoooxo
                   oxoooxo
                   ooooooo
                   oxoooxo
                   oxoooxo""",
        observation_option=:left_and_same,
        wall_penalty=-0.1,
        observation_sigma=0.3)),
    :joint_meet_3_3x3_wp_uni_init => (JointMeetPOMDP, (num_agents=3,
        map_str="ooo\nooo\nooo",
        observation_option=:boundaries_both,
        wall_penalty=-0.1)),
)

function get_problem(
    param_option::Symbol,
    observation_agent::Int;
    kwparams::NamedTuple=NamedTuple()
)
    if !haskey(PARAM_OPTIONS, param_option)
        throw(ArgumentError("Invalid parameter option: $param_option"))
    end    

    problem_type = PARAM_OPTIONS[param_option][1]
    
    if !isempty(kwparams)
        @info "Using custom parameters instead of the defined in $param_option"
    else
        kwparams = PARAM_OPTIONS[param_option][2]
    end

    return problem_type(; observation_agent=observation_agent, kwparams...)
end
