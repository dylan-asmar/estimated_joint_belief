using CSV
using DataFrames
using Printf


fn = "results_2024-10-15.csv"

resutls_fn = joinpath("src", "results", fn)

df = CSV.read(resutls_fn, DataFrame)

method_order = ["mpomdp", "conflate_joint", "conflate_alpha", "conflate_action", "pomdp_1", "independent"]

method_mapping = Dict(
        "mpomdp"           => "MPOMDP",
        "pomdp_1"          => "MPOMDP-I",
        "conflate_joint"   => "MCAS",
        "conflate_alpha"   => "MCAS-α",
        "conflate_action"  => "MCAS",
        "independent"      => "Independent"
)

problem_order = [
    "tiger", "tiger_3", "tiger_4", 
    "broadcast", "broadcast_dp_wp_3", 
    "joint_meet_2x2", "joint_meet_2x2_13", "joint_meet_2x2_ui_wp", 
    "joint_meet_3x3", "joint_meet_3x3_ag_ui_wp", "joint_meet_3x3_ag_ui_wp_3", 
    "joint_meet_19_lr_ui_wp", 
    "box_push", "box_push_so", 
    "wireless", "wireless_wp", 
    "stochastic_mars", "stochastic_mars_ui", "stochastic_mars_ui_3", "stochastic_mars_5g_ui"]

problem_mapping = Dict(
    "tiger"                    => ("Dec-Tiger", "---", 2),
    "tiger_3"                  => ("Dec-Tiger", "---", 3),
    "tiger_4"                  => ("Dec-Tiger", "---", 4),
    "broadcast"                => ("Broadcast", "---", 2),
    "broadcast_dp_wp_3"       => ("Broadcast", "DP, WP", 3),
    "joint_meet_2x2"           => ("Meet 2 x 2", "---", 2),
    "joint_meet_2x2_13"        => ("Meet 2 x 2", "SS", 2),
    "joint_meet_2x2_ui_wp" => ("Meet 2 x 2", "UI, WP", 2),
    "joint_meet_3x3"           => ("Meet 3 x 3", "---", 2),
    "joint_meet_3x3_ag_ui_wp" => ("Meet 3 x 3", "AG, UI, WP", 2),
    "joint_meet_3x3_ag_ui_wp_3" => ("Meet 3 x 3", "AG, UI, WP", 3),
    "joint_meet_19_lr_ui_wp" => ("Meet 19", "UI, WP", 2),
    "box_push" => ("Box Push", "---", 2),
    "box_push_so" => ("Box Push", "SO", 2),
    "wireless" => ("Wireless", "---", 2),
    "wireless_wp" => ("Wireless", "WP", 2),
    "stochastic_mars" => ("Mars Rover", "---", 2),
    "stochastic_mars_ui" => ("Mars Rover", "UI", 2),
    "stochastic_mars_ui_3" => ("Mars Rover", "UI", 3),
    "stochastic_mars_5g_ui" => ("Mars Rover", "5G, UI", 2)
)


cols = [("Cummulative Discounted Reward",:ave_cum_discounted_reward, :ci_discounted_reward),
("Max Number of Beliefs",:ave_max_num_beliefs_2, :ci_max_num_beliefs_2),
("Average Number of Beliefs",:ave_ave_num_beliefs_2, :ci_ave_num_beliefs_2)]

for (title, col, col_ci) in cols
    println()
    println(title)
    println()
    
# col = :ave_cum_discounted_reward
# col_ci = :ci_discounted_reward

    @printf("%-10s %12s %3s %12s %12s %12s %12s %12s %12s\n", "Problem", "Quals", "n", 
        "MPOMDP", "MPOMDP-C", "MCAS-α", "MCAS", "MPOMDP-I", "Independent")
    println("-"^104)

    for (pi, problem) in enumerate(problem_order)
        data = []
        methods = []
        for method in method_order
            data_df = df[df.problem_symbol .== problem, :]
            data_df = data_df[data_df.control_option .== method, :]
            if isempty(data_df)
                data = vcat(data, [0.0, 0.0])
            else
                data = vcat(data, [data_df[!, col], data_df[!, col_ci]])
            end
            methods = vcat(methods, method_mapping[method])
        end
        
        # Flatten the data array
        data = vcat(data...)
        
        if pi != 1
            p_name = problem_mapping[problem][1]
            p_name_prev = problem_mapping[problem_order[pi-1]][1]
            if p_name != p_name_prev
                println("-"^104)
            end
        end
        
        @printf("%-10s %12s %3d %6.1f ± %3.1f %6.1f ± %3.1f %6.1f ± %3.1f %6.1f ± %3.1f %6.1f ± %3.1f %6.1f ± %3.1f\n",
            problem_mapping[problem][1], problem_mapping[problem][2], problem_mapping[problem][3],
            data[1][1], data[2][1], data[3][1], data[4][1], data[5][1], data[6][1], data[7][1], data[8][1], data[9][1], data[10][1], data[11][1], data[12][1])
            
    end

    println()
end


cols = [("Percent of Runs that Max Belief Reached", :percent_reached_max_2)]
    
for (title, col) in cols
    println()
    println(title)
    println()

    @printf("%-10s %12s %3s %7s %7s\n", "Problem", "Quals", "n", "MCAS-α", "MCAS")
    println("-"^45)
    for problem in problem_order
        data = []
        for method in ["conflate_alpha", "conflate_action"]
            data_df = df[df.problem_symbol .== problem, :]
            data_df = data_df[data_df.control_option .== method, :]
            if isempty(data_df)
                data = vcat(data, 0.0)
            else
                data = vcat(data, data_df[!, col])
            end
        end
        
        @printf("%-10s %12s %3d %7.2f%% %7.2f%%\n", problem_mapping[problem][1], 
            problem_mapping[problem][2], problem_mapping[problem][3],
            data[1], data[2])
    end
end
