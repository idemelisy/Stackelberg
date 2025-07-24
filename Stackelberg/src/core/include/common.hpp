#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <functional>

namespace posg_core {

    /**
     * @brief Represents an action taken by an agent
     */
    class Action {
    private:
        int action_id;
        int agent_id;  // 0 for leader, 1 for follower

    public:
        Action(); // Default constructor
        Action(int action_id, int agent_id);
        
        int get_action_id() const { return action_id; }
        int get_agent_id() const { return agent_id; }
        
        bool operator==(const Action& other) const;
        bool operator!=(const Action& other) const;
        bool operator<(const Action& other) const;
        
        std::string to_string() const;
    };

    /**
     * @brief Represents an observation received by an agent
     */
    class Observation {
    private:
        int observation_id;
        int agent_id;  // 0 for leader, 1 for follower

    public:
        Observation(int observation_id, int agent_id);
        
        int get_observation_id() const { return observation_id; }
        int get_agent_id() const { return agent_id; }
        
        bool operator==(const Observation& other) const;
        bool operator!=(const Observation& other) const;
        bool operator<(const Observation& other) const;
        
        std::string to_string() const;
    };

    /**
     * @brief Represents a history of actions and observations for an agent
     * 
     * From Paper: Agent History
     * An agent history h_i = (a_i^1, z_i^1, a_i^2, z_i^2, ..., a_i^t, z_i^t) where:
     * - a_i^j is the j-th action taken by agent i
     * - z_i^j is the j-th observation received by agent i
     * 
     * This tracks the complete information available to an agent at any timestep.
     */
    class AgentHistory {
    private:
        std::vector<Action> actions;
        std::vector<Observation> observations;
        int agent_id;

    public:
        AgentHistory(int agent_id);
        
        void add_action(const Action& action);
        void add_observation(const Observation& observation);
        
        const std::vector<Action>& get_actions() const { return actions; }
        const std::vector<Observation>& get_observations() const { return observations; }
        int get_agent_id() const { return agent_id; }
        size_t length() const { return actions.size(); }
        
        std::string to_string() const;
        bool operator==(const AgentHistory& other) const;
        bool operator!=(const AgentHistory& other) const;
        bool operator<(const AgentHistory& other) const;
    };

    /**
     * @brief Represents a joint action (leader action + follower action)
     */
    class JointAction {
    private:
        Action leader_action;
        Action follower_action;

    public:
        JointAction(const Action& leader_action, const Action& follower_action);
        
        const Action& get_leader_action() const { return leader_action; }
        const Action& get_follower_action() const { return follower_action; }
        
        bool operator==(const JointAction& other) const;
        bool operator!=(const JointAction& other) const;
        
        std::string to_string() const;
    };

    /**
     * @brief Represents a joint observation (leader observation + follower observation)
     */
    class JointObservation {
    private:
        Observation leader_observation;
        Observation follower_observation;

    public:
        JointObservation(const Observation& leader_obs, const Observation& follower_obs);
        
        const Observation& get_leader_observation() const { return leader_observation; }
        const Observation& get_follower_observation() const { return follower_observation; }
        
        bool operator==(const JointObservation& other) const;
        bool operator!=(const JointObservation& other) const;
        
        std::string to_string() const;
    };

    // Hash functions for use in unordered containers
    struct ActionHash {
        size_t operator()(const Action& action) const;
    };

    struct ObservationHash {
        size_t operator()(const Observation& obs) const;
    };

    struct JointActionHash {
        size_t operator()(const JointAction& joint_action) const;
    };

    struct JointObservationHash {
        size_t operator()(const JointObservation& joint_obs) const;
    };

} // namespace posg_core 

// Hash specializations for core types
namespace std {
    /**
     * @brief Hash specialization for posg_core::Action
     *
     * Hashes both action_id and agent_id for use in unordered_map/set.
     */
    template<>
    struct hash<posg_core::Action> {
        std::size_t operator()(const posg_core::Action& a) const noexcept {
            std::size_t h1 = std::hash<int>{}(a.get_action_id());
            std::size_t h2 = std::hash<int>{}(a.get_agent_id());
            return h1 ^ (h2 << 1);
        }
    };

    /**
     * @brief Hash specialization for posg_core::Observation
     *
     * Hashes both observation_id and agent_id for use in unordered_map/set.
     */
    template<>
    struct hash<posg_core::Observation> {
        std::size_t operator()(const posg_core::Observation& o) const noexcept {
            std::size_t h1 = std::hash<int>{}(o.get_observation_id());
            std::size_t h2 = std::hash<int>{}(o.get_agent_id());
            return h1 ^ (h2 << 1);
        }
    };

    /**
     * @brief Hash specialization for posg_core::AgentHistory
     *
     * Hashes the agent_id and all actions/observations in the history.
     */
    template<>
    struct hash<posg_core::AgentHistory> {
        std::size_t operator()(const posg_core::AgentHistory& h) const {
            std::size_t hash = std::hash<int>{}(h.get_agent_id());
            for (const auto& a : h.get_actions()) {
                hash ^= std::hash<int>{}(a.get_action_id()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
            for (const auto& o : h.get_observations()) {
                hash ^= std::hash<int>{}(o.get_observation_id()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
            return hash;
        }
    };
} 