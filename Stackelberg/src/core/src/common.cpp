#include "common.hpp"
#include <sstream>

namespace posg_core {

    // Action implementation
    Action::Action() : action_id(0), agent_id(0) {}
    
    Action::Action(int action_id, int agent_id) : action_id(action_id), agent_id(agent_id) {}
    
    bool Action::operator==(const Action& other) const {
        return action_id == other.action_id && agent_id == other.agent_id;
    }
    
    bool Action::operator!=(const Action& other) const {
        return !(*this == other);
    }
    
    bool Action::operator<(const Action& other) const {
        if (agent_id != other.agent_id) {
            return agent_id < other.agent_id;
        }
        return action_id < other.action_id;
    }
    
    std::string Action::to_string() const {
        std::ostringstream oss;
        oss << "Action(" << action_id << ", " << agent_id << ")";
        return oss.str();
    }

    // Observation implementation
    Observation::Observation(int observation_id, int agent_id) : observation_id(observation_id), agent_id(agent_id) {}
    
    bool Observation::operator==(const Observation& other) const {
        return observation_id == other.observation_id && agent_id == other.agent_id;
    }
    
    bool Observation::operator!=(const Observation& other) const {
        return !(*this == other);
    }
    
    bool Observation::operator<(const Observation& other) const {
        if (agent_id != other.agent_id) {
            return agent_id < other.agent_id;
        }
        return observation_id < other.observation_id;
    }
    
    std::string Observation::to_string() const {
        std::ostringstream oss;
        oss << "Observation(" << observation_id << ", " << agent_id << ")";
        return oss.str();
    }

    // JointAction implementation
    JointAction::JointAction(const Action& leader_action, const Action& follower_action) 
        : leader_action(leader_action), follower_action(follower_action) {}
    
    bool JointAction::operator==(const JointAction& other) const {
        return leader_action == other.leader_action && follower_action == other.follower_action;
    }
    
    bool JointAction::operator!=(const JointAction& other) const {
        return !(*this == other);
    }
    
    std::string JointAction::to_string() const {
        std::ostringstream oss;
        oss << "JointAction(" << leader_action.to_string() << ", " << follower_action.to_string() << ")";
        return oss.str();
    }

    // JointObservation implementation
    JointObservation::JointObservation(const Observation& leader_obs, const Observation& follower_obs) 
        : leader_observation(leader_obs), follower_observation(follower_obs) {}
    
    bool JointObservation::operator==(const JointObservation& other) const {
        return leader_observation == other.leader_observation && follower_observation == other.follower_observation;
    }
    
    bool JointObservation::operator!=(const JointObservation& other) const {
        return !(*this == other);
    }
    
    std::string JointObservation::to_string() const {
        std::ostringstream oss;
        oss << "JointObservation(" << leader_observation.to_string() << ", " << follower_observation.to_string() << ")";
        return oss.str();
    }

    // Hash function implementations
    size_t ActionHash::operator()(const Action& action) const {
        return std::hash<int>()(action.get_action_id()) ^ (std::hash<int>()(action.get_agent_id()) << 1);
    }

    size_t ObservationHash::operator()(const Observation& obs) const {
        return std::hash<int>()(obs.get_observation_id()) ^ (std::hash<int>()(obs.get_agent_id()) << 1);
    }

    size_t JointActionHash::operator()(const JointAction& joint_action) const {
        ActionHash action_hash;
        return action_hash(joint_action.get_leader_action()) ^ (action_hash(joint_action.get_follower_action()) << 1);
    }

    size_t JointObservationHash::operator()(const JointObservation& joint_obs) const {
        ObservationHash obs_hash;
        return obs_hash(joint_obs.get_leader_observation()) ^ (obs_hash(joint_obs.get_follower_observation()) << 1);
    }

} // namespace posg_core 