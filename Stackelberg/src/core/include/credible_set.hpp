#pragma once

#include "occupancy_state.hpp"
#include "conditional_occupancy_state.hpp"
#include <set>
#include <vector>
#include <unordered_map>

namespace posg_core {

    // Forward declarations
    class Action;
    class Observation;
    class JointAction;
    class JointObservation;
    class TransitionModel;
    class ObservationModel;

    /**
     * @brief Represents a credible set - a set of occupancy states
     * 
     * This is a fundamental concept from the AAAI 2026 paper. A credible set represents
     * the collection of occupancy states that are reachable under some follower response
     * to the leader's policy. This is the state space of the Credible MDP.
     */
    class CredibleSet {
    private:
        // Set of occupancy states in this credible set
        std::set<OccupancyState, std::less<OccupancyState>> occupancy_states;
        
        // Timestep tracking
        int timestep;
        
        // Cache for conditional occupancy states
        mutable std::unordered_map<AgentHistory, std::vector<ConditionalOccupancyState>> 
            conditional_occupancy_cache;

    public:
        // Constructors
        CredibleSet();
        CredibleSet(const std::vector<OccupancyState>& states);
        CredibleSet(const OccupancyState& single_state);
        
        // Core credible set operations
        void add_occupancy_state(const OccupancyState& occupancy_state);
        void remove_occupancy_state(const OccupancyState& occupancy_state);
        
        // Credible set access
        const std::set<OccupancyState, std::less<OccupancyState>>& get_occupancy_states() const { 
            return occupancy_states; 
        }
        size_t size() const { return occupancy_states.size(); }
        bool empty() const { return occupancy_states.empty(); }
        
        // Accessor for timestep (needed for hashing and external logic)
        int get_timestep() const { return timestep; }
        // Setter for timestep (needed for CMDP logic)
        void set_timestep(int t) { timestep = t; }
        
        // Get conditional occupancy states for all follower histories
        std::vector<ConditionalOccupancyState> get_conditional_occupancy_states() const;
        
        // Get conditional occupancy states for a specific follower history
        std::vector<ConditionalOccupancyState> get_conditional_occupancy_states(
            const AgentHistory& follower_history) const;
        
        // Information measures
        double entropy() const;
        double distance_to(const CredibleSet& other) const;
        
        // Hausdorff distance between credible sets
        double hausdorff_distance(const CredibleSet& other) const;
        
        // Utility methods
        bool contains(const OccupancyState& occupancy_state) const;
        std::string to_string() const;
        
        // Comparison operators
        bool operator==(const CredibleSet& other) const;
        bool operator!=(const CredibleSet& other) const;
        bool operator<(const CredibleSet& other) const;
        
        // Hash function for unordered_map
        struct Hash {
            std::size_t operator()(const CredibleSet& credible_set) const;
        };
    };

    using CredibleSetHash = CredibleSet::Hash;

} // namespace posg_core 