// (c) 2017 Blai Bonet
// Modified by Florent Teichteil-Koenigsbuch in 2019

#ifndef SIM_PLANNER_H
#define SIM_PLANNER_H

#include <deque>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>
#include <memory>

#include "planner.h"
#include "node.h"
#include "logger.h"
#include "utils.h"

template <typename Environment>
struct SimPlanner : Planner<Environment> {
    const int simulator_budget_;
    const size_t num_tracked_atoms_;

    mutable size_t simulator_calls_;
    mutable double sim_time_;
    mutable double sim_reset_time_;
    mutable double sim_save_environment_time_;

    mutable size_t get_atoms_calls_;
    mutable double get_atoms_time_;
    mutable double novel_atom_time_;
    mutable double update_novelty_time_;

    typename Environment::Observation initial_sim_observation_;
    std::vector<typename Environment::Action> action_set_;

    SimPlanner(Environment &sim,
               size_t frameskip,
               int simulator_budget,
               size_t num_tracked_atoms,
               int debug_threshold,
               int random_seed,
               std::string logger_mode)
      : Planner<Environment>(sim, frameskip, debug_threshold, random_seed, logger_mode),
        simulator_budget_(simulator_budget),
        num_tracked_atoms_(num_tracked_atoms) {
        //static_assert(std::numeric_limits<double>::is_iec559, "IEEE 754 required");
        sim.enumerate_actions([this](const typename Environment::Action& action)->void {action_set_.push_back(action);});
        reset_env();
    }
    virtual ~SimPlanner() { }

    void reset_stats() const {
        simulator_calls_ = 0;
        sim_time_ = 0;
        sim_reset_time_ = 0;
        sim_save_environment_time_ = 0;
        update_novelty_time_ = 0;
        get_atoms_calls_ = 0;
        get_atoms_time_ = 0;
        novel_atom_time_ = 0;
    }

    virtual double simulator_time() const {
        return sim_time_ + sim_reset_time_ + sim_save_environment_time_;
    }
    virtual size_t simulator_calls() const {
        return simulator_calls_;
    }
    virtual typename Environment::Action random_action() const {
        return action_set_[lrand48() % action_set_.size()];
    }

    typename Environment::Action random_zero_value_action(const Node<Environment> *root, double discount) const {
        assert(root != 0);
        assert((root->num_children_ > 0) && (root->first_child_ != nullptr));
        std::vector<typename Environment::Action> zero_value_actions;
        for( Node<Environment> *child = root->first_child_; child != nullptr; child = child->sibling_ ) {
            if( child->qvalue(discount) == 0 )
                zero_value_actions.push_back(child->action_);
        }
        assert(!zero_value_actions.empty());
        return zero_value_actions[lrand48() % zero_value_actions.size()];
    }

    void call_simulator(const typename Environment::Action& action, typename Environment::Observation& observation, double& reward, bool& termination) {
        ++simulator_calls_;
        double start_time = Utils::read_time_in_seconds();
        observation = this->sim_.step(action, reward, termination);
        assert(reward != -std::numeric_limits<double>::infinity());
        sim_time_ += Utils::read_time_in_seconds() - start_time;
    }

    void reset_env() {
        double start_time = Utils::read_time_in_seconds();
        initial_sim_observation_ = this->sim_.reset_env();
        sim_reset_time_ += Utils::read_time_in_seconds() - start_time;
    }

    void save_environment() { // stack version (for rolloutIW)
        double start_time = Utils::read_time_in_seconds();
        this->sim_.save_environment();
        sim_save_environment_time_ += Utils::read_time_in_seconds() - start_time;
    }

    void save_environment(const typename Environment::Observation& observation) { // map version (for bfsIW)
        double start_time = Utils::read_time_in_seconds();
        this->sim_.save_environment(observation);
        sim_save_environment_time_ += Utils::read_time_in_seconds() - start_time;
    }

    void restore_environment() { // stack version (for rolloutIW)
        double start_time = Utils::read_time_in_seconds();
        this->sim_.restore_environment();
        sim_save_environment_time_ += Utils::read_time_in_seconds() - start_time;
    }

    void restore_environment(const typename Environment::Observation& observation) { // map version (for bfsIW)
        double start_time = Utils::read_time_in_seconds();
        this->sim_.restore_environment(observation);
        sim_save_environment_time_ += Utils::read_time_in_seconds() - start_time;
    }

    // update info for node
    void update_info(Node<Environment> *node) {
        assert(node->is_info_valid_ != 2);
        assert(node->observation_ == nullptr);
        assert(node->parent_ != nullptr);
        assert((node->parent_->is_info_valid_ == 1) || (node->parent_->observation_ != nullptr));
        if( node->parent_->observation_ == nullptr ) {
            // do recursion on parent
            update_info(node->parent_);
        }
        assert(node->parent_->observation_ != nullptr);
        double reward;
        bool termination;
        typename Environment::Observation observation;
        call_simulator(node->action_, observation, reward, termination);
        assert(reward != std::numeric_limits<double>::infinity());
        assert(reward != -std::numeric_limits<double>::infinity());
        node->observation_ = std::make_unique<typename Environment::Observation>(observation);
        if( node->is_info_valid_ == 0 ) {
            node->reward_ = reward;
            node->terminal_ = termination;
            get_atoms(node);
            node->path_reward_ = node->parent_ == nullptr ? 0 : node->parent_->path_reward_;
            node->path_reward_ += node->reward_;
        }
        node->is_info_valid_ = 2;
    }

    // get atoms from ram or screen
    void get_atoms(const Node<Environment> *node) const {
        assert(node->feature_atoms_.empty());
        ++get_atoms_calls_;
        double start_time = Utils::read_time_in_seconds();
        this->sim_.convert_observation_to_feature_atoms(*(node->observation_), node->feature_atoms_);
        if( (node->parent_ != nullptr) && (node->parent_->feature_atoms_ == node->feature_atoms_) ) {
            node->frame_rep_ = node->parent_->frame_rep_ + this->frameskip_;
            assert((node->num_children_ == 0) && (node->first_child_ == nullptr));
        }
        get_atoms_time_ += Utils::read_time_in_seconds() - start_time;
    }

    // novelty tables: a (simple) novelty table maps feature indices to best depth at which
    // features have been seen. Best depth is initialized to max.int. Novelty table associated
    // to node is a unique simple table if subtables is disabled. Otherwise, there is one table
    // for each different logscore. The table for a node is the table for its logscore.
    int logscore(double path_reward) const {
        if( path_reward <= 0 ) {
            return 0;
        } else {
            int logr = int(floorf(log2f(path_reward)));
            return path_reward < 1 ? logr : 1 + logr;
        }
    }
    int get_index_for_novelty_table(const Node<Environment> *node, bool use_novelty_subtables) const {
        return !use_novelty_subtables ? 0 : logscore(node->path_reward_);
    }

    std::vector<int>& get_novelty_table(const Node<Environment> *node, std::map<int, std::vector<int> > &novelty_table_map, bool use_novelty_subtables) const {
        int index = get_index_for_novelty_table(node, use_novelty_subtables);
        std::map<int, std::vector<int> >::iterator it = novelty_table_map.find(index);
        if( it == novelty_table_map.end() ) {
            novelty_table_map.insert(std::make_pair(index, std::vector<int>()));
            std::vector<int> &novelty_table = novelty_table_map.at(index);
            novelty_table = std::vector<int>(num_tracked_atoms_, std::numeric_limits<int>::max());
            return novelty_table;
        } else {
            return it->second;
        }
    }

    size_t update_novelty_table(size_t depth, const std::vector<int> &feature_atoms, std::vector<int> &novelty_table) const {
        double start_time = Utils::read_time_in_seconds();
        size_t first_index = 0;
        size_t number_updated_entries = 0;
        for( size_t k = first_index; k < feature_atoms.size(); ++k ) {
            assert((feature_atoms[k] >= 0) && (feature_atoms[k] < int(novelty_table.size())));
            if( int(depth) < novelty_table[feature_atoms[k]] ) {
                novelty_table[feature_atoms[k]] = depth;
                ++number_updated_entries;
            }
        }
        update_novelty_time_ += Utils::read_time_in_seconds() - start_time;
        return number_updated_entries;
    }

    int get_novel_atom(size_t depth, const std::vector<int> &feature_atoms, const std::vector<int> &novelty_table) const {
        double start_time = Utils::read_time_in_seconds();
        for( size_t k = 0; k < feature_atoms.size(); ++k ) {
            assert(feature_atoms[k] < int(novelty_table.size()));
            if( novelty_table[feature_atoms[k]] > int(depth) ) {
                novel_atom_time_ += Utils::read_time_in_seconds() - start_time;
                return feature_atoms[k];
            }
        }
        for( size_t k = 0; k < feature_atoms.size(); ++k ) {
            if( novelty_table[feature_atoms[k]] == int(depth) ) {
                novel_atom_time_ += Utils::read_time_in_seconds() - start_time;
                return feature_atoms[k];
            }
        }
        novel_atom_time_ += Utils::read_time_in_seconds() - start_time;
        assert(novelty_table[feature_atoms[0]] < int(depth));
        return feature_atoms[0];
    }

    size_t num_entries(const std::vector<int> &novelty_table) const {
        assert(novelty_table.size() == num_tracked_atoms_);
        size_t n = 0;
        for( size_t k = 0; k < novelty_table.size(); ++k )
            n += novelty_table[k] < std::numeric_limits<int>::max();
        return n;
    }

    // prefix
    void apply_prefix(const typename Environment::Observation &initial_observation,
                      const std::vector<typename Environment::Action> &prefix,
                      typename Environment::Observation *last_observation = nullptr) {
        assert(!prefix.empty());
        typename Environment::Observation obs;
        double reward;
        bool termination;
        for( size_t k = 0; k < prefix.size(); ++k ) {
            if( (last_observation != nullptr) && (1 + k == prefix.size()) ) {
                *last_observation = obs;
            }
            if (k > 0) { // first action is random and not to be applied
                call_simulator(prefix[k], obs, reward, termination);
            }
        }
    }

    void print_prefix(Logger::mode_t logger_mode, const std::vector<typename Environment::Action> &prefix) const {
        Logger::Continuation(logger_mode) << "[";
        for( size_t k = 0; k < prefix.size(); ++k )
            Logger::Continuation(logger_mode) << prefix[k] << ",";
        Logger::Continuation(logger_mode) << "]" << std::flush;
    }

    // generate states along given branch
    void generate_states_along_branch(Node<Environment> *node,
                                      const std::deque<typename Environment::Action> &branch) {
        for( size_t pos = 0; pos < branch.size(); ++pos ) {
            if( node->observation_ == nullptr ) {
                assert(node->is_info_valid_ == 1);
                update_info(node);
            }

            Node<Environment> *selected = nullptr;
            for( Node<Environment> *child = node->first_child_; child != nullptr; child = child->sibling_ ) {
                if( typename Environment::ActionEqual()(child->action_, branch[pos]) ) {
                    selected = child;
                    break;
                }
            }
            assert(selected != nullptr);
            node = selected;
        }
    }
};

#endif

