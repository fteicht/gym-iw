// (c) 2017 Blai Bonet

#ifndef PLANNER_H
#define PLANNER_H

#include <deque>
#include <string>
#include <vector>
#include "node.h"
#include "logger.h"

template <typename Environment>
struct Planner {
    Environment &sim_;
    const size_t frameskip_;
    int random_seed_;
    int debug_threshold_;

    double acc_simulator_time_;
    double acc_reward_;
    size_t acc_simulator_calls_;
    size_t max_simulator_calls_;
    size_t acc_decisions_;
    size_t acc_frames_;
    size_t acc_random_decisions_;
    size_t acc_height_;
    size_t acc_expanded_;

    Planner(Environment &sim,
            size_t frameskip,
            int debug_threshold,
            int random_seed,
            std::string logger_mode) 
        : sim_(sim),
          frameskip_(frameskip),
          debug_threshold_(debug_threshold) {
        // set random seed for lrand48() and drand48()
        srand48(random_seed);

        // set logger mode and log mode
        Logger::mode_t lg = Logger::Silent;
        if( logger_mode == "debug" ) {
            lg = Logger::Debug;
        } else if( logger_mode == "info" ) {
            lg = Logger::Info;
        } else if( logger_mode == "warning" ) {
            lg = Logger::Warning;
        } else if( logger_mode == "error" ) {
            lg = Logger::Error;
        } else if( logger_mode == "stats" ) {
            lg = Logger::Stats;
        } else if( logger_mode == "silent" ) {
            lg = Logger::Silent;
        } else {
            Logger::Error << "invalid logger mode '" << logger_mode << "'" << std::endl;
            exit(-1);
        }
        Logger::set_mode(lg);
        Logger::set_debug_threshold(debug_threshold);
    }

    virtual ~Planner() { }
    virtual std::string name() const = 0;
    virtual std::string str() const = 0;
    virtual double simulator_time() const = 0;
    virtual size_t simulator_calls() const = 0;
    virtual bool random_decision() const = 0;
    virtual size_t height() const = 0;
    virtual size_t expanded() const = 0;
    virtual typename Environment::Action random_action() const = 0;
    virtual Node<Environment>* get_branch(const std::vector<typename Environment::Action> &prefix,
                                          Node<Environment> *root,
                                          double last_reward,
                                          std::deque<typename Environment::Action> &branch) = 0;
    
    void run_episode(int initial_noops,
                     int lookahead_caching,
                     double prefix_length_to_execute,
                     bool execute_single_action,
                     size_t max_execution_length_in_frames,
                     std::vector<typename Environment::Action> &prefix) {
        assert(prefix.empty());
        reset_stat_variables();

        // fill initial prefix
        if( initial_noops == 0 ) {
            prefix.push_back(random_action());
        } else {
            assert(initial_noops > 0);
            while( initial_noops-- > 0 )
                prefix.push_back(typename Environment::Action());
        }

        // reset simulator
        typename Environment::Observation current_observation = sim_.reset_env(true);

        // execute actions in initial prefix
        double last_reward = std::numeric_limits<double>::infinity();
        bool termination;
        for( size_t k = 0; k < prefix.size(); ++k ) {
            current_observation = sim_.step(prefix[k], last_reward, termination);
            acc_reward_ += last_reward;
        }
        assert(last_reward != std::numeric_limits<double>::infinity());

        // play
        Node<Environment> *node = nullptr;
        std::deque<typename Environment::Action> branch;
        for( size_t frame = 0; !termination && (frame < max_execution_length_in_frames); frame += frameskip_ ) {
            Logger::Debug << "Episode frame: " << frame << std::endl;

            // if empty branch, get branch
            if( branch.empty() ) {
                ++acc_decisions_;

                if( (node != nullptr) && (lookahead_caching == 1) ) {
                    node->clear_cached_observations();
                    assert(node->observation_ == nullptr);
                    assert(node->parent_ != nullptr);
                    assert(node->parent_->observation_ != nullptr);
                }

                // Save current observation
                sim_.save_observation(current_observation);

                node = get_branch(prefix, node, last_reward, branch);
                acc_simulator_time_ += simulator_time();
                acc_simulator_calls_ += simulator_calls();
                max_simulator_calls_ = std::max(max_simulator_calls_, simulator_calls());
                acc_random_decisions_ += random_decision() ? 1 : 0;
                acc_height_ += height();
                acc_expanded_ += expanded();

                if( branch.empty() ) {
                    Logger::Error << "no more available actions!" << std::endl;
                    break;
                }

                // calculate executable prefix
                assert(prefix_length_to_execute >= 0);
                if( !execute_single_action && (prefix_length_to_execute > 0) ) {
                    int target_branch_length = int(branch.size() * prefix_length_to_execute);
                    if( target_branch_length == 0 ) target_branch_length = 1;
                    assert(target_branch_length > 0);
                    while( int(branch.size()) > target_branch_length )
                        branch.pop_back();
                    assert(target_branch_length == int(branch.size()));
                } else if( execute_single_action ) {
                    assert(branch.size() >= 1);
                    while( branch.size() > 1 )
                        branch.pop_back();
                }

                Logger::Info << "executable-prefix: len=" << branch.size() << ", actions=[";
                for( size_t j = 0; j < branch.size(); ++j )
                    Logger::Continuation(Logger::Info) << branch[j] << ",";
                Logger::Continuation(Logger::Info) << "]" << std::endl;
            }

            // select action to apply
            typename Environment::Action action = branch.front();
            branch.pop_front();

            // restore saved observation
            sim_.restore_observation(current_observation);

            // apply action
            current_observation = sim_.step(action, last_reward, termination);
            prefix.push_back(action);
            acc_reward_ += last_reward;
            acc_frames_ += frameskip_;
            Logger::Stats << "step-stats: acc-reward=" << acc_reward_ << ", acc-frames=" << acc_frames_ << std::endl;

            // advance/destroy lookhead tree
            if( node != nullptr ) {
                if( (lookahead_caching == 0) || (node->num_children_ == 0) ) {
                    remove_tree(node);
                    node = nullptr;
                } else {
                    assert(node->parent_->observation_ != nullptr);
                    node = node->advance(action);
                }
            }

            // prune branch if got positive reward
            if( execute_single_action || ((prefix_length_to_execute == 0) && (last_reward > 0)) )
                branch.clear();
        }

        // cleanup
        if( node != nullptr ) {
            assert(node->parent_ != nullptr);
            assert(node->parent_->parent_ == nullptr);
            remove_tree(node->parent_);
        }
    }

    void reset_stat_variables() {
        acc_simulator_time_ = 0;
        acc_reward_ = 0;
        acc_simulator_calls_ = 0;
        max_simulator_calls_ = 0;
        acc_decisions_ = 0;
        acc_frames_ = 0;
        acc_random_decisions_ = 0;
        acc_height_ = 0;
        acc_expanded_ = 0;
    }

    void play(int episodes = 1,
              int initial_random_noops = 30,
              int lookahead_caching = 2,
              double prefix_length_to_execute = 0.0,
              bool execute_single_action = false,
              int max_execution_length_in_frames = 18000) {
        Logger::Info << "planner=" << name() << std::endl;

        // set number of initial noops
        assert(initial_random_noops > 0);
        int initial_noops = lrand48() % initial_random_noops;

        // play
        for( int k = 0; k < episodes; ++k ) {
            std::vector<typename Environment::Action> prefix;
            double start_time = Utils::read_time_in_seconds();
            run_episode(initial_noops, lookahead_caching, prefix_length_to_execute, execute_single_action, max_execution_length_in_frames, prefix);
            double elapsed_time = Utils::read_time_in_seconds() - start_time;
            log_stats();
            Logger::Stats
            << "episode-stats:"
            // planner
            << " planner=" << str()
            // general options
            << " debug-threshold=" << debug_threshold_
            << " frameskip=" << frameskip_
            << " seed=" << random_seed_
            // episodes and execution length
            << " episodes=" << episodes
            << " max-execution-length=" << max_execution_length_in_frames
            // online execution
            << " initial-noops=" << initial_random_noops
            << " execute-single-action=" << execute_single_action
            << " caching=" << lookahead_caching
            << " prefix-length-to-execute=" << prefix_length_to_execute;
            // planner-specific
            log_stats();
            Logger::Stats
            // data
            << " score=" << acc_reward_
            << " frames=" << acc_frames_
            << " decisions=" << acc_decisions_
            << " simulator-calls=" << acc_simulator_calls_
            << " max-simulator-calls=" << max_simulator_calls_
            << " total-time=" << elapsed_time
            << " simulator-time=" << acc_simulator_time_
            << " sum-expanded=" << acc_expanded_
            << " sum-height=" << acc_height_
            << " random-decisions=" << acc_random_decisions_
            << std::endl;
        }
    }

    virtual void log_stats() const =0;
};

template <typename Environment>
struct RandomPlanner : Planner<Environment> {
    std::vector<typename Environment::Action> action_set_;
    size_t action_set_size_;

    RandomPlanner(Environment &sim,
                  size_t frameskip,
                  int debug_threshold,
                  int random_seed,
                  std::string logger_mode)
      : Planner<Environment>(sim, frameskip, debug_threshold, random_seed, logger_mode) {
        sim.enumerate_actions([this](const typename Environment::Action& action)->void {action_set_.push_back(action);});
        action_set_size_ = action_set_.size();
    }
    virtual ~RandomPlanner() { }

    virtual std::string name() const {
        return std::string("random()");
    }
    virtual std::string str() const {
        return std::string("random");
    }
    virtual double simulator_time() const {
        return 0;
    }
    virtual size_t simulator_calls() const {
        return 0;
    }
    virtual bool random_decision() const {
        return true;
    }
    virtual size_t height() const {
        return 0;
    }
    virtual size_t expanded() const {
        return 0;
    }

    virtual typename Environment::Action random_action() const {
        return action_set_[lrand48() % action_set_size_];
    }

    virtual Node<Environment>* get_branch(const std::vector<typename Environment::Action> &prefix,
                                          Node<Environment> *root,
                                          double last_reward,
                                          std::deque<typename Environment::Action> &branch) {
        assert(branch.empty());
        branch.push_back(random_action());
        return nullptr;
    }

    virtual void log_stats() const {}
};

template <typename Environment>
struct FixedPlanner : public Planner<Environment> {
    mutable std::deque<typename Environment::Action> actions_;

    FixedPlanner(Environment &sim,
                 size_t frameskip,
                 const std::vector<typename Environment::Action> &actions,
                 int debug_threshold,
                 int random_seed,
                 std::string logger_mode)
      : Planner<Environment>(sim, frameskip, debug_threshold, random_seed, logger_mode) {
        actions_ = std::deque<typename Environment::Action>(actions.begin(), actions.end());
    }
    virtual ~FixedPlanner() { }

    typename Environment::Action pop_first_action() const {
        typename Environment::Action action = actions_.front();
        actions_.pop_front();
        return action;
    }

    virtual std::string name() const {
        return std::string("fixed(sz=") + std::to_string(actions_.size()) + ")";
    }
    virtual std::string str() const {
        return std::string("fixed");
    }
    virtual double simulator_time() const {
        return 0;
    }
    virtual size_t simulator_calls() const {
        return 0;
    }
    virtual bool random_decision() const {
        return true;
    }
    virtual size_t height() const {
        return 0;
    }
    virtual size_t expanded() const {
        return 0;
    }

    virtual typename Environment::Action random_action() const {
        assert(!actions_.empty());
        return pop_first_action();
    }

    virtual Node<Environment>* get_branch(const std::vector<typename Environment::Action> &prefix,
                                          Node<Environment> *root,
                                          double last_reward,
                                          std::deque<typename Environment::Action> &branch) {
        assert(branch.empty());
        if( !actions_.empty() ) {
            typename Environment::Action action = pop_first_action();
            branch.push_back(action);
        }
        return nullptr;
    }

    virtual void log_stats() const {}
};

#endif

