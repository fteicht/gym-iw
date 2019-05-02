// (c) 2017 Blai Bonet

#ifndef PLANNER_H
#define PLANNER_H

#include <deque>
#include <string>
#include <vector>
#include "node.h"

template <typename Environment>
struct Planner {
    Planner() { }
    virtual ~Planner() { }
    virtual std::string name() const = 0;
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
};

template <typename Environment>
struct RandomPlanner : Planner<Environment> {
    std::vector<typename Environment::Action> action_set_;
    size_t action_set_size_;

    RandomPlanner(Environment &env)
      : Planner<Environment>() {
        env.enumerate_actions([this](const typename Environment::Action& action)->void {action_set_.push_back(action);});
        action_set_size_ = action_set_.size();
    }
    virtual ~RandomPlanner() { }

    virtual std::string name() const {
        return std::string("random()");
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

    virtual Node<Environment>* get_branch(Environment &env,
                                          const std::vector<typename Environment::Action> &prefix,
                                          Node<Environment> *root,
                                          double last_reward,
                                          std::deque<typename Environment::Action> &branch) {
        assert(branch.empty());
        branch.push_back(random_action());
        return nullptr;
    }
};

template <typename Environment>
struct FixedPlanner : public Planner<Environment> {
    mutable std::deque<typename Environment::Action> actions_;

    FixedPlanner(const std::vector<typename Environment::Action> &actions)
      : Planner<Environment>() {
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

    virtual Node<Environment>* get_branch(Environment &env,
                                       const std::vector<typename Environment::Action> &prefix,
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
};

#endif

