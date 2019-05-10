// (c) 2019 Florent Teichteil

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include "gym_spaces.h"
#include "bfsIW.h"
#include "rolloutIW.h"

namespace py = pybind11;

class GymProxy {
public :
    typedef py::object Observation;
    typedef py::object Action;
    
    GymProxy(const py::object& gym_env, const std::string& planner, const std::string& encoding,
             double space_relative_precision = 0.001,
             size_t frameskip = 15,
             double simulator_budget = 150000,
             double time_budget = std::numeric_limits<double>::infinity(),
             bool novelty_subtables = false,
             bool random_actions = false,
             size_t max_rep = 30,
             double discount = 1.00,
             int nodes_threshold = 50000,
             bool break_ties_using_rewards = false,
             size_t max_depth = 1500,
             int debug_threshold = 0,
             int random_seed = 0,
             std::string logger_mode = "info")
    : gym_env_(gym_env), space_relative_precision_(space_relative_precision) {
        if (!py::hasattr(gym_env, "observation_space") || !py::hasattr(gym_env, "action_space")) {
            py::print("ERROR: Gym environment should have 'observation_space' and 'action_space' attributes");
        } else {
            if (encoding == "byte") {
                observation_space_ = GymSpace::import_from_python(gym_env_.attr("observation_space"), GymSpace::ENCODING_BYTE_VECTOR, space_relative_precision);
                action_space_ = GymSpace::import_from_python(gym_env_.attr("action_space"), GymSpace::ENCODING_BYTE_VECTOR, space_relative_precision);
            } else if (encoding == "variable") {
                observation_space_ = GymSpace::import_from_python(gym_env_.attr("observation_space"), GymSpace::ENCODING_VARIABLE_VECTOR, space_relative_precision);
                action_space_ = GymSpace::import_from_python(gym_env_.attr("action_space"), GymSpace::ENCODING_VARIABLE_VECTOR, space_relative_precision);
            } else {
                py::print("ERROR: unsupported feature atom vector encoding '" + encoding + "'");
            }
            if (planner == "bfs-iw") {
                planner_ = std::make_unique<BfsIW<GymProxy>>(*this, frameskip,
                                                             observation_space_->get_number_of_tracked_atoms(), simulator_budget,
                                                             time_budget, novelty_subtables, random_actions, max_rep,
                                                             discount, nodes_threshold, break_ties_using_rewards,
                                                             debug_threshold, random_seed, logger_mode);
            } else if (planner == "rollout-iw") {
                planner_ = std::make_unique<RolloutIW<GymProxy>>(*this, frameskip,
                                                                 observation_space_->get_number_of_tracked_atoms(), simulator_budget,
                                                                 time_budget, novelty_subtables, random_actions, max_rep,
                                                                 discount, nodes_threshold, max_depth,
                                                                 debug_threshold, random_seed, logger_mode);
            } else {
                py::print("ERROR: unsupported planner '" + planner + "'");
            }
        }
    }

    // export to python
    unsigned int get_number_of_observation_feature_atoms() const {
        return observation_space_->get_number_of_feature_atoms();
    }

    // export to python
    unsigned int get_number_of_action_feature_atoms() const {
        return action_space_->get_number_of_feature_atoms();
    }

    // export to python
    py::array_t<std::int64_t> convert_observation_to_feature_atoms(const py::object& observation) {
        return py::cast(observation_space_->convert_element_to_feature_atoms(observation));
    }

    // export to python
    py::array_t<std::int64_t> convert_action_to_feature_atoms(const py::object& action) {
        return py::cast(action_space_->convert_element_to_feature_atoms(action));
    }

    // export to python
    py::object convert_feature_atoms_to_observation(const py::array_t<std::int64_t>& feature_atoms) {
        return observation_space_->convert_feature_atoms_to_element(py::cast<std::vector<int>>(feature_atoms));
    }

    // export to python
    py::object convert_feature_atoms_to_action(const py::array_t<std::int64_t>& feature_atoms) {
        return action_space_->convert_feature_atoms_to_element(py::cast<std::vector<int>>(feature_atoms));
    }

    // export to python
    void enumerate_observations_conv(const std::function<void(const py::array_t<std::int64_t>&)>& f) {
        observation_space_->enumerate([&f](const std::vector<int>& v)->void {
            f(py::cast(v));
        });
    }

    // export to python
    void enumerate_actions_conv(const std::function<void(const py::array_t<std::int64_t>&)>& f) {
        action_space_->enumerate([&f](const std::vector<int>& v)->void {
            f(py::cast(v));
        });
    }

    // export to python
    void play(int episodes = 1,
              int initial_random_noops = 30,
              int lookahead_caching = 2,
              double prefix_length_to_execute = 0.0,
              bool execute_single_action = false,
              int max_execution_length_in_frames = 18000) {
        try {
            planner_->play(episodes, initial_random_noops, lookahead_caching,
                           prefix_length_to_execute, execute_single_action, max_execution_length_in_frames);
        } catch (const std::exception& e) {
            py::print("Python binding error: " + std::string(e.what()));
        }
    }

    // export to python
    void start_episode(int lookahead_caching = 2) {
        try {
            planner_->start_episode(lookahead_caching);
        } catch (const std::exception& e) {
            py::print("Python binding error: " + std::string(e.what()));
        }
    }

    // export to python
    py::object act(const py::object& observation, double reward, bool done) {
        try {
            return planner_->act(observation, reward, done);
        } catch (const std::exception& e) {
            py::print("Python binding error: " + std::string(e.what()));
            return py::object();
        }
    }

    // export to python
    void end_episode() {
        try {
            planner_->end_episode();
        } catch (const std::exception& e) {
            py::print("Python binding error: " + std::string(e.what()));
        }
    }

    // import from python
    py::object reset_env(bool clear_saved_environments = false) {
        try {
            if (!py::hasattr(gym_env_, "reset")) {
                py::print("ERROR: Gym env without 'reset' method");
                return py::none();
            } else {
                py::object observation = gym_env_.attr("reset")();
                if (clear_saved_environments)
                    saved_environments_.clear();
                saved_environments_.insert(std::make_pair(observation, gym_env_));
                last_saved_observation_ = observation;
                return observation;
            }
        } catch (const std::exception& e) {
            py::print("Python binding error: " + std::string(e.what()));
            return py::none();
        }
    }

    std::vector<int> reset_env_conv() {
        return observation_space_->convert_element_to_feature_atoms(reset_env());
    }

    // import from python
    py::object step(const py::object& action, double& reward, bool& termination) {
        try {
            if (!py::hasattr(gym_env_, "reset")) {
                py::print("ERROR: Gym env without 'step' method");
                return py::none();
            } else {
                py::tuple step_return = gym_env_.attr("step")(action);
                if (!py::isinstance<py::float_>(step_return[1])) {
                    py::print("ERROR: Gym env's step method's returned tuple's second element not of type 'float'");
                    return py::none();
                }
                reward = py::cast<double>(step_return[1]);
                if (!py::isinstance<py::bool_>(step_return[2])) {
                    py::print("ERROR: Gym env's step method's returned tuple's third element not of type 'bool'");
                    return py::none();
                }
                termination = py::cast<double>(step_return[2]);
                return step_return[0];
            }
        } catch (const std::exception& e) {
            py::print("Python binding error: " + std::string(e.what()));
            return py::none();
        }
    }

    std::vector<int> step_conv(const std::vector<int>& action, double& reward, bool& termination) {
        return observation_space_->convert_element_to_feature_atoms(step(action_space_->convert_feature_atoms_to_element(action), reward, termination));
    }

    void convert_observation_to_feature_atoms(const py::object& observation, std::vector<int>& feature_atoms) {
        feature_atoms = observation_space_->convert_element_to_feature_atoms(observation);
    }

    void convert_action_to_feature_atoms(const py::object& action, std::vector<int>& feature_atoms) {
        feature_atoms = action_space_->convert_element_to_feature_atoms(action);
    }

    void convert_feature_atoms_to_observation(const std::vector<int>& feature_atoms, py::object& observation) {
        observation = observation_space_->convert_feature_atoms_to_element(feature_atoms);
    }

    void convert_feature_atoms_to_action(const std::vector<int>& feature_atoms, py::object& action) {
        action = action_space_->convert_feature_atoms_to_element(feature_atoms);
    }

    void enumerate_observations(const std::function<void(const py::object&)>& f) {
        observation_space_->enumerate([&f, this](const std::vector<int>& v)->void {
            f(observation_space_->convert_feature_atoms_to_element(v));
        });
    }

    void enumerate_actions(const std::function<void(const py::object&)>& f) {
        action_space_->enumerate([&f, this](const std::vector<int>& v)->void {
            f(action_space_->convert_feature_atoms_to_element(v));
        });
    }

    void save_observation(const py::object& observation) {
        try {
            if (saved_environments_.find(observation) == saved_environments_.end()) {
                py::object copy = py::module::import("copy").attr("copy");
                saved_environments_[observation] = gym_env_;
                gym_env_ = copy(gym_env_);
                last_saved_observation_ = observation;
            }
        } catch (const std::exception& e) {
            py::print("Python binding error: " + std::string(e.what()));
        }
    }

    void restore_observation(const py::object& observation) {
        try {
            if (!ObservationEqual()(observation, last_saved_observation_)) {
                auto i = saved_environments_.find(observation);
                if (i == saved_environments_.end()) {
                    py::print("ERROR: no such observation to restore");
                    return;
                }
                gym_env_ = i->second;
                last_saved_observation_ = observation;
            }
        } catch (const std::exception& e) {
            py::print("Python binding error: " + std::string(e.what()));
        }
    }

    struct ActionEqual {
        bool operator()(const py::object& a, const py::object& b) const {
            return a.is(b); // actions share the same underlying python object pointers (since generated from the same action vector)
        }
    };

    struct ActionLess {
        bool operator()(const py::object& a, const py::object& b) const {
            return a.ptr() < b.ptr(); // actions share the same underlying python object pointers (since generated from the same action vector)
        }
    };

    struct ObservationEqual {
        bool operator()(const py::object& a, const py::object& b) const {
            return a.is(b);
        }
    };

    struct ObservationLess {
        bool operator()(const py::object& a, const py::object& b) const {
            return a.ptr() < b.ptr();
        }
    };

private :
    py::object gym_env_;
    double space_relative_precision_;
    std::unique_ptr<GymSpace> observation_space_;
    std::unique_ptr<GymSpace> action_space_;
    std::map<py::object, py::object, ObservationLess> saved_environments_; // used by BFS algorithm to backtrack to previously observed observations (saved in nodes) in GYM (no other way than copying the cloning the environment)
    py::object last_saved_observation_;
    std::unique_ptr<SimPlanner<GymProxy>> planner_;
};