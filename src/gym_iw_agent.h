// (c) 2019 Florent Teichteil

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "gym_spaces.h"

namespace py = pybind11;

class GymIwAgent {
public :
    GymIwAgent(py::object& gym_env, double space_relative_precision = 0.001)
    : gym_env_(gym_env), space_relative_precision_(space_relative_precision) {
        if (!py::hasattr(gym_env, "observation_space") || !py::hasattr(gym_env, "action_space")) {
            py::print("ERROR: Gym environment should have 'observation_space' and 'action_space' attributes");
        } else {
            observation_space_ = GymSpace::import_from_python(gym_env_.attr("observation_space"), space_relative_precision);
            action_space_ = GymSpace::import_from_python(gym_env_.attr("action_space"), space_relative_precision);
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
        std::vector<int> feature_atoms(observation_space_->get_number_of_feature_atoms(), 0);
        observation_space_->convert_element_to_feature_atoms(observation, feature_atoms);
        return py::cast(feature_atoms);
    }

    // export to python
    py::array_t<std::int64_t> convert_action_to_feature_atoms(const py::object& action) {
        std::vector<int> feature_atoms(action_space_->get_number_of_feature_atoms(), 0);
        action_space_->convert_element_to_feature_atoms(action, feature_atoms);
        return py::cast(feature_atoms);
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
    py::object act(const py::object& observation, double reward, bool done) {
        try {
            std::vector<int> feature_atoms(observation_space_->get_number_of_feature_atoms(), 0);
            observation_space_->convert_element_to_feature_atoms(observation, feature_atoms);
            std::vector<int> action; // TODO: link with planner
            return action_space_->convert_feature_atoms_to_element(action);
        } catch (const std::exception& e) {
            py::print("Python binding error: " + std::string(e.what()));
            return py::object();
        }
    }

    // import from python
    std::vector<int> reset_env() {
        try {
            if (!py::hasattr(gym_env_, "reset")) {
                py::print("ERROR: Gym env without 'reset' method");
                return std::vector<int>();
            } else {
                py::object observation = gym_env_.attr("reset")();
                std::vector<int> feature_atoms(observation_space_->get_number_of_feature_atoms(), 0);
                observation_space_->convert_element_to_feature_atoms(observation, feature_atoms);
                return feature_atoms;
            }
        } catch (const std::exception& e) {
            py::print("Python binding error: " + std::string(e.what()));
            return std::vector<int>();
        }
    }

    // import from python
    std::vector<int> step(const std::vector<int>& action, double& reward, bool& termination) {
        try {
            if (!py::hasattr(gym_env_, "reset")) {
                py::print("ERROR: Gym env without 'step' method");
                return std::vector<int>();
            } else {
                py::tuple step_return = gym_env_.attr("step")(action_space_->convert_feature_atoms_to_element(action));
                std::vector<int> feature_atoms(observation_space_->get_number_of_feature_atoms(), 0);
                observation_space_->convert_element_to_feature_atoms(step_return[0], feature_atoms);
                if (!py::isinstance<py::float_>(step_return[1])) {
                    py::print("ERROR: Gym env's step method's returned tuple's second element not of type 'float'");
                    return std::vector<int>();
                }
                reward = py::cast<double>(step_return[1]);
                if (!py::isinstance<py::bool_>(step_return[2])) {
                    py::print("ERROR: Gym env's step method's returned tuple's third element not of type 'bool'");
                    return std::vector<int>();
                }
                termination = py::cast<double>(step_return[2]);
                return feature_atoms;
            }
        } catch (const std::exception& e) {
            py::print("Python binding error: " + std::string(e.what()));
            return std::vector<int>();
        }
    }

private :
    py::object& gym_env_;
    double space_relative_precision_;
    std::unique_ptr<GymSpace> observation_space_;
    std::unique_ptr<GymSpace> action_space_;
};