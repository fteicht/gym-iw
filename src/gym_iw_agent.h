// (c) 2019 Florent Teichteil

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include "gym_spaces.h"

namespace py = pybind11;

class GymIwAgent {
public :
    GymIwAgent(py::object& gym_env, const std::string& encoding, double space_relative_precision = 0.001)
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
    py::object act(const py::object& observation, double reward, bool done) {
        try {
            std::vector<int> observation_encoding = observation_space_->convert_element_to_feature_atoms(observation);
            std::vector<int> action; // TODO: link with planner
            return action_space_->convert_feature_atoms_to_element(action);
        } catch (const std::exception& e) {
            py::print("Python binding error: " + std::string(e.what()));
            return py::object();
        }
    }

    // export to python
    void enumerate_observations(const std::function<void(const py::array_t<std::int64_t>&)>& f) {
        observation_space_->enumerate([&f](const std::vector<int>& v)->void {
            f(py::cast(v));
        });
    }

    // export to python
    void enumerate_actions(const std::function<void(const py::array_t<std::int64_t>&)>& f) {
        action_space_->enumerate([&f](const std::vector<int>& v)->void {
            f(py::cast(v));
        });
    }

    // import from python
    std::vector<int> reset_env() {
        try {
            if (!py::hasattr(gym_env_, "reset")) {
                py::print("ERROR: Gym env without 'reset' method");
                return std::vector<int>();
            } else {
                py::object observation = gym_env_.attr("reset")();
                return observation_space_->convert_element_to_feature_atoms(observation);
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
                std::vector<int> observation_encoding = observation_space_->convert_element_to_feature_atoms(step_return[0]);
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
                return observation_encoding;
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