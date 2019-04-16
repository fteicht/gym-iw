// (c) 2019 Florent Teichteil

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "gym_spaces.h"

namespace py = pybind11;

class GymIwAgent {
public :
    GymIwAgent(py::object& gym_env, double space_relative_precision = 0.001)
    : gym_env_(gym_env), space_relative_precision_(space_relative_precision) {
        observation_space_ = GymSpace::import_from_python(gym_env_.attr("observation_space"), space_relative_precision);
        action_space_ = GymSpace::import_from_python(gym_env_.attr("action_space"), space_relative_precision);
    }

    unsigned int get_number_of_observation_feature_atoms() const {
        return observation_space_->get_number_of_feature_atoms();
    }

    unsigned int get_number_of_action_feature_atoms() const {
        return action_space_->get_number_of_feature_atoms();
    }

    py::array_t<std::int64_t> convert_observation_to_feature_atoms(const py::object& element) {
        std::vector<int> feature_atoms(observation_space_->get_number_of_feature_atoms(), 0);
        observation_space_->convert_element_to_feature_atoms(element, feature_atoms);
        return py::cast(feature_atoms);
    }

    py::array_t<std::int64_t> convert_action_to_feature_atoms(const py::object& element) {
        std::vector<int> feature_atoms(action_space_->get_number_of_feature_atoms());
        action_space_->convert_element_to_feature_atoms(element, feature_atoms);
        return py::cast(feature_atoms);
    }

    // py::object act(py::object& observation, double reward, bool done) {
        
    // }

private :
    py::object& gym_env_;
    double space_relative_precision_;
    std::unique_ptr<GymSpace> observation_space_;
    std::unique_ptr<GymSpace> action_space_;
};