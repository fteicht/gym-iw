// (c) 2019 Florent Teichteil-Koenigsbuch

#include <pybind11/pybind11.h>
#include "gym_proxy.h"

namespace py = pybind11;

PYBIND11_MODULE(__gym_iw, giw) {
    py::class_<GymProxy> gym_iw_agent(giw, "__GymIwAgent");
        gym_iw_agent
            .def(py::init<const py::object&, const std::string&, const std::string&,
                 double, size_t, double, double, bool, bool, size_t, double,
                 int, bool, size_t, int, int, const std::string&>(),
                 py::arg("environment"),
                 py::arg("planner"),
                 py::arg("encoding")="byte",
                 py::arg("space_relative_precision")=0.001,
                 py::arg("frameskip")=15,
                 py::arg("simulator_budget")=150000,
                 py::arg("time_budget")=std::numeric_limits<double>::infinity(),
                 py::arg("novelty_subtables")=false,
                 py::arg("random_actions")=false,
                 py::arg("max_rep")=30,
                 py::arg("discount")=1.00,
                 py::arg("nodes_threshold")=50000,
                 py::arg("break_ties_using_rewards")=false,
                 py::arg("max_depth")=1500,
                 py::arg("debug_threshold")=0,
                 py::arg("random_seed")=0,
                 py::arg("logger_mode")="info")
            .def("get_number_of_observation_feature_atoms", &GymProxy::get_number_of_observation_feature_atoms)
            .def("get_number_of_action_feature_atoms", &GymProxy::get_number_of_action_feature_atoms)
            .def("convert_observation_to_feature_atoms", (py::array_t<std::int64_t> (GymProxy::*) (const py::object&)) &GymProxy::convert_observation_to_feature_atoms)
            .def("convert_action_to_feature_atoms", (py::array_t<std::int64_t> (GymProxy::*) (const py::object&)) &GymProxy::convert_action_to_feature_atoms)
            .def("convert_feature_atoms_to_observation", (py::object (GymProxy::*) (const py::array_t<std::int64_t>&)) &GymProxy::convert_feature_atoms_to_observation)
            .def("convert_feature_atoms_to_action", (py::object (GymProxy::*) (const py::array_t<std::int64_t>&)) &GymProxy::convert_feature_atoms_to_action)
            .def("enumerate_observations", &GymProxy::enumerate_observations_conv)
            .def("enumerate_actions", &GymProxy::enumerate_actions_conv)
            .def("play", &GymProxy::play, "Play with the environment by optimizing and running several episodes",
                 py::arg("episodes")=1,
                 py::arg("initial_random_noops")=30,
                 py::arg("lookahead_caching")=2,
                 py::arg("prefix_length_to_execute")=0.0,
                 py::arg("execute_single_action")=false,
                 py::arg("max_execution_length_in_frames")=18000)
            .def("start_episode", &GymProxy::start_episode, py::arg("lookahead_caching")=2)
            .def("act", &GymProxy::act, py::arg("observation"), py::arg("reward"), py::arg("done"))
            .def("end_episode", &GymProxy::end_episode)
            ;
}
