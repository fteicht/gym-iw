#include <pybind11/pybind11.h>
#include "gym_iw_agent.h"

namespace py = pybind11;

PYBIND11_MODULE(gym_iw, giw) {
    py::class_<GymIwAgent> bfs_iw_agent(giw, "BfsIwAgent");
        bfs_iw_agent
            .def(py::init<py::object&>())
            .def("get_number_of_observation_feature_atoms", &GymIwAgent::get_number_of_observation_feature_atoms)
            .def("get_number_of_action_feature_atoms", &GymIwAgent::get_number_of_action_feature_atoms)
            .def("convert_observation_to_feature_atoms", &GymIwAgent::convert_observation_to_feature_atoms)
            .def("convert_action_to_feature_atoms", &GymIwAgent::convert_action_to_feature_atoms)
            .def("convert_feature_atoms_to_observation", &GymIwAgent::convert_feature_atoms_to_observation)
            .def("convert_feature_atoms_to_action", &GymIwAgent::convert_feature_atoms_to_action)
            //.def("act", &GymIwAgent::act)
            ;
}
