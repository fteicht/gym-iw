// (c) 2019 Florent Teichteil

#include "gym_spaces.h"
#include <algorithm>

std::unique_ptr<GymSpace> GymSpace::import_from_python(const py::object& gym_space, double space_relative_precision, unsigned int feature_atom_vector_begin) {
    std::string space = py::str(gym_space.attr("__class__").attr("__name__"));
    
    if (space == "Box") {
        if (!py::hasattr(gym_space, "dtype")) {
            py::print("ERROR: Gym box space missing attribute 'dtype'");
            return std::unique_ptr<GymSpace>();
        }
        std::string dtype = py::str(gym_space.attr("dtype"));
        if (dtype == "bool_")
            return BoxSpace<bool>::import_from_python(gym_space, 0, feature_atom_vector_begin);
        else if (dtype == "int_")
            return BoxSpace<long int>::import_from_python(gym_space, 0, feature_atom_vector_begin);
        else if (dtype == "intc")
            return BoxSpace<int>::import_from_python(gym_space, 0, feature_atom_vector_begin);
        else if (dtype == "intp")
            return BoxSpace<std::size_t>::import_from_python(gym_space, 0, feature_atom_vector_begin);
        else if (dtype == "int8")
            return BoxSpace<std::int8_t>::import_from_python(gym_space, 0, feature_atom_vector_begin);
        else if (dtype == "int16")
            return BoxSpace<std::int16_t>::import_from_python(gym_space, 0, feature_atom_vector_begin);
        else if (dtype == "int32")
            return BoxSpace<std::int32_t>::import_from_python(gym_space, 0, feature_atom_vector_begin);
        else if (dtype == "int64")
            return BoxSpace<std::int64_t>::import_from_python(gym_space, 0, feature_atom_vector_begin);
        else if (dtype == "uint8")
            return BoxSpace<std::uint8_t>::import_from_python(gym_space, 0, feature_atom_vector_begin);
        else if (dtype == "uint16")
            return BoxSpace<std::uint16_t>::import_from_python(gym_space, 0, feature_atom_vector_begin);
        else if (dtype == "uint32")
            return BoxSpace<std::uint32_t>::import_from_python(gym_space, 0, feature_atom_vector_begin);
        else if (dtype == "uint64")
            return BoxSpace<std::uint64_t>::import_from_python(gym_space, 0, feature_atom_vector_begin);
        else if (dtype == "float_")
            return BoxSpace<double>::import_from_python(gym_space, space_relative_precision, feature_atom_vector_begin);
        else if (dtype == "float32")
            return BoxSpace<float>::import_from_python(gym_space, space_relative_precision, feature_atom_vector_begin);
        else if (dtype == "float64")
            return BoxSpace<double>::import_from_python(gym_space, space_relative_precision, feature_atom_vector_begin);
        else {
            py::print("ERROR: Unhandled array dtype '" + dtype + "' when importing Gym box space bounds");
            return std::unique_ptr<GymSpace>();
        }
    } else if (space == "Dict") {
        return DictSpace::import_from_python(gym_space, space_relative_precision, feature_atom_vector_begin);
    } else if (space == "Discrete") {
        return DiscreteSpace::import_from_python(gym_space, feature_atom_vector_begin);
    } else if (space == "MultiBinary") {
        return MultiBinarySpace::import_from_python(gym_space, feature_atom_vector_begin);
    } else if (space == "MultiDiscrete") {
        return MultiDiscreteSpace::import_from_python(gym_space, feature_atom_vector_begin);
    } else if (space == "Tuple") {
        return TupleSpace::import_from_python(gym_space, space_relative_precision, feature_atom_vector_begin);
    } else {
        py::print("ERROR: Unhandled Gym space '" + space + "'");
        return std::unique_ptr<GymSpace>();
    }
}


template <typename T>
BoxSpace<T>::BoxSpace(const py::array_t<T>& low, const py::array_t<T>& high, double space_relative_precision, unsigned int feature_atom_vector_begin)
try : GymSpace(feature_atom_vector_begin), low_(low), high_(high), space_relative_precision_(space_relative_precision) {
    if (low_.ndim() != high_.ndim()) {
        throw std::domain_error("Gym box space's 'low' and 'high' arrays not of the same dimension");
    }
    for (unsigned int d = 0 ; d < low_.ndim() ; d++) {
        if (low_.shape(d) != high_.shape(d)) {
            throw std::domain_error("Gym box space's 'low' and 'high' arrays' dimension " + std::to_string(d) + " not of the same size");
        }
    }
    number_of_feature_atoms_ = low_.size();
} catch (const std::exception& e) {
    throw e;
}


template <typename T>
std::unique_ptr<GymSpace> BoxSpace<T>::import_from_python(const py::object& gym_space, double space_relative_precision, unsigned int feature_atom_vector_begin) {
    try {
        if (!py::hasattr(gym_space, "low") || !py::hasattr(gym_space, "high")) {
            py::print("ERROR: Gym box space missing attributes 'low' or 'high'");
            return std::unique_ptr<GymSpace>();
        }
        if (!py::isinstance<py::array_t<T>>(gym_space.attr("low")) || !py::isinstance<py::array_t<T>>(gym_space.attr("high"))) {
            py::print("ERROR: Gym box space's attributes 'low' or 'high' not numpy arrays");
            return std::unique_ptr<GymSpace>();
        }
        return std::make_unique<BoxSpace<T>>(py::cast<py::array_t<T>>(gym_space.attr("low")), py::cast<py::array_t<T>>(gym_space.attr("high")), space_relative_precision, feature_atom_vector_begin);
    } catch (const py::cast_error& e) {
        py::print("ERROR: Python casting error (python object of unexpected type) when importing Gym box space");
        return std::unique_ptr<GymSpace>();
    } catch (const std::exception& e) {
        py::print("ERROR: " + std::string(e.what()));
        return std::unique_ptr<GymSpace>();
    }
}


template <typename T>
void BoxSpace<T>::convert_element_to_feature_atoms_int(const py::object& element, std::vector<int>& feature_atoms) const {
    try {
        if (!py::isinstance<py::array_t<T>>(element)) {
            py::print("ERROR: Gym box space element not a numpy array or missing explicit 'dtype' matching that of 'low' amd 'high' arrays");
            return;
        }
        py::buffer_info buf = py::cast<py::array_t<T>>(element).request();
        if (low_.size() != buf.size) {
            py::print("ERROR: Gym box space element and 'low' array not of the same size");
            return;
        }
        for (unsigned int i = 0 ; i < buf.size ; i++) {
            feature_atoms[feature_atom_vector_begin_ + i] = ((T *) buf.ptr)[i];
        }
    } catch (const py::cast_error& e) {
        py::print("ERROR: Python casting error (python object of unexpected type) when converting Gym box space element to a feature atom vector");
    }
}


template <typename T>
void BoxSpace<T>::convert_element_to_feature_atoms_float(const py::object& element, std::vector<int>& feature_atoms) const {
    try {
        if (!py::isinstance<py::array_t<T>>(element)) {
            py::print("ERROR: Gym box space element not a numpy array or missing explicit 'dtype' matching that of 'low' amd 'high' arrays");
            return;
        }
        py::buffer_info ebuf = py::cast<py::array_t<T>>(element).request();
        py::buffer_info lbuf = low_.request(); // request() does not change the array
        py::buffer_info hbuf = high_.request(); // request() does not change the array
        if (lbuf.size != ebuf.size) {
            py::print("ERROR: Gym box space element and 'low' array not of the same size");
            return;
        }
        for (unsigned int i = 0 ; i < ebuf.size ; i++) {
            feature_atoms[feature_atom_vector_begin_ + i] = (int) std::floor(normalize(((T*) ebuf.ptr)[i], ((T*) lbuf.ptr)[i], ((T*) hbuf.ptr)[i], space_relative_precision_) / space_relative_precision_);
        }
    } catch (const py::cast_error& e) {
        py::print("ERROR: Python casting error (python object of unexpected type) when converting Gym box space element to a feature atom vector");
    }
}


template <typename T>
py::object BoxSpace<T>::convert_feature_atoms_to_element_int(const std::vector<int>& feature_atoms) const {
    const ssize_t* shape = low_.shape();
    py::array_t<T> result = py::array_t<T>(std::vector<ssize_t>(shape, shape + low_.ndim()));
    py::buffer_info buf = result.request();
    for (unsigned int i = 0 ; i < buf.size ; i++) {
        ((T*) buf.ptr)[i] = feature_atoms[feature_atom_vector_begin_ + i];
    }
    return result;
}


template <typename T>
py::object BoxSpace<T>::convert_feature_atoms_to_element_float(const std::vector<int>& feature_atoms) const {
    const ssize_t* shape = low_.shape();
    py::array_t<T> result = py::array_t<T>(std::vector<ssize_t>(shape, shape + low_.ndim()));
    py::buffer_info ebuf = result.request();
    py::buffer_info lbuf = low_.request(); // request() does not change the array
    py::buffer_info hbuf = high_.request(); // request() does not change the array
    for (unsigned int i = 0 ; i < ebuf.size ; i++) {
        ((T*) ebuf.ptr)[i] = inv_normalize(std::max((T) 0.0, std::min((T) (feature_atoms[feature_atom_vector_begin_ + i] * space_relative_precision_), (T) 1.0)),
                                           ((T*) lbuf.ptr)[i], ((T*) hbuf.ptr)[i], space_relative_precision_);
    }
    return result;
}


DictSpace::DictSpace(const py::dict& spaces, double space_relative_precision, unsigned int feature_atom_vector_begin)
try : GymSpace(feature_atom_vector_begin) {
    number_of_feature_atoms_ = 0;
    for (auto s : spaces) {
        auto i = spaces_.insert(std::make_pair(py::cast<py::str>(s.first),
                                               std::move(GymSpace::import_from_python(py::cast<py::object>(s.second),
                                                                                      space_relative_precision,
                                                                                      feature_atom_vector_begin + number_of_feature_atoms_)))
                               ).first;
        number_of_feature_atoms_ += i->second->get_number_of_feature_atoms();
    }
} catch (const py::cast_error& e) {
    throw std::logic_error("ERROR: Python casting error (python object of unexpected type) when importing Gym dict space");
}


std::unique_ptr<GymSpace> DictSpace::import_from_python(const py::object& gym_space, double space_relative_precision, unsigned int feature_atom_vector_begin) {
    try {
        if (!py::hasattr(gym_space, "spaces")) {
            py::print("ERROR: Gym dict space missing attribute 'spaces'");
            return std::unique_ptr<GymSpace>();
        }
        if (!py::isinstance<py::dict>(gym_space.attr("spaces"))) {
            py::print("ERROR: Gym dict space's 'spaces' not of type 'dict'");
            return std::unique_ptr<GymSpace>();
        }
        return std::make_unique<DictSpace>(py::cast<py::dict>(gym_space.attr("spaces")), space_relative_precision, feature_atom_vector_begin);
    } catch (const py::cast_error& e) {
        py::print("ERROR: Python casting error (python object of unexpected type) when trying to interpret Gym dict space's subspaces as a python dict");
        return std::unique_ptr<GymSpace>();
    } catch (const std::exception& e) {
        py::print("ERROR: " + std::string(e.what()));
        return std::unique_ptr<GymSpace>();
    }
}


void DictSpace::convert_element_to_feature_atoms(const py::object& element, std::vector<int>& feature_atoms) const {
    try {
        if (!py::isinstance<py::dict>(element)) {
            py::print("ERROR: Gym dict space element not of type 'dict'");
            return;
        }
        py::dict d = py::cast<py::dict>(element);
        for (auto i : d) {
            auto e = spaces_.find(py::cast<py::str>(i.first));
            if (e == spaces_.end()) {
                py::print("ERROR: key '" + std::string(py::cast<py::str>(i.first)) + "' not in the Gym dict space key list");
                return;
            }
            e->second->convert_element_to_feature_atoms(py::cast<py::object>(i.second), feature_atoms);
        }
    } catch (const py::cast_error& e) {
        py::print("ERROR: Python casting error (python object of unexpected type) when trying to convert Gym dict space element to a feature atom vector");
    }
}


py::object DictSpace::convert_feature_atoms_to_element(const std::vector<int>& feature_atoms) const {
    py::dict result = py::dict();
    for (const auto& s : spaces_) {
        result[py::str(s.first)] = s.second->convert_feature_atoms_to_element(feature_atoms);
    }
    return result;
}


DiscreteSpace::DiscreteSpace(const py::int_& n, unsigned int feature_atom_vector_begin)
: GymSpace(feature_atom_vector_begin), n_(n) {
    number_of_feature_atoms_ = 1;
}


std::unique_ptr<GymSpace> DiscreteSpace::import_from_python(const py::object& gym_space, unsigned int feature_atom_vector_begin) {
    try {
        if (!py::hasattr(gym_space, "n")) {
            py::print("ERROR: Gym discrete space missing attribute 'n'");
            return std::unique_ptr<GymSpace>();
        }
        if (!py::isinstance<py::int_>(gym_space.attr("n"))) {
            py::print("ERROR: Gym discrete space's 'n' not of type 'int'");
            return std::unique_ptr<GymSpace>();
        }
        return std::make_unique<DiscreteSpace>(py::cast<py::int_>(gym_space.attr("n")), feature_atom_vector_begin);
    } catch (const py::cast_error& e) {
        py::print("ERROR: Python casting error (python object of unexpected type) when trying to interpret Gym discrete space size as a python int");
        return std::unique_ptr<GymSpace>();
    }
}


void DiscreteSpace::convert_element_to_feature_atoms(const py::object& element, std::vector<int>& feature_atoms) const {
    try {
        if (!py::isinstance<py::int_>(element)) {
            py::print("ERROR: Gym discrete space element not of type 'int'");
            return;
        }
        feature_atoms[feature_atom_vector_begin_] = py::cast<py::int_>(element);
    } catch (const py::cast_error& e) {
        py::print("ERROR: Python casting error (python object of unexpected type) when trying to convert Gym discrete space element to a feature atom vector");
    }
}


py::object DiscreteSpace::convert_feature_atoms_to_element(const std::vector<int>& feature_atoms) const {
    return py::int_(feature_atoms[feature_atom_vector_begin_]);
}


MultiBinarySpace::MultiBinarySpace(const py::int_& n, unsigned int feature_atom_vector_begin)
: GymSpace(feature_atom_vector_begin), n_(n) {
    number_of_feature_atoms_ = n_;
}


std::unique_ptr<GymSpace> MultiBinarySpace::import_from_python(const py::object& gym_space, unsigned int feature_atom_vector_begin) {
    try {
        if (!py::hasattr(gym_space, "n")) {
            py::print("ERROR: Gym multi-binary space missing attribute 'n'");
            return std::unique_ptr<GymSpace>();
        }
        if (!py::isinstance<py::int_>(gym_space.attr("n"))) {
            py::print("ERROR: Gym multi-binary space's 'n' not of type 'int'");
            return std::unique_ptr<GymSpace>();
        }
        return std::make_unique<MultiBinarySpace>(py::cast<py::int_>(gym_space.attr("n")), feature_atom_vector_begin);
    } catch (const py::cast_error& e) {
        py::print("ERROR: Python casting error (python object of unexpected type) when trying to interpret Gym multi-binary space dimension as a python int");
        return std::unique_ptr<GymSpace>();
    }
}


void MultiBinarySpace::convert_element_to_feature_atoms(const py::object& element, std::vector<int>& feature_atoms) const {
    try {
        if (!py::isinstance<py::array_t<std::int8_t>>(element)) {
            py::print("ERROR: Gym multi-binary space element not a numpy array of int8 dtype");
            return;
        }
        py::buffer_info buf = py::cast<py::array_t<std::int8_t>>(element).request();
        if (buf.size != n_) {
            py::print("ERROR: Gym multi-binary space element numpy array not of same dimension as the space's dimension");
            return;
        }
        for (unsigned int i = 0 ; i < n_ ; i++) {
            feature_atoms[feature_atom_vector_begin_ + i] = ((std::int8_t*) buf.ptr)[i];
        }
    } catch (const py::cast_error& e) {
        py::print("ERROR: Python casting error (python object of unexpected type) when trying to convert Gym multi-binary space element to a feature atom vector");
    }
}


py::object MultiBinarySpace::convert_feature_atoms_to_element(const std::vector<int>& feature_atoms) const {
    py::array_t<std::int8_t> result = py::array_t<std::int8_t>(n_);
    py::buffer_info buf = result.request();
    for (unsigned int i = 0 ; i < n_ ; i++) {
        ((std::int8_t*) buf.ptr)[i] = feature_atoms[feature_atom_vector_begin_ + i];
    }
    return result;
}


MultiDiscreteSpace::MultiDiscreteSpace(const py::array_t<unsigned int>& nvec, unsigned int feature_atom_vector_begin)
try : GymSpace(feature_atom_vector_begin), nvec_(nvec) {
    if (nvec_.ndim() != 1) {
        throw std::domain_error("Gym multi-discrete space dimension different from 1");
    }
    number_of_feature_atoms_ = nvec_.size();
} catch (const std::exception& e) {
    throw;
}


std::unique_ptr<GymSpace> MultiDiscreteSpace::import_from_python(const py::object& gym_space, unsigned int feature_atom_vector_begin) {
    try {
        if (!py::hasattr(gym_space, "nvec")) {
            py::print("ERROR: Gym multi-discrete space missing attribute 'nvec'");
            return std::unique_ptr<GymSpace>();
        }
        if (!py::isinstance<py::array_t<std::int64_t>>(gym_space.attr("nvec"))) {
            py::print("ERROR: Gym multi-discrete space's 'nvec' not a numpy array");
            return std::unique_ptr<GymSpace>();
        }
        return std::make_unique<MultiDiscreteSpace>(py::cast<py::array_t<std::int64_t>>(gym_space.attr("nvec")), feature_atom_vector_begin);
    } catch (const py::cast_error& e) {
        py::print("ERROR: Python casting error (python object of unexpected type) when trying to interpret Gym multi-discrete space dimension vector as a python array of positive integers");
        return std::unique_ptr<GymSpace>();
    }
}


void MultiDiscreteSpace::convert_element_to_feature_atoms(const py::object& element, std::vector<int>& feature_atoms) const {
    try {
        if (!py::isinstance<py::array_t<std::int64_t>>(element)) {
            py::print("ERROR: Gym multi-discrete space element not a numpy array of int64 dtype");
            return;
        }
        py::buffer_info buf = py::cast<py::array_t<std::int64_t>>(element).request();
        if (buf.size != nvec_.size()) {
            py::print("ERROR: Gym multi-discrete space element numpy array not of same dimension as the space's dimension");
            return;
        }
        for (unsigned int i = 0 ; i < nvec_.size() ; i++) {
            feature_atoms[feature_atom_vector_begin_ + i] = ((std::int64_t*) buf.ptr)[i];
        }
    } catch (const py::cast_error& e) {
        py::print("ERROR: Python casting error (python object of unexpected type) when trying to convert Gym multi-discrete space element to a feature atom vector");
    }
}


py::object MultiDiscreteSpace::convert_feature_atoms_to_element(const std::vector<int>& feature_atoms) const {
    py::array_t<std::int64_t> result = py::array_t<std::int64_t>(nvec_.size());
    py::buffer_info buf = result.request();
    for (unsigned int i = 0 ; i < nvec_.size() ; i++) {
        ((std::int64_t*) buf.ptr)[i] = feature_atoms[feature_atom_vector_begin_ + i];
    }
    return result;
}


TupleSpace::TupleSpace(const py::tuple& spaces, double space_relative_precision, unsigned int feature_atom_vector_begin)
: GymSpace(feature_atom_vector_begin) {
    number_of_feature_atoms_ = 0;
    for (auto s : spaces) {
        spaces_.push_back(std::move(GymSpace::import_from_python(py::cast<py::object>(s), space_relative_precision, feature_atom_vector_begin + number_of_feature_atoms_)));
        number_of_feature_atoms_ += spaces_.back()->get_number_of_feature_atoms();
    }
}


std::unique_ptr<GymSpace> TupleSpace::import_from_python(const py::object& gym_space, double space_relative_precision, unsigned int feature_atom_vector_begin) {
    try {
        if (!py::hasattr(gym_space, "spaces")) {
            py::print("ERROR: Gym tuple space missing attribute 'spaces'");
            return std::unique_ptr<GymSpace>();
        }
        if (!py::isinstance<py::tuple>(gym_space.attr("spaces"))) {
            py::print("ERROR: Gym tuple space's 'spaces' not of type 'tuple'");
            return std::unique_ptr<GymSpace>();
        }
        return std::make_unique<TupleSpace>(py::cast<py::tuple>(gym_space.attr("spaces")), space_relative_precision, feature_atom_vector_begin);
    } catch (const py::cast_error& e) {
        py::print("ERROR: Python casting error (python object of unexpected type) when trying to interpret Gym tuple space's subspaces as a python tuple");
        return std::unique_ptr<GymSpace>();
    }
}


void TupleSpace::convert_element_to_feature_atoms(const py::object& element, std::vector<int>& feature_atoms) const {
    try {
        if (!py::isinstance<py::tuple>(element)) {
            py::print("ERROR: Gym tuple space element not of type 'tuple'");
            return;
        }
        py::tuple t = py::cast<py::tuple>(element);
        unsigned int i = 0;
        for (auto s = spaces_.begin() ; s != spaces_.end() ; s++) {
            if (i >= t.size()) {
                py::print("ERROR: Gym tuple space element size less than the space's tuple size");
                return;
            }
            (*s)->convert_element_to_feature_atoms(t[i], feature_atoms);
            i++;
        }
        if (i < t.size()) {
            py::print("ERROR: Gym tuple space element size larger than the space's tuple size");
            return;
        }
    } catch (const py::cast_error& e) {
        py::print("ERROR: Python casting error (python object of unexpected type) when trying to convert Gym tuple space element to a feature atom vector");
    }
}


py::object TupleSpace::convert_feature_atoms_to_element(const std::vector<int>& feature_atoms) const {
    py::tuple result = py::tuple(spaces_.size());
    unsigned int i = 0;
    for (const auto& s : spaces_) {
        result[i] = s->convert_feature_atoms_to_element(feature_atoms);
        i++;
    }
    return result;
}
