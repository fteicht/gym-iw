// (c) 2019 Florent Teichteil

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <list>
#include <map>

namespace py = pybind11;

class GymSpace {
public :
    GymSpace(unsigned int feature_atom_vector_begin = 0) : number_of_feature_atoms_(0), feature_atom_vector_begin_(feature_atom_vector_begin) {}
    virtual ~GymSpace() {}

    static std::unique_ptr<GymSpace> import_from_python(const py::object& gym_space, double space_relative_precision = 0.001, unsigned int feature_atom_vector_begin = 0);
    inline unsigned int get_number_of_feature_atoms() const {return number_of_feature_atoms_;}
    virtual void convert_element_to_feature_atoms(const py::object& element, std::vector<int>& feature_atoms) const =0;
    virtual py::object convert_feature_atoms_to_element(const std::vector<int>& feature_atoms) const =0;
    
protected :
    unsigned int number_of_feature_atoms_;
    unsigned int feature_atom_vector_begin_; // index of the first element of this Gym space in the whole feature atom vector
};


template <typename T>
class BoxSpace : public GymSpace {
public :
    BoxSpace(const py::array_t<T>& low, const py::array_t<T>& high, double space_relative_precision = 0.001, unsigned int feature_atom_vector_begin = 0);
    virtual ~BoxSpace() {}

    static std::unique_ptr<GymSpace> import_from_python(const py::object& gym_space, double space_relative_precision = 0.001, unsigned int feature_atom_vector_begin = 0);
    
    virtual void convert_element_to_feature_atoms(const py::object& element, std::vector<int>& feature_atoms) const {
        convert_element_to_feature_atoms_generic(element, feature_atoms);
    }

    virtual py::object convert_feature_atoms_to_element(const std::vector<int>& feature_atoms) const {
        return convert_feature_atoms_to_element_generic(feature_atoms);
    }

private :
    mutable py::array_t<T> low_; // dirty trick to make py::array_t<T>::request() work with 'const this' since it seems it does not modify the array
    mutable py::array_t<T> high_; // dirty trick to make py::array_t<T>::request() work with 'const this' since it seems it does not modify the array
    double space_relative_precision_;

    inline static T sigmoid(const T& x, const T& decay) {
        return ((T) 1.0) / (((T) 1.0) + std::exp(-decay * x));
    }

    inline static T inv_sigmoid(const T& x, const T& decay) {
        return -std::log((((T) 1.0) / x) - ((T) 1.0)) / decay;
    }

    inline static T normalize(const T& x, const T& min, const T& max, const T& decay) {
        return (sigmoid(x, decay) - sigmoid(min, decay)) / (sigmoid(max, decay) - sigmoid(min, decay));
    }

    inline static T inv_normalize(const T& x, const T& min, const T& max, const T& decay) {
        return inv_sigmoid(sigmoid(min, decay) + (x * (sigmoid(max, decay) - sigmoid(min, decay))), decay);
    }

    template <typename TT = T>
    inline typename std::enable_if<std::is_integral<TT>::value, void>::type convert_element_to_feature_atoms_generic(const py::object& element, std::vector<int>& feature_atoms) const {
        return convert_element_to_feature_atoms_int(element, feature_atoms);
    }

    template <typename TT = T>
    inline typename std::enable_if<std::is_floating_point<TT>::value, void>::type convert_element_to_feature_atoms_generic(const py::object& element, std::vector<int>& feature_atoms) const {
        return convert_element_to_feature_atoms_float(element, feature_atoms);
    }

    void convert_element_to_feature_atoms_int(const py::object& element, std::vector<int>& feature_atoms) const;
    void convert_element_to_feature_atoms_float(const py::object& element, std::vector<int>& feature_atoms) const;
    
    template <typename TT = T>
    inline typename std::enable_if<std::is_integral<TT>::value, py::object>::type convert_feature_atoms_to_element_generic(const std::vector<int>& feature_atoms) const {
        return convert_feature_atoms_to_element_int(feature_atoms);
    }
    
    template <typename TT = T>
    inline typename std::enable_if<std::is_floating_point<TT>::value, py::object>::type convert_feature_atoms_to_element_generic(const std::vector<int>& feature_atoms) const {
        return convert_feature_atoms_to_element_float(feature_atoms);
    }

    py::object convert_feature_atoms_to_element_int(const std::vector<int>& feature_atoms) const;
    py::object convert_feature_atoms_to_element_float(const std::vector<int>& feature_atoms) const;
};


class DictSpace : public GymSpace {
public :
    DictSpace(const py::dict& spaces, double space_relative_precision = 0.001, unsigned int feature_atom_vector_begin = 0);
    virtual ~DictSpace() {}

    static std::unique_ptr<GymSpace> import_from_python(const py::object& gym_space, double space_relative_precision = 0.001, unsigned int feature_atom_vector_begin = 0);
    virtual void convert_element_to_feature_atoms(const py::object& element, std::vector<int>& feature_atoms) const;
    virtual py::object convert_feature_atoms_to_element(const std::vector<int>& feature_atoms) const;

private :
    std::map<std::string, std::unique_ptr<GymSpace>> spaces_;
};


class DiscreteSpace : public GymSpace {
public :
    DiscreteSpace(const py::int_& n, unsigned int feature_atom_vector_begin = 0);
    virtual ~DiscreteSpace() {}

    static std::unique_ptr<GymSpace> import_from_python(const py::object& gym_space, unsigned int feature_atom_vector_begin = 0);
    virtual void convert_element_to_feature_atoms(const py::object& element, std::vector<int>& feature_atoms) const;
    virtual py::object convert_feature_atoms_to_element(const std::vector<int>& feature_atoms) const;
    
private :
    unsigned int n_;
};


class MultiBinarySpace : public GymSpace {
public :
    MultiBinarySpace(const py::int_& n, unsigned int feature_atom_vector_begin = 0);
    virtual ~MultiBinarySpace() {}

    static std::unique_ptr<GymSpace> import_from_python(const py::object& gym_space, unsigned int feature_atom_vector_begin = 0);
    virtual void convert_element_to_feature_atoms(const py::object& element, std::vector<int>& feature_atoms) const;
    virtual py::object convert_feature_atoms_to_element(const std::vector<int>& feature_atoms) const;
    
private :
    unsigned int n_;
};


class MultiDiscreteSpace : public GymSpace {
public :
    MultiDiscreteSpace(const py::array_t<unsigned int>& nvec, unsigned int feature_atom_vector_begin = 0);
    virtual ~MultiDiscreteSpace() {}

    static std::unique_ptr<GymSpace> import_from_python(const py::object& gym_space, unsigned int feature_atom_vector_begin = 0);
    virtual void convert_element_to_feature_atoms(const py::object& element, std::vector<int>& feature_atoms) const;
    virtual py::object convert_feature_atoms_to_element(const std::vector<int>& feature_atoms) const;
    
private :
    py::array_t<std::int64_t> nvec_;
};


class TupleSpace : public GymSpace {
public :
    TupleSpace(const py::tuple& spaces, double space_relative_precision = 0.001, unsigned int feature_atom_vector_begin = 0);
    virtual ~TupleSpace() {}

    static std::unique_ptr<GymSpace> import_from_python(const py::object& gym_space, double space_relative_precision = 0.001, unsigned int feature_atom_vector_begin = 0);
    virtual void convert_element_to_feature_atoms(const py::object& element, std::vector<int>& feature_atoms) const;
    virtual py::object convert_feature_atoms_to_element(const std::vector<int>& feature_atoms) const;
    
private :
    std::list<std::unique_ptr<GymSpace>> spaces_;
};
