// (c) 2017 Blai Bonet

#ifndef NODE_H
#define NODE_H

#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <pybind11/pybind11.h>

namespace py = pybind11;

template <typename Environment> class Node;

template <typename Environment>
inline void remove_tree(Node<Environment> *node);

template <typename Environment>
class Node {
  public:
    bool visited_;                           // label
    bool solved_;                            // label
    typename Environment::Action action_;    // action mapping parent to this node
    int depth_;                              // node's depth
    int height_;                             // node's height (calculated)
    double reward_;                           // reward for this node
    double path_reward_;                      // reward of full path leading to this node
    int is_info_valid_;                      // is info valid? (0=no, 1=partial, 2=full)
    bool terminal_;                          // is node a terminal node?
    double value_;                            // backed up value

    mutable std::unique_ptr<typename Environment::Observation> observation_;         // observation for this node
    mutable std::vector<int> feature_atoms_; // features made true by this node
    mutable int num_novel_features_;         // number of features this node makes novel
    mutable int frame_rep_;                  // frame counter for number identical feature atoms through ancestors

    // structure
    int num_children_;                       // number of children
    Node *first_child_;                      // first child
    Node *sibling_;                          // right sibling of this node
    Node *parent_;                           // pointer to parent node

    Node(Node *parent, const typename Environment::Action& action, size_t depth)
      : visited_(false),
        solved_(false),
        action_(action),
        depth_(depth),
        height_(0),
        reward_(0),
        path_reward_(0),
        is_info_valid_(0),
        terminal_(false),
        value_(0),
        num_novel_features_(0),
        frame_rep_(0),
        num_children_(0),
        first_child_(nullptr),
        sibling_(nullptr),
        parent_(parent) {
    }
    ~Node() { }

    void remove_children() {
        while( first_child_ != nullptr ) {
            Node *child = first_child_;
            first_child_ = first_child_->sibling_;
            remove_tree(child);
        }
    }

    void expand(const typename Environment::Action& action) {
        Node *new_child = new Node(this, action, 1 + depth_);
        new_child->sibling_ = first_child_;
        first_child_ = new_child;
        ++num_children_;
    }
    void expand(const std::vector<typename Environment::Action> &actions, bool random_shuffle = true) {
        assert((num_children_ == 0) && (first_child_ == nullptr));
        for( size_t k = 0; k < actions.size(); ++k )
            expand(actions[k]);
        //if( random_shuffle ) std::random_shuffle(children_.begin(), children_.end()); // CHECK: missing
        assert(num_children_ == int(actions.size()));
    }

    void clear_cached_observations() {
        if( is_info_valid_ == 2 ) {
            observation_.reset();
            is_info_valid_ = 1;
        }
        for( Node *child = first_child_; child != nullptr; child = child->sibling_ )
            child->clear_cached_observations();
    }

    Node* advance(const typename Environment::Action& action) {
        assert((num_children_ > 0) && (first_child_ != nullptr));
        assert((parent_ == nullptr) || (parent_->parent_ == nullptr));
        if( parent_ != nullptr ) {
            delete parent_;
            parent_ = nullptr;
        }

        Node *selected = nullptr;
        Node *sibling = nullptr;
        for( Node *child = first_child_; child != nullptr; child = sibling ) {
            sibling = child->sibling_;
            if( typename Environment::ActionEqual()(child->action_, action) )
                selected = child;
            else
                remove_tree(child);
        }
        assert(selected != nullptr);

        selected->sibling_ = nullptr;
        first_child_ = selected;
        return selected;
    }

    void normalize_depth(int depth = 0) {
        depth_ = depth;
        for( Node *child = first_child_; child != nullptr; child = child->sibling_ )
            child->normalize_depth(1 + depth);
    }

    void reset_frame_rep_counters(int frameskip, int parent_frame_rep) {
        if( frame_rep_ > 0 ) {
            frame_rep_ = parent_frame_rep + frameskip;
            for( Node *child = first_child_; child != nullptr; child = child->sibling_ )
                child->reset_frame_rep_counters(frameskip, frame_rep_);
        }
    }
    void reset_frame_rep_counters(int frameskip) {
        reset_frame_rep_counters(frameskip, -frameskip);
    }

    void recompute_path_rewards(const Node *ref = nullptr) {
        if( this == ref ) {
            path_reward_ = 0;
        } else {
            assert(parent_ != nullptr);
            path_reward_ = parent_->path_reward_ + reward_;
        }
        for( Node *child = first_child_; child != nullptr; child = child->sibling_ )
            child->recompute_path_rewards();
    }

    void solve_and_backpropagate_label() {
        assert(!solved_);
        if( !solved_ ) {
            solved_ = true;
            if( parent_ != nullptr ) {
                assert(!parent_->solved_);
                bool unsolved_siblings = false;
                for( Node *child = parent_->first_child_; child != nullptr; child = child->sibling_ ) {
                    if( !child->solved_ ) {
                        unsolved_siblings = true;
                        break;
                    }
                }
                if( !unsolved_siblings )
                    parent_->solve_and_backpropagate_label();
            }
        }
    }

    double qvalue(double discount) const {
        return reward_ + discount * value_;
    }

    double backup_values_upward(double discount) { // NOT USED
        assert((num_children_ == 0) || (is_info_valid_ != 0));

        value_ = 0;
        if( num_children_ > 0 ) {
            assert(first_child_ != nullptr);
            double max_child_value = -std::numeric_limits<double>::infinity();
            for( Node *child = first_child_; child != nullptr; child = child->sibling_ ) {
                double child_value = child->qvalue(discount);
                max_child_value = std::max(max_child_value, child_value);
            }
            value_ = max_child_value;
        }

        if( parent_ == nullptr )
            return value_;
        else
            return parent_->backup_values_upward(discount);
    }

    double backup_values(double discount) {
        assert((num_children_ == 0) || (is_info_valid_ != 0));
        value_ = 0;
        if( num_children_ > 0 ) {
            assert(first_child_ != nullptr);
            double max_child_value = -std::numeric_limits<double>::infinity();
            for( Node *child = first_child_; child != nullptr; child = child->sibling_ ) {
                child->backup_values(discount);
                double child_value = child->qvalue(discount);
                max_child_value = std::max(max_child_value, child_value);
            }
            value_ = max_child_value;
        }
        return value_;
    }

    double backup_values_along_branch(const std::deque<typename Environment::Action> &branch, double discount, size_t index = 0) { // NOT USED
        //assert(is_info_valid_ && !children_.empty()); // tree now grows past branch
        if( index == branch.size() ) {
            backup_values(discount);
            return value_;
        } else {
            assert(index < branch.size());
            double value_along_branch = 0;
            const typename Environment::Action &action = branch[index];
            double max_child_value = -std::numeric_limits<double>::infinity();
            for( Node *child = first_child_; child != nullptr; child = child->sibling_ ) {
                if( child->action_ == action )
                    value_along_branch = child->backup_values_along_branch(branch, discount, ++index);
                max_child_value = std::max(max_child_value, child->value_);
            }
            value_ = reward_ + discount * max_child_value;
            return reward_ + discount * value_along_branch;
        }
    }

    const Node *best_tip_node(double discount) const { // NOT USED
        if( num_children_ == 0 ) {
            return this;
        } else {
            assert(first_child_ != nullptr);
            size_t num_best_children = 0;
            for( Node *child = first_child_; child != nullptr; child = child->sibling_ )
                num_best_children += child->qvalue(discount) == value_;
            assert(num_best_children > 0);
            size_t index_best_child = lrand48() % num_best_children;
            for( Node *child = first_child_; child != nullptr; child = child->sibling_ ) {
                if( child->qvalue(discount) == value_ ) {
                    if( index_best_child == 0 )
                        return child->best_tip_node(discount);
                    --index_best_child;
                }
            }
            assert(0);
        }
    }

    void best_branch(std::deque<typename Environment::Action> &branch, double discount) const {
        if( num_children_ > 0 ) {
            assert(first_child_ != nullptr);
            size_t num_best_children = 0;
            for( Node *child = first_child_; child != nullptr; child = child->sibling_ )
                num_best_children += child->qvalue(discount) == value_;
            assert(num_best_children > 0);
            size_t index_best_child = lrand48() % num_best_children;
            for( Node *child = first_child_; child != nullptr; child = child->sibling_ ) {
                if( child->qvalue(discount) == value_ ) {
                    if( index_best_child == 0 ) {
                        branch.push_back(child->action_);
                        child->best_branch(branch, discount);
                        break;
                    }
                    --index_best_child;
                }
            }
        }
    }

    void longest_zero_value_branch(double discount, std::deque<typename Environment::Action> &branch) const {
        assert(value_ == 0);
        if( num_children_ > 0 ) {
            assert(first_child_ != nullptr);
            size_t max_height = 0;
            size_t num_best_children = 0;
            for( Node *child = first_child_; child != nullptr; child = child->sibling_ ) {
                if( (child->qvalue(discount) == 0) && (child->height_ >= int(max_height)) ) {
                    if( child->height_ > int(max_height) ) {
                        max_height = child->height_;
                        num_best_children = 0;
                    }
                    ++num_best_children;
                }
            }
            assert(num_best_children > 0);
            size_t index_best_child = lrand48() % num_best_children;
            for( Node *child = first_child_; child != nullptr; child = child->sibling_ ) {
                if( (child->qvalue(discount) == 0) && (child->height_ == int(max_height)) ) {
                    if( index_best_child == 0 ) {
                        branch.push_back(child->action_);
                        child->longest_zero_value_branch(discount, branch);
                        break;
                    }
                    --index_best_child;
                }
            }
        }
    }

    size_t num_tip_nodes() const {
        if( num_children_ == 0 ) {
            return 1;
        } else {
            assert(first_child_ != nullptr);
            size_t n = 0;
            for( Node *child = first_child_; child != nullptr; child = child->sibling_ )
                n += child->num_tip_nodes();
            return n;
        }
    }

    size_t num_nodes() const {
        size_t n = 1;
        for( Node *child = first_child_; child != nullptr; child = child->sibling_ )
            n += child->num_nodes();
        return n;
    }

    int calculate_height() {
        height_ = 0;
        if( num_children_ > 0 ) {
            assert(first_child_ != nullptr);
            for( Node *child = first_child_; child != nullptr; child = child->sibling_ ) {
                int child_height = child->calculate_height();
                height_ = std::max(height_, child_height);
            }
            height_ += 1;
        }
        return height_;
    }

    void print_branch(std::ostream &os, const std::deque<typename Environment::Action> &branch, size_t index = 0) const {
        print(os);
        if( index < branch.size() ) {
            typename Environment::Action action = branch[index];
            bool child_found = false;
            for( Node *child = first_child_; child != nullptr; child = child->sibling_ ) {
                if( child->action_ == action ) {
                    child->print_branch(os, branch, ++index);
                    child_found = true;
                    break;
                }
            }
            assert(child_found);
        }
    }

    void print(std::ostream &os) const {
        os << "node:"
           << " valid=" << is_info_valid_
           << ", solved=" << solved_
           << ", value=" << value_
           << ", reward=" << reward_
           << ", path-reward=" << path_reward_
           << ", action=" << action_
           << ", depth=" << depth_
           << ", children=[";
        for( Node *child = first_child_; child != nullptr; child = child->sibling_ )
            os << child->value_ << " ";
        os << "] (this=" << this << ", parent=" << parent_ << ")"
           << std::endl;
    }

    void print_tree(std::ostream &os) const {
        print(os);
        for( Node *child = first_child_; child != nullptr; child = child->sibling_ )
            child->print_tree(os);
    }
};

template <typename Environment>
inline void remove_tree(Node<Environment> *node) {
    Node<Environment> *sibling = nullptr;
    for( Node<Environment> *child = node->first_child_; child != nullptr; child = sibling ) {
        sibling = child->sibling_;
        remove_tree(child);
    }
    delete node;
}

#endif

