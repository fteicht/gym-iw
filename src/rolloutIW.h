// (c) 2017 Blai Bonet

#ifndef ROLLOUT_IW_H
#define ROLLOUT_IW_H

#include <cassert>
#include <map>
#include <string>
#include <vector>

#include "sim_planner.h"
#include "logger.h"

template <typename Environment>
struct RolloutIW : SimPlanner<Environment> {
    const double time_budget_;
    const bool novelty_subtables_;
    const bool random_actions_;
    const size_t max_rep_;
    const double discount_;
    const int nodes_threshold_;
    const size_t max_depth_;

    mutable size_t num_rollouts_;
    mutable size_t num_expansions_;
    mutable size_t num_cases_[4];
    mutable double total_time_;
    mutable double expand_time_;
    mutable size_t root_height_;
    mutable bool random_decision_;

    RolloutIW(Environment &sim,
              size_t frameskip,
              size_t num_tracked_atoms,
              int simulator_budget,
              double time_budget,
              bool novelty_subtables,
              bool random_actions,
              size_t max_rep,
              double discount,
              int nodes_threshold,
              size_t max_depth,
              int debug_threshold,
              int random_seed,
              std::string logger_mode)
      : SimPlanner<Environment>(sim, frameskip, simulator_budget, num_tracked_atoms, debug_threshold, random_seed, logger_mode),
        time_budget_(time_budget),
        novelty_subtables_(novelty_subtables),
        random_actions_(random_actions),
        max_rep_(max_rep),
        discount_(discount),
        nodes_threshold_(nodes_threshold),
        max_depth_(max_depth) {
    }
    virtual ~RolloutIW() { }

    virtual std::string name() const {
        return std::string("rollout(")
          + "frameskip=" + std::to_string(this->frameskip_)
          + ",simulator-budget=" + std::to_string(this->simulator_budget_)
          + ",time-budget=" + std::to_string(this->time_budget_)
          + ",novelty-subtables=" + std::to_string(novelty_subtables_)
          + ",random-actions=" + std::to_string(random_actions_)
          + ",max-rep=" + std::to_string(max_rep_)
          + ",discount=" + std::to_string(discount_)
          + ",nodes-threshold=" + std::to_string(nodes_threshold_)
          + ",max-depth=" + std::to_string(max_depth_)
          + ")";
    }

    virtual std::string str() const {
        return std::string("rollout-iw");
    }

    virtual bool random_decision() const {
        return random_decision_;
    }
    virtual size_t height() const {
        return root_height_;
    }
    virtual size_t expanded() const {
        return num_expansions_;
    }

    virtual Node<Environment>* get_branch(const std::vector<typename Environment::Action> &prefix,
                                          Node<Environment> *root,
                                          double last_reward,
                                          std::deque<typename Environment::Action> &branch) {
        assert(!prefix.empty());

        Logger::Info << "**** rollout: get branch ****" << std::endl;
        Logger::Info << "prefix: sz=" << prefix.size() << ", actions=";
        this->print_prefix(Logger::Info, prefix);
        Logger::Continuation(Logger::Info) << std::endl;
        Logger::Info << "input:"
                     << " #nodes=" << (root == nullptr ? "na" : std::to_string(root->num_nodes()))
                     << ", #tips=" << (root == nullptr ? "na" : std::to_string(root->num_tip_nodes()))
                     << ", height=" << (root == nullptr ? "na" : std::to_string(root->height_))
                     << std::endl;

        // reset stats and start timer
        reset_stats();
        double start_time = Utils::read_time_in_seconds();

        // novelty table and other vars
        std::map<int, std::vector<int> > novelty_table_map;

        // construct root node
        assert((root == nullptr) || (typename Environment::ActionEqual()(root->action_, prefix.back())));
        if( root == nullptr ) {
            Node<Environment> *root_parent = new Node<Environment>(nullptr, typename Environment::Action(), -1);
            root_parent->observation_ = std::make_unique<typename Environment::Observation>();
            this->apply_prefix(this->initial_sim_observation_, prefix, root_parent->observation_.get());
            root = new Node<Environment>(root_parent, prefix.back(), 0);
        }
        assert(root->parent_ != nullptr);
        root->parent_->parent_ = nullptr;

        // if root has some children, make sure it has all children
        if( root->num_children_ > 0 ) {
            assert(root->first_child_ != nullptr);
            std::set<typename Environment::Action, typename Environment::ActionLess> root_actions;
            for( Node<Environment> *child = root->first_child_; child != nullptr; child = child->sibling_ )
                root_actions.insert(child->action_);

            // complete children
            assert(root->num_children_ <= int(this->action_set_.size()));
            if( root->num_children_ < int(this->action_set_.size()) ) {
                for( size_t k = 0; k < this->action_set_.size(); ++k ) {
                    if( root_actions.find(this->action_set_[k]) == root_actions.end() )
                        root->expand(this->action_set_[k]);
                }
            }
            assert(root->num_children_ == int(this->action_set_.size()));
        } else {
            // make sure this root node isn't marked as frame rep
            root->parent_->feature_atoms_.clear();
        }

        // normalize depths, reset rep counters, and recompute path rewards
        root->parent_->depth_ = -1;
        root->normalize_depth();
        root->reset_frame_rep_counters(this->frameskip_);
        root->recompute_path_rewards(root);

        // construct/extend lookahead tree
        if( int(root->num_nodes()) < nodes_threshold_ ) {
            double elapsed_time = Utils::read_time_in_seconds() - start_time;

            // clear solved labels
            clear_solved_labels(root);
            root->parent_->solved_ = false;
            Logger::Debug << "";
            while( !root->solved_ && (int(this->simulator_calls_) < this->simulator_budget_) && (elapsed_time < this->time_budget_) ) {
                Logger::Continuation(Logger::Debug) << '.' << std::flush;
                this->save_environment();
                rollout(root, novelty_table_map);
                this->restore_environment();
                elapsed_time = Utils::read_time_in_seconds() - start_time;
            }
            Logger::Continuation(Logger::Debug) << std::endl;
        }

        // if nothing was expanded, return random actions (it can only happen with small time budget)
        if( root->num_children_ == 0 ) {
            assert(root->first_child_ == nullptr);
            assert(time_budget_ != std::numeric_limits<double>::infinity());
            random_decision_ = true;
            branch.push_back(this->random_action());
        } else {
            assert(root->first_child_ != nullptr);

            // backup values and calculate heights
            root->backup_values(discount_);
            root->calculate_height();
            root_height_ = root->height_;

            // print info about root node
            Logger::Debug << Logger::green()
                          << "root:"
                          << " solved=" << root->solved_
                          << ", value=" << root->value_
                          << ", imm-reward=" << root->reward_
                          << ", children=[";
            for( Node<Environment> *child = root->first_child_; child != nullptr; child = child->sibling_ )
                Logger::Continuation(Logger::Debug) << child->qvalue(discount_) << ":" << child->action_ << " ";
            Logger::Continuation(Logger::Debug) << "]" << Logger::normal() << std::endl;

            // compute branch
            if( root->value_ != 0 ) {
                root->best_branch(branch, discount_);
            } else {
                if( random_actions_ ) {
                    random_decision_ = true;
                    branch.push_back(this->random_zero_value_action(root, discount_));
                } else {
                    root->longest_zero_value_branch(discount_, branch);
                    assert(!branch.empty());
                }
            }

            // make sure states along branch exist (only needed when doing partial caching)
            this->generate_states_along_branch(root, branch);

            // print branch
            assert(!branch.empty());
            Logger::Debug << "branch:"
                          << " value=" << root->value_
                          << ", size=" << branch.size()
                          << ", actions:"
                          << std::endl;
            //root->print_branch(logos_, branch);
        }

        // stop timer and print stats
        total_time_ = Utils::read_time_in_seconds() - start_time;
        print_stats(Logger::Stats, *root, novelty_table_map);

        // return root node
        return root;
    }

    void rollout(Node<Environment> *root, std::map<int, std::vector<int> > &novelty_table_map) {
        ++num_rollouts_;

        // update root info
        if( root->is_info_valid_ != 2 )
            this->update_info(root);

        // perform rollout
        Node<Environment> *node = root;
        while( !node->solved_ ) {
            assert(node->is_info_valid_ == 2);

            // if first time at this node, expand node
            expand_if_necessary(node);

            // pick random unsolved child
            node = pick_unsolved_child(node);
            assert(!node->solved_);

            // update info
            if( node->is_info_valid_ != 2 )
                this->update_info(node);

            // report non-zero rewards
            if( node->reward_ > 0 ) {
                Logger::Continuation(Logger::Debug) << Logger::yellow() << "+" << Logger::normal() << std::flush;
            } else if( node->reward_ < 0 ) {
                Logger::Continuation(Logger::Debug) << "-" << std::flush;
            }

            // if terminal, label as solved and terminate rollout
            if( node->terminal_ ) {
                node->visited_ = true;
                assert((node->num_children_ == 0) && (node->first_child_ == nullptr));
                node->solve_and_backpropagate_label();
                //logos_ << "T[reward=" << node->reward_ << "]" << std::flush;
                break;
            }

            // verify repetitions of feature atoms (screen mode)
            if( node->frame_rep_ > int(max_rep_) ) {
                node->visited_ = true;
                assert((node->num_children_ == 0) && (node->first_child_ == nullptr));
                node->solve_and_backpropagate_label();
                //logos_ << "R" << std::flush;
                break;
            } else if( node->frame_rep_ > 0 ) {
                node->visited_ = true;
                //logos_ << "r" << std::flush;
                continue;
            }

            // calculate novelty
            std::vector<int> &novelty_table = this->get_novelty_table(node, novelty_table_map, novelty_subtables_);
            int atom = this->get_novel_atom(node->depth_, node->feature_atoms_, novelty_table);
            assert((atom >= 0) && (atom < int(novelty_table.size())));

            // five cases
            if( node->depth_ > int(max_depth_) ) {
                node->visited_ = true;
                assert((node->num_children_ == 0) && (node->first_child_ == nullptr));
                node->solve_and_backpropagate_label();
                //logos_ << "D" << std::flush;
                break;
            } else if( novelty_table[atom] > node->depth_ ) { // novel => not(visited)
                // when caching, the assertion
                //
                //   assert(!node->visited_);
                //
                // may be false as there may be nodes in given tree. We just replace
                // it by the if() expression below. Table updates are only peformed for
                // nodes added to existing tree.
                if( !node->visited_ ) {
                    ++num_cases_[0];
                    node->visited_ = true;
                    node->num_novel_features_ = this->update_novelty_table(node->depth_, node->feature_atoms_, novelty_table);
                    //logos_ << Utils::green() << "n" << Utils::normal() << std::flush;
                }
                continue;
            } else if( !node->visited_ && (novelty_table[atom] <= node->depth_) ) { // not(novel) and not(visited) => PRUNE
                ++num_cases_[1];
                node->visited_ = true;
                assert((node->num_children_ == 0) && (node->first_child_ == nullptr));
                node->solve_and_backpropagate_label();
                //logos_ << "x" << node->depth_ << std::flush;
                break;
            } else if( node->visited_ && (novelty_table[atom] < node->depth_) ) { // not(novel) and visited => PRUNE
                ++num_cases_[2];
                //node->remove_children();
                node->reward_ = -std::numeric_limits<double>::infinity();
                Logger::Continuation(Logger::Debug) << "-" << std::flush;
                node->solve_and_backpropagate_label();
                //logos_ << "X" << node->depth_ << std::flush;
                break;
            } else { // optimal and visited => CONTINUE
                assert(node->visited_ && (novelty_table[atom] == node->depth_));
                ++num_cases_[3];
                //logos_ << "c" << std::flush;
                continue;
            }
        }
    }

    void expand_if_necessary(Node<Environment> *node) const {
        if( node->num_children_ == 0 ) {
            assert(node->first_child_ == nullptr);
            if( node->frame_rep_ == 0 ) {
                ++num_expansions_;
                double start_time = Utils::read_time_in_seconds();
                node->expand(this->action_set_);
                expand_time_ += Utils::read_time_in_seconds() - start_time;
            } else {
                assert(node->parent_ != nullptr);
                node->expand(node->action_);
            }
            assert((node->num_children_ > 0) && (node->first_child_ != nullptr));
        }
    }

    Node<Environment>* pick_unsolved_child(const Node<Environment> *node) const {
        Node<Environment> *selected = nullptr;

        // decide to pick among all unsolved children or among those
        // with biggest number of novel features
        bool filter_unsolved_children = false; //lrand48() % 2;

        // select unsolved child
        size_t num_candidates = 0;
        int novel_features_threshold = std::numeric_limits<int>::min();;
        for( Node<Environment> *child = node->first_child_; child != nullptr; child = child->sibling_ ) {
            if( !child->solved_ && (child->num_novel_features_ >= novel_features_threshold) ) {
                if( filter_unsolved_children && (child->num_novel_features_ > novel_features_threshold) ) {
                    novel_features_threshold = child->num_novel_features_;
                    num_candidates = 0;
                }
                ++num_candidates;
            }
        }
        assert(num_candidates > 0);
        size_t index = lrand48() % num_candidates;
        for( Node<Environment> *child = node->first_child_; child != nullptr; child = child->sibling_ ) {
            if( !child->solved_ && (child->num_novel_features_ >= novel_features_threshold) ) {
                if( index == 0 ) {
                    selected = child;
                    break;
                }
                --index;
            }
        }
        assert(selected != nullptr);
        assert(!selected->solved_);
        return selected;
    }

    void clear_solved_labels(Node<Environment> *node) const {
        node->solved_ = false;
        for( Node<Environment> *child = node->first_child_; child != nullptr; child = child->sibling_ )
            clear_solved_labels(child);
    }

    void reset_stats() const {
        SimPlanner<Environment>::reset_stats();
        num_rollouts_ = 0;
        num_expansions_ = 0;
        num_cases_[0] = 0;
        num_cases_[1] = 0;
        num_cases_[2] = 0;
        num_cases_[3] = 0;
        total_time_ = 0;
        expand_time_ = 0;
        root_height_ = 0;
        random_decision_ = false;
    }

    void print_stats(Logger::mode_t logger_mode, const Node<Environment> &root, const std::map<int, std::vector<int> > &novelty_table_map) const {
        logger_mode << "decision-stats:"
                    << " #rollouts=" << num_rollouts_
                    << " #entries=[";

        for( std::map<int, std::vector<int> >::const_iterator it = novelty_table_map.begin(); it != novelty_table_map.end(); ++it )
            Logger::Continuation(logger_mode) << it->first << ":" << this->num_entries(it->second) << "/" << it->second.size() << ",";

        Logger::Continuation(logger_mode)
          << "]"
          << " #nodes=" << root.num_nodes()
          << " #tips=" << root.num_tip_nodes()
          << " height=[" << root.height_ << ":";

        for( Node<Environment> *child = root.first_child_; child != nullptr; child = child->sibling_ )
            Logger::Continuation(logger_mode) << child->height_ << ",";

        Logger::Continuation(logger_mode)
          << "]"
          << " #expansions=" << num_expansions_
          << " #cases=[" << num_cases_[0] << "," << num_cases_[1] << "," << num_cases_[2] << "," << num_cases_[3] << "]"
          << " #sim=" << this->simulator_calls_
          << " total-time=" << total_time_
          << " simulator-time=" << this->sim_time_
          << " reset-time=" << this->sim_reset_time_
          << " save/restore-environment-time=" << this->sim_save_environment_time_
          << " expand-time=" << expand_time_
          << " update-novelty-time=" << this->update_novelty_time_
          << " get-atoms-calls=" << this->get_atoms_calls_
          << " get-atoms-time=" << this->get_atoms_time_
          << " novel-atom-time=" << this->novel_atom_time_
          << std::endl;
    }

    virtual void log_stats() const {
        Logger::Stats
        << " simulator-budget=" << this->simulator_budget_
        << " time-budget=" << time_budget_
        << " discount=" << discount_
        << " max-rep=" << max_rep_
        << " nodes-threshold=" << nodes_threshold_
        << " novelty-subtables=" << novelty_subtables_
        << " random-actions=" << random_actions_
        << " max-depth=" << max_depth_;
    }
};

#endif

