// (c) 2017 Blai Bonet

#ifndef BFS_IW_H
#define BFS_IW_H

#include <cassert>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <vector>

#include "sim_planner.h"
#include "logger.h"

template <typename Environment>
struct BfsIW : SimPlanner<Environment> {
    const double time_budget_;
    const bool novelty_subtables_;
    const bool random_actions_;
    const size_t max_rep_;
    const double discount_;
    const int nodes_threshold_;
    const bool break_ties_using_rewards_;

    mutable size_t num_expansions_;
    mutable double total_time_;
    mutable double expand_time_;
    mutable size_t root_height_;
    mutable bool random_decision_;

    BfsIW(Environment &sim,
          size_t frameskip,
          size_t num_tracked_atoms,
          double simulator_budget,
          double time_budget,
          bool novelty_subtables,
          bool random_actions,
          size_t max_rep,
          double discount,
          int nodes_threshold,
          bool break_ties_using_rewards,
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
        break_ties_using_rewards_(break_ties_using_rewards) {
    }
    virtual ~BfsIW() { }

    virtual std::string name() const {
        return std::string("bfs(")
          + "frameskip=" + std::to_string(this->frameskip_)
          + ",simulator-budget=" + std::to_string(this->simulator_budget_)
          + ",time-budget=" + std::to_string(time_budget_)
          + ",novelty-subtables=" + std::to_string(novelty_subtables_)
          + ",random-actions=" + std::to_string(random_actions_)
          + ",max-rep=" + std::to_string(max_rep_)
          + ",discount=" + std::to_string(discount_)
          + ",nodes-threshold=" + std::to_string(nodes_threshold_)
          + ",break-ties-using-rewards=" + std::to_string(break_ties_using_rewards_)
          + ")";
    }

    virtual std::string str() const {
        return std::string("bfs-iw");
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

        Logger::Info << "**** bfs: get branch ****" << std::endl;
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

        // novelty table
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

        // normalize depths and recompute path rewards
        root->parent_->depth_ = -1;
        root->normalize_depth();
        root->reset_frame_rep_counters(this->frameskip_);
        root->recompute_path_rewards(root);

        // construct/extend lookahead tree
        if( int(root->num_nodes()) < nodes_threshold_ ) {
            bfs(root, novelty_table_map);
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
                          << " value=" << root->value_
                          << ", imm-reward=" << root->reward_
                          << ", children=[";
            for( Node<Environment> *child = root->first_child_; child != nullptr; child = child->sibling_ )
                Logger::Continuation(Logger::Debug) << child->value_ << ":" << child->action_ << " ";
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

    // breadth-first search with ties broken in favor of bigger path reward
    struct NodeComparator {
        bool break_ties_using_rewards_;
        NodeComparator(bool break_ties_using_rewards) : break_ties_using_rewards_(break_ties_using_rewards) {
        }
        bool operator()(const Node<Environment> *lhs, const Node<Environment> *rhs) const {
            return
              (lhs->depth_ > rhs->depth_) ||
              (break_ties_using_rewards_ && (lhs->depth_ == rhs->depth_) && (lhs->path_reward_ < rhs->path_reward_));
        }
    };

    void bfs(Node<Environment> *root, std::map<int, std::vector<int> > &novelty_table_map) {
        // priority queue
        NodeComparator cmp(break_ties_using_rewards_);
        std::priority_queue<Node<Environment>*, std::vector<Node<Environment>*>, NodeComparator> q(cmp);

        // add tip nodes to queue
        add_tip_nodes_to_queue(root, q);
        Logger::Info << "queue: sz=" << q.size() << std::endl;

        // save current observation
        assert(root->observation_ != nullptr);
        typename Environment::Observation current_observation = *(root->observation_);
        //this->save_observation(current_observation);
        this->save_environment();

        // explore in breadth-first manner
        double start_time = Utils::read_time_in_seconds();
        while( !q.empty() && (int(this->simulator_calls_) < this->simulator_budget_) && (Utils::read_time_in_seconds() - start_time < this->time_budget_) ) {
            Node<Environment> *node = q.top();
            q.pop();

            // print debug info
            Logger::Continuation(Logger::Debug) << node->depth_ << "@" << node->path_reward_ << std::flush;

            // update node info
            assert((node->num_children_ == 0) && (node->first_child_ == nullptr));
            assert(node->visited_ || (node->is_info_valid_ != 2));
            if( node->is_info_valid_ != 2 ) {
                assert(node->parent_->observation_ != nullptr);
                if (!(typename Environment::ObservationEqual()(*(node->parent_->observation_), current_observation))) {
                    //this->save_observation(current_observation);
                    //this->restore_observation(*(node->parent_->observation_));
                    this->save_environment();
                    this->restore_environment();
                } 
                this->update_info(node);
                current_observation = *(node->observation_);
                assert((node->num_children_ == 0) && (node->first_child_ == nullptr));
                node->visited_ = true;
            }

            // check termination at this node
            if( node->terminal_ ) {
                Logger::Continuation(Logger::Debug) << "t" << "," << std::flush;
                continue;
            }

            // verify max repetitions of feature atoms (screen mode)
            if( node->frame_rep_ > int(max_rep_) ) {
                Logger::Continuation(Logger::Debug) << "r" << node->frame_rep_ << "," << std::flush;
                continue;
            }

            // calculate novelty and prune
            if( node->frame_rep_ == 0 ) {
                // calculate novelty
                std::vector<int> &novelty_table = this->get_novelty_table(node, novelty_table_map, novelty_subtables_);
                int atom = this->get_novel_atom(node->depth_, node->feature_atoms_, novelty_table);
                assert((atom >= 0) && (atom < int(novelty_table.size())));

                // prune node using novelty
                if( novelty_table[atom] <= node->depth_ ) {
                    Logger::Continuation(Logger::Debug) << "p" << "," << std::flush;
                    continue;
                }

                // update novelty table
                this->update_novelty_table(node->depth_, node->feature_atoms_, novelty_table);
            }
            Logger::Continuation(Logger::Debug) << "+" << std::flush;

            // expand node
            if( node->frame_rep_ == 0 ) {
                ++num_expansions_;
                double start_time = Utils::read_time_in_seconds();
                node->expand(this->action_set_, false);
                expand_time_ += Utils::read_time_in_seconds() - start_time;
            } else {
                assert(node->parent_ != nullptr);
                node->expand(node->action_);
            }
            assert((node->num_children_ > 0) && (node->first_child_ != nullptr));
            Logger::Continuation(Logger::Debug) << node->num_children_ << "," << std::flush;

            // add children to queue
            for( Node<Environment> *child = node->first_child_; child != nullptr; child = child->sibling_ )
                q.push(child);
        }
        Logger::Continuation(Logger::Debug) << std::endl;
    }

    void add_tip_nodes_to_queue(Node<Environment> *node, std::priority_queue<Node<Environment>*, std::vector<Node<Environment>*>, NodeComparator> &pq) const {
        std::deque<Node<Environment>*> q;
        q.push_back(node);
        while( !q.empty() ) {
            Node<Environment> *n = q.front();
            q.pop_front();
            if( n->num_children_ == 0 ) {
                assert(n->first_child_ == nullptr);
                pq.push(n);
            } else {
                assert(n->first_child_ != nullptr);
                for( Node<Environment> *child = n->first_child_; child != nullptr; child = child->sibling_ )
                    q.push_back(child);
            }
        }
    }

    void reset_stats() const {
        SimPlanner<Environment>::reset_stats();
        num_expansions_ = 0;
        total_time_ = 0;
        expand_time_ = 0;
        root_height_ = 0;
        random_decision_ = false;
    }

    void print_stats(Logger::mode_t logger_mode, const Node<Environment> &root, const std::map<int, std::vector<int> > &novelty_table_map) const {
        logger_mode << "decision-stats:"
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
          << " #sim=" << this->simulator_calls_
          << " total-time=" << total_time_
          << " simulator-time=" << this->sim_time_
          << " reset-time=" << this->sim_reset_time_
          << " get/set-state-time=" << this->sim_save_environment_time_
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
        << " break-ties-using-rewards=" << break_ties_using_rewards_;
    }
};

#endif

