//
//  linear_program.hpp
//  spacetime_meshing_v2
//
//  Created by Christian Howard on 4/5/21.
//

#ifndef linear_program_hpp
#define linear_program_hpp

#include <glpk.h>
#include <limits>
#include <armadillo>

namespace math {

/**
 This class wraps the GLPK library written in C for computing
 a linear program
 */
class linear_program {
public:
    
    // ctor/dtor
    linear_program();
    ~linear_program();
    
    // helper methods for
    // setting up the linear program
    void set_dim(int dim);
    
    // set whether objective is maximized or minimized
    void obj_is_maximized(bool is_maximized);
    
    // Assumes constraints of form Ax <= b
    void set_matrix_constraints(const arma::mat& A, const arma::vec& b);
    
    // assumes objective is c^Tx
    void set_objective_vector(const arma::vec& c);
    
    // methods for solving linear program
    void solve();
    void get_solution(std::vector<double>& x_out) const;
    bool did_find_solution() const;
    
    // methods to refresh the linear program
    void clear();
    
private:
    
    // GLPK LP problem handle
    glp_prob *prob;
    glp_smcp params;
    
    // other data
    bool do_maximize;
    int dim, out_flag;
    size_t num_row_constraints;
    std::vector<int> id_list, ridx, cidx;
    std::vector<double> values;
};


}

#endif /* linear_program_hpp */
