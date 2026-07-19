//
//  linear_program.cpp
//  spacetime_meshing_v2
//
//  Created by Christian Howard on 4/5/21.
//

#include "linear_program.hpp"
#include <cassert>
#define assertm(exp, msg) assert(((void)msg, exp))

namespace math {

    // ctor/dtor
linear_program::linear_program()
:dim(0), out_flag(-1), prob(nullptr), num_row_constraints(0),
ridx(1024), cidx(1024), values(1024), id_list(128),
do_maximize(true)
{
    prob = glp_create_prob();
    glp_init_smcp(&params);
    params.msg_lev = GLP_MSG_OFF;
    
}
linear_program::~linear_program() {
    if( prob ){
        glp_delete_prob(prob);
        prob = nullptr;
    }
}
    
    // helper methods for
    // setting up the linear program
void linear_program::set_dim(int dim_) {
    assertm(dim_ > 0, "dimension of LP is trivial");
    
    if( dim_ != dim ){
        
        // clear the minimum number of columns
        if( dim_ < dim ){
            
            // delete columns
            id_list.clear();
            
            // need to delete this many
            int num_del = dim - dim_;
            
            // clear rows at the end to minimize changes
            // to the ordinal numbering
            for(int i = 0; i < num_del; ++i){
                id_list.emplace_back(dim - i);
            }
            glp_del_cols(prob, num_del, id_list.data());
            
        }
        // or add new columns
        else{
            // add columns
            glp_add_cols(prob, (dim_ - dim) );
            int del = dim_ - dim;
            for(int i = 0; i < del; ++i){
                glp_set_col_bnds(prob, i+1, GLP_FR, 0.0, 0.0);
            }// end for i
            
        }
    }
    
    dim = dim_;
        
}

void linear_program::obj_is_maximized(bool is_maximized) {
    do_maximize = is_maximized;
}
    
    // Assumes constraints of form Ax <= b
void linear_program::set_matrix_constraints(const arma::mat& A, const arma::vec& b) {
    
    // get size info
    auto nr = A.n_rows;
    auto nc = A.n_cols;
    auto len= b.n_elem;
    
    // do some sanity checking assertions
    assertm(dim > 0, "dimension of LP is trivial");
    assertm(dim == nc, "constraint matrix does not match up to dimension of LP");
    assertm(nr == len, "A and b have different number of rows, which is wrong");
    
    // clear some rows, if needed
    if( nr < num_row_constraints ){
        
        // delete rows
        id_list.clear();
        
        // need to delete this many
        int num_del = num_row_constraints - nr;
        
        // clear rows at the end to minimize changes
        // to the ordinal numbering
        for(int i = 0; i < num_del; ++i){
            id_list.emplace_back(num_row_constraints - i);
        }
        glp_del_rows(prob, num_del, id_list.data());
        
    }
    // add some rows, if needed
    else if( nr > num_row_constraints ){
        glp_add_rows(prob, (nr - num_row_constraints) );
    }
    
    // set the number of row constraints to the new number
    num_row_constraints = nr;
    
    // construct the LP
    
    // add the rows aux vars
    for(int i = 0; i < nr; ++i){
        glp_set_row_bnds(prob, (i+1), GLP_UP, 0.0, b[i] );
    }
    
    // add the constraint matrix
    ridx.clear(); cidx.clear(); values.clear();
    ridx.emplace_back(0);
    cidx.emplace_back(0);
    values.emplace_back(0.0);
    for(int r = 0; r < nr; ++r){
        for(int c = 0; c < nc; ++c){
            ridx.emplace_back(r+1);
            cidx.emplace_back(c+1);
            values.emplace_back(A(r,c));
        }// end for c
    }// end for r
    int ne = static_cast<int>(ridx.size()) - 1;
    glp_load_matrix(prob, ne, ridx.data(), cidx.data(), values.data());
    
}

void linear_program::set_objective_vector(const arma::vec& c) {
    auto len= c.n_elem;
    
    // perform a few sanity checking assertions
    assertm(dim > 0, "dimension of LP is trivial");
    assertm(dim == len, "c should be equal to dimension of the problem");
    
    // set the objective function
    for(int i = 0; i < len; ++i){
        glp_set_obj_coef(prob, (i+1), c[i] );
    }
}
    
    // methods for solving linear program
void linear_program::solve() {
    int opt_type = GLP_MAX;
    if( !do_maximize ){
        opt_type = GLP_MIN;
    }
    glp_set_obj_dir(prob, opt_type);
    out_flag = glp_simplex(prob, &params);
    int a = 0;
}

bool linear_program::did_find_solution() const {
    return out_flag == 0;
}
void linear_program::get_solution(std::vector<double>& x_inout) const {
    
    assertm(dim > 0, "dimension of LP is trivial");
    
    // if the inout vector is not big enough, resize
    if( x_inout.size() != dim ){
        x_inout.resize(dim);
    }
    
    // populate the output
    for(int i = 0; i < dim; ++i){
        x_inout[i] = glp_get_col_prim(prob, (i+1));
    }
}
    
    // methods to refresh the linear program
void linear_program::clear() {
    glp_erase_prob(prob);
    dim = 0;
    num_row_constraints = 0;
}
    
    // GLPK LP problem handle
    //glp_prob *prob;
    
    // other data
    //size_t num_row_constraints;
    

}
