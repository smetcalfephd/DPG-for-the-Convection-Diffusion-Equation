#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dg_vector.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/numerics/data_out.h>

#include <iostream>
#include <fstream>

using namespace dealii;

// Convection function.

template <int dim>
class Convection :  public Function<dim>
{
public:
    Convection ();

    virtual void vector_value (const Point<dim> &p, Vector<double> &values) const;
    virtual void vector_value_list (const std::vector<Point<dim> > &points, std::vector<Vector<double> > &value_list) const;
};

template <int dim>
Convection<dim>::Convection () : Function<dim> (dim)
{}

template <int dim>
inline
void Convection<dim>::vector_value (const Point<dim> &p, Vector<double> &values) const
{
Assert (values.size() == dim, ExcDimensionMismatch (values.size(), dim));
// Use p(0), p(1), p(2) to denote x, y and z.

switch(dim)
{
case 1: values(0) = 1; break;
case 2: values(0) = 1; values(1) = 1; break;
case 3: values(0) = 1; values(1) = 1; values(2) = 1; break;
}
}

template <int dim>
void Convection<dim>::vector_value_list (const std::vector<Point<dim> > &points, std::vector<Vector<double> > &value_list) const
{
const unsigned int no_of_points = points.size();

Assert (value_list.size() == no_of_points, ExcDimensionMismatch (value_list.size(), no_of_points));

    for (unsigned int p = 0; p < no_of_points; ++p)
    Convection<dim>::vector_value (points[p], value_list[p]);
}

// Forcing function.

template <int dim>
class Forcing:  public Function<dim>
{
public:
    Forcing () : Function<dim>() {};

virtual void value_list (const std::vector<Point<dim> > &points,std::vector<double> &values, const unsigned int component = 0) const;};

template <int dim>
void Forcing<dim>::value_list(const std::vector<Point<dim> > &points, std::vector<double> &values, const unsigned int) const
{
Assert(values.size() == points.size(), ExcDimensionMismatch(values.size(),points.size()));
// Use points[i](0), points[i](1), points[i](2) to denote x, y and z.

    for (unsigned int i = 0; i < values.size(); ++i)
    values[i] = 1;
}

template <int dim>
class ConvectionDiffusionDPG
{
public:
  	ConvectionDiffusionDPG ();
    void run ();
    
    double epsilon = 0.01; // Diffusion coefficient.
    unsigned int degree = 2; // Polynomial degree of the trial space.
	unsigned int degree_offset = 2; // The amount by which we offset the polynomial degree of the test space.
    
private:
    void setup_system (); // Set up the system.
	void assemble_system (); // Assemble the system.
	void interior_solve (); // Solve the system. 
	void trace_solve (); // Solve the system. 
	void output_solution () const; // Output the solution.
    void compute_local_bilinear_form_values (const typename DoFHandler<dim>::active_cell_iterator &trial_cell, const typename DoFHandler<dim>::active_cell_iterator &test_cell, const FEValues<dim> &fe_values_trial_cell, const FEValues<dim> &fe_values_test_cell, FEFaceValues<dim> &fe_values_trial_face, FEFaceValues<dim> &fe_values_test_face, Vector<double> &local_bilinear_form_values); // Intermediate function needed in assemble_system to assemble the system matrix.
    void compute_local_optimal_test_functions (const typename DoFHandler<dim>::active_cell_iterator &test_cell, const FEValues<dim> &fe_values_test_cell, const Vector<double> &local_bilinear_form_values, Vector<double> &local_optimal_test_functions); // Intermediate function needed in assemble_system to compute the optimal test functions. 

    Triangulation<dim> triangulation;
    FESystem<dim> fe_trial_cell, fe_trial_face, fe_trial, fe_test;
	DoFHandler<dim> dof_handler_trial_cell, dof_handler_trial_face, dof_handler_trial, dof_handler_test;

	ConstraintMatrix trace_constraints;
	SparsityPattern trace_sparsity_pattern;

	SparseMatrix<double> trace_system_matrix;
	Vector<double> interior_solution, skeleton_solution, trace_right_hand_side, derp, local_optimal_test_functions, local_bilinear_form_values;
};

// The trial space consists of the following (in order): a scalar corresponding to u, a vector corresponding to grad(u), a face term corresponding to trace(u) and a face term corresponding to the flux.
// The test space consists of the following (in order): a scalar test function corresponding to u and a vector test function corresponding to grad(u).
// The method also seems to work if you replace {FE_RaviartThomas<dim>(degree - 1), 1} by {FE_Q<dim>(degree), dim} as long as you also replace {FE_DGRaviartThomas<dim>(degree - 1 + degree_offset), 1} by {FE_DGQ<dim>(degree), dim}.
// IMPORTANT: For dim = 1, FE_RaviartThomas<dim>(degree - 1) in the trial space and FE_DGRaviartThomas<dim>(degree + degree_offset - 1) need to be replaced by FE_Q<dim>(degree - 1) and FE_DGQ<dim>(degree + degree_offset - 1) respectively as RT spaces are undefined for dim = 1.

template <int dim>
ConvectionDiffusionDPG<dim>::ConvectionDiffusionDPG ()
                :
				fe_trial_cell (FE_DGQ<dim>(degree), 1, FE_DGRaviartThomas<dim>(degree - 1), 1), fe_trial_face (FE_FaceQ<dim>(degree + 1), 1, FE_FaceQ<dim>(degree), 1),
                fe_trial (fe_trial_cell, 1, fe_trial_face, 1), fe_test (FE_DGQ<dim>(degree + degree_offset), 1, FE_DGRaviartThomas<dim>(degree - 1 + degree_offset), 1),
				dof_handler_trial_cell (triangulation), dof_handler_trial_face (triangulation), 
				dof_handler_trial (triangulation), dof_handler_test (triangulation)

{}

// Setup the system

template <int dim>
void ConvectionDiffusionDPG<dim>::setup_system ()
{
dof_handler_trial_cell.distribute_dofs (fe_trial_cell); dof_handler_trial_face.distribute_dofs (fe_trial_face);
dof_handler_trial.distribute_dofs (fe_trial); dof_handler_test.distribute_dofs (fe_test);

const unsigned int no_of_interior_trial_dofs = dof_handler_trial_cell.n_dofs();
const unsigned int no_of_skeleton_trial_dofs = dof_handler_trial_face.n_dofs();
const unsigned int no_of_trial_dofs_per_cell = fe_trial.dofs_per_cell;
const unsigned int no_of_test_dofs_per_cell = fe_test.dofs_per_cell;

const FEValuesExtractors::Scalar index (0);
const ComponentMask comp_mask = fe_trial_face.component_mask (index);

trace_constraints.clear ();
DoFTools::make_hanging_node_constraints (dof_handler_trial_face, trace_constraints);
DoFTools::make_zero_boundary_constraints (dof_handler_trial_face, trace_constraints, comp_mask); // Apply zero boundary conditions to the trace of u.
trace_constraints.close ();

DynamicSparsityPattern skeleton_dsp (no_of_skeleton_trial_dofs);
DoFTools::make_sparsity_pattern (dof_handler_trial_face, skeleton_dsp);
trace_constraints.condense (skeleton_dsp);
trace_sparsity_pattern.copy_from (skeleton_dsp);

trace_system_matrix.reinit (trace_sparsity_pattern);
interior_solution.reinit(no_of_interior_trial_dofs);
skeleton_solution.reinit (no_of_skeleton_trial_dofs);
trace_right_hand_side.reinit (no_of_skeleton_trial_dofs);
derp.reinit (no_of_trial_dofs_per_cell*no_of_test_dofs_per_cell*triangulation.n_active_cells());
local_bilinear_form_values.reinit (no_of_trial_dofs_per_cell*no_of_test_dofs_per_cell);
local_optimal_test_functions.reinit (no_of_trial_dofs_per_cell*no_of_test_dofs_per_cell);
}

// Assemble the system

template <int dim>
void ConvectionDiffusionDPG<dim>::assemble_system ()
{
const QGauss<dim>  quadrature_formula_cell (degree+degree_offset+1);
const QGauss<dim-1>  quadrature_formula_face (degree+degree_offset+1);

const unsigned int no_of_quad_points_cell = quadrature_formula_cell.size();
const unsigned int no_of_trial_dofs_per_cell = fe_trial.dofs_per_cell;
const unsigned int no_of_interior_trial_dofs_per_cell = fe_trial_cell.dofs_per_cell;
const unsigned int no_of_face_trial_dofs_per_cell = fe_trial_face.dofs_per_cell;
const unsigned int no_of_test_dofs_per_cell = fe_test.dofs_per_cell;

typename DoFHandler<dim>::active_cell_iterator trial_cell = dof_handler_trial.begin_active(), final_cell = dof_handler_trial.end();
typename DoFHandler<dim>::active_cell_iterator trial_cell_interior = dof_handler_trial_cell.begin_active();
typename DoFHandler<dim>::active_cell_iterator trial_cell_trace = dof_handler_trial_face.begin_active();
typename DoFHandler<dim>::active_cell_iterator test_cell = dof_handler_test.begin_active();

FEValues<dim> fe_values_trial_cell (fe_trial_cell, quadrature_formula_cell, update_values | update_quadrature_points | update_JxW_values);
FEValues<dim> fe_values_test_cell (fe_test, quadrature_formula_cell, update_values | update_gradients | update_quadrature_points | update_JxW_values);
FEFaceValues<dim> fe_values_trial_face (fe_trial_face, quadrature_formula_face, update_values | update_quadrature_points | update_JxW_values);
FEFaceValues<dim> fe_values_test_face (fe_test, quadrature_formula_face, update_values | update_quadrature_points | update_JxW_values | update_normal_vectors);
   
FullMatrix<double> local_system_matrix (no_of_face_trial_dofs_per_cell, no_of_face_trial_dofs_per_cell);
FullMatrix<double> local_system_submatrix_00 (no_of_interior_trial_dofs_per_cell, no_of_interior_trial_dofs_per_cell);
FullMatrix<double> local_system_submatrix_01 (no_of_interior_trial_dofs_per_cell, no_of_face_trial_dofs_per_cell);
FullMatrix<double> local_system_submatrix_10 (no_of_face_trial_dofs_per_cell, no_of_interior_trial_dofs_per_cell);
FullMatrix<double> local_system_submatrix_11 (no_of_face_trial_dofs_per_cell, no_of_face_trial_dofs_per_cell);
FullMatrix<double> intermediate_matrix (no_of_face_trial_dofs_per_cell, no_of_interior_trial_dofs_per_cell);
FullMatrix<double> intermediate_matrix1 (no_of_interior_trial_dofs_per_cell, no_of_face_trial_dofs_per_cell);
Vector<double> local_interior_solution (no_of_interior_trial_dofs_per_cell);
Vector<double> local_right_hand_side (no_of_interior_trial_dofs_per_cell);
Vector<double> local_right_hand_side_boundary_conditions (no_of_face_trial_dofs_per_cell);
Vector<double> intermediate_vector (no_of_face_trial_dofs_per_cell);
std::vector<double> forcing_values (no_of_quad_points_cell);
std::vector<types::global_dof_index> local_dof_indices_trial_cell (no_of_interior_trial_dofs_per_cell);
std::vector<types::global_dof_index> local_dof_indices_trial_face (no_of_face_trial_dofs_per_cell);

const FEValuesExtractors::Scalar index (0);
const ComponentMask comp_mask = fe_trial_face.component_mask (index);
std::vector<bool> boundary_dofs(dof_handler_trial_face.n_dofs());
DoFTools::extract_boundary_dofs(dof_handler_trial_face, comp_mask, boundary_dofs);

    for (; trial_cell!=final_cell; ++trial_cell, ++trial_cell_interior, ++trial_cell_trace, ++test_cell)
    {
	fe_values_trial_cell.reinit (trial_cell_interior);
	fe_values_test_cell.reinit (test_cell);

    compute_local_bilinear_form_values (trial_cell_trace, test_cell, fe_values_trial_cell, fe_values_test_cell, fe_values_trial_face, fe_values_test_face, local_bilinear_form_values);
	compute_local_optimal_test_functions (test_cell, fe_values_test_cell, local_bilinear_form_values, local_optimal_test_functions);

	Forcing<dim>().value_list (fe_values_test_cell.get_quadrature_points(), forcing_values);

	trial_cell_interior->get_dof_indices (local_dof_indices_trial_cell);
	trial_cell_trace->get_dof_indices (local_dof_indices_trial_face);

	local_system_matrix = 0; local_system_submatrix_00 = 0; local_system_submatrix_01 = 0; local_system_submatrix_10 = 0; local_system_submatrix_11 = 0; local_right_hand_side = 0; local_right_hand_side_boundary_conditions = 0; intermediate_matrix *= 0; intermediate_vector = 0;

	    for (unsigned int i = 0; i < no_of_trial_dofs_per_cell; ++i)
		{
		unsigned int comp_i = fe_trial.system_to_base_index(i).first.first;
		unsigned int basis_i = fe_trial.system_to_base_index(i).second;

	        for (unsigned int j = 0; j < no_of_trial_dofs_per_cell; ++j)
			{
			unsigned int comp_j = fe_trial.system_to_base_index(j).first.first;
		    unsigned int basis_j = fe_trial.system_to_base_index(j).second;

			if (comp_i == 0 && comp_j == 0)
			{
			    for (unsigned int k = 0; k < no_of_test_dofs_per_cell; ++k)
				{
				local_system_submatrix_00(basis_i,basis_j) += local_optimal_test_functions(k + i*no_of_test_dofs_per_cell)*local_bilinear_form_values(k + j*no_of_test_dofs_per_cell);
				}
			}

			if (comp_i == 0 && comp_j == 1)
			{
			    for (unsigned int k = 0; k < no_of_test_dofs_per_cell; ++k)
				{
				local_system_submatrix_01(basis_i,basis_j) += local_optimal_test_functions(k + i*no_of_test_dofs_per_cell)*local_bilinear_form_values(k + j*no_of_test_dofs_per_cell);
				}
			}

			if (comp_i == 1 && comp_j == 0)
			{
			    for (unsigned int k = 0; k < no_of_test_dofs_per_cell; ++k)
				{
				local_system_submatrix_10(basis_i,basis_j) += local_optimal_test_functions(k + i*no_of_test_dofs_per_cell)*local_bilinear_form_values(k + j*no_of_test_dofs_per_cell);
				}
			}

			if (comp_i == 1 && comp_j == 1)
			{
			    for (unsigned int k = 0; k < no_of_test_dofs_per_cell; ++k)
				{
				local_system_submatrix_11(basis_i,basis_j) += local_optimal_test_functions(k + i*no_of_test_dofs_per_cell)*local_bilinear_form_values(k + j*no_of_test_dofs_per_cell);
				}
			}

			}
		}

		// Assemble the local contributions to the system matrix and the right hand side vector.
		for (unsigned int k = 0; k < no_of_test_dofs_per_cell; ++k)
		{
		unsigned int comp_k = fe_test.system_to_base_index(k).first.first;

		if (comp_k == 0)
		{
		    for (unsigned int quad_point = 0; quad_point < no_of_quad_points_cell; ++quad_point)
			{
			double test_cell_value = forcing_values[quad_point]*fe_values_test_cell.shape_value_component(k,quad_point,0)*fe_values_test_cell.JxW(quad_point);
			    
				for (unsigned int i = 0; i < no_of_trial_dofs_per_cell; ++i)
	            {
				unsigned int comp_i = fe_trial.system_to_base_index(i).first.first;
		        unsigned int basis_i = fe_trial.system_to_base_index(i).second;

				if (comp_i == 0) {local_right_hand_side(basis_i) += local_optimal_test_functions(k + i*no_of_test_dofs_per_cell)*test_cell_value;} else {local_right_hand_side_boundary_conditions(basis_i) += local_optimal_test_functions(k + i*no_of_test_dofs_per_cell)*test_cell_value;}
			    }
			}
		}
		}

		local_system_submatrix_00.gauss_jordan();
		local_system_submatrix_10.mmult(intermediate_matrix, local_system_submatrix_00);
		intermediate_matrix *= -1;

		intermediate_matrix.vmult(intermediate_vector, local_right_hand_side);
		local_right_hand_side_boundary_conditions.add(1, intermediate_vector);

		intermediate_matrix.mmult(local_system_matrix, local_system_submatrix_01);
		local_system_matrix.add(1, local_system_submatrix_11);

	    // Place the local system matrix contributions in the system matrix and the local right hand side contributions in the right hand side vector.
        for (unsigned int i = 0; i < no_of_face_trial_dofs_per_cell; ++i)
	    {
		    for (unsigned int j = 0; j < no_of_face_trial_dofs_per_cell; ++j)
		    {
		    trace_system_matrix(local_dof_indices_trial_face[i], local_dof_indices_trial_face[j]) += local_system_matrix(i,j);
		    }

     	trace_right_hand_side(local_dof_indices_trial_face[i]) += local_right_hand_side_boundary_conditions(i);
	    }

		local_system_submatrix_00.vmult (local_interior_solution, local_right_hand_side);

		// Place the local system matrix contributions in the system matrix and the local right hand side contributions in the right hand side vector.
        for (unsigned int i = 0; i < no_of_interior_trial_dofs_per_cell; ++i)
	    {
     	interior_solution(local_dof_indices_trial_cell[i]) += local_interior_solution(i);
	    }

		local_system_submatrix_00.mmult(intermediate_matrix1, local_system_submatrix_01); 

		unsigned int cell_no = trial_cell->active_cell_index();
		unsigned int hodor = no_of_interior_trial_dofs_per_cell*no_of_face_trial_dofs_per_cell*cell_no;

		for (unsigned int i = 0; i < no_of_interior_trial_dofs_per_cell; ++i)
		    for (unsigned int j = 0; j < no_of_face_trial_dofs_per_cell; ++j)
			{
			derp(i + j*no_of_interior_trial_dofs_per_cell + hodor) = intermediate_matrix1(i,j);
			}
    }
}

// Solve the system. 
// Using a CG solver since DPG yields a symmetric positive definite matrix. TODO: Work on preconditioner.

template <int dim>
void ConvectionDiffusionDPG<dim>::interior_solve ()
{
const unsigned int no_of_interior_trial_dofs_per_cell = fe_trial_cell.dofs_per_cell;
const unsigned int no_of_face_trial_dofs_per_cell = fe_trial_face.dofs_per_cell;

typename DoFHandler<dim>::active_cell_iterator trial_cell = dof_handler_trial_cell.begin_active(), final_cell = dof_handler_trial_cell.end();
typename DoFHandler<dim>::active_cell_iterator trial_cell_trace = dof_handler_trial_face.begin_active();

std::vector<types::global_dof_index> local_dof_indices_trial_cell (no_of_interior_trial_dofs_per_cell);
std::vector<types::global_dof_index> local_dof_indices_trial_face (no_of_face_trial_dofs_per_cell);

Vector<double> local_interior_solution(no_of_interior_trial_dofs_per_cell);
Vector<double> rhs_values(no_of_face_trial_dofs_per_cell);

    for (; trial_cell!=final_cell; ++trial_cell, ++trial_cell_trace)
    {
	trial_cell->get_dof_indices (local_dof_indices_trial_cell);
	trial_cell_trace->get_dof_indices (local_dof_indices_trial_face);

	unsigned int cell_no = trial_cell->active_cell_index();
	unsigned int hodor = no_of_interior_trial_dofs_per_cell*no_of_face_trial_dofs_per_cell*cell_no;

	local_interior_solution = 0;

	    for (unsigned int i = 0; i < no_of_face_trial_dofs_per_cell; ++i)
		{
		rhs_values(i) = skeleton_solution(local_dof_indices_trial_face[i]);
		}

        for (unsigned int i = 0; i < no_of_interior_trial_dofs_per_cell; ++i)
		    for (unsigned int j = 0; j < no_of_face_trial_dofs_per_cell; ++j)
		    {
		    local_interior_solution(i) += derp(i + j*no_of_interior_trial_dofs_per_cell + hodor)*rhs_values(j);
		    }

        for (unsigned int i = 0; i < no_of_interior_trial_dofs_per_cell; ++i)
		{
		interior_solution(local_dof_indices_trial_cell[i]) -= local_interior_solution(i);
		}

	}
}

template <int dim>
void ConvectionDiffusionDPG<dim>::trace_solve ()
{
SolverControl solver_control (10000, 1e-8);
SolverCG<> solver (solver_control);

trace_constraints.condense (trace_system_matrix, trace_right_hand_side);

SparseILU<double> ilu;
ilu.initialize (trace_system_matrix);
solver.solve (trace_system_matrix, skeleton_solution, trace_right_hand_side, ilu);

trace_constraints.distribute (skeleton_solution);
}

// Output the solution. 
// Currently outputting: GNUPLOT.

template <int dim>
void ConvectionDiffusionDPG<dim>::output_solution () const
{
std::vector<std::string> solution_names (1, "solution");
for (unsigned int d = 0; d < dim; ++d) {solution_names.push_back ("gradient");}

std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation (1, DataComponentInterpretation::component_is_scalar);
for (unsigned int d = 0; d < dim; ++d) {data_component_interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);}

DataOut<dim> data_out;
data_out.attach_dof_handler (dof_handler_trial_cell);
data_out.add_data_vector (interior_solution, solution_names, DataOut<dim>::type_dof_data, data_component_interpretation);
data_out.build_patches ();

const std::string filename = "solution.gnuplot";

std::ofstream gnuplot_output (filename.c_str());
data_out.write_gnuplot (gnuplot_output);
}

// Intermediate function needed in assemble_system to assemble the system matrix. 
// Takes current cell information as an input and outputs a vector whose k + i*no_of_test_dofs_per_cell entry corresponds to B(phi_i, psi_k) where {phi_i} is a local basis for U_h and {psi_k} is a local basis for the the full enriched space V_h.
// B({u,sigma,trace(u),trace(sigma)},{v,tau}) := (u, div(tau) - b*grad(v))_K + (sigma, tau/epsilon + grad(v))_K - (trace(u), tau*n)_dK + (trace(sigma), v)_dK.
// TODO: Work on efficiency of computation of the face terms.

template <int dim>
void ConvectionDiffusionDPG<dim>::compute_local_bilinear_form_values (const typename DoFHandler<dim>::active_cell_iterator &trial_cell, const typename DoFHandler<dim>::active_cell_iterator &test_cell, const FEValues<dim> &fe_values_trial_cell, const FEValues<dim> &fe_values_test_cell, FEFaceValues<dim> &fe_values_trial_face, FEFaceValues<dim> &fe_values_test_face, Vector<double> &local_bilinear_form_values)
{
const unsigned int no_of_quad_points_cell = fe_values_trial_cell.get_quadrature().size();
const unsigned int no_of_quad_points_face = fe_values_trial_face.get_quadrature().size();
const unsigned int no_of_total_trial_dofs_per_cell = fe_trial.dofs_per_cell;
const unsigned int no_of_interior_trial_dofs_per_cell = fe_trial_cell.dofs_per_cell;
const unsigned int no_of_boundary_trial_dofs_per_cell = fe_trial_face.dofs_per_cell;
const unsigned int no_of_test_dofs_per_cell = fe_test.dofs_per_cell;

Vector<double> cell_values (no_of_interior_trial_dofs_per_cell*no_of_test_dofs_per_cell);
Vector<double> face_values (no_of_boundary_trial_dofs_per_cell*no_of_test_dofs_per_cell);
std::vector<Vector<double> > convection_values (no_of_quad_points_cell, Vector<double>(dim));
Convection<dim>().vector_value_list (fe_values_test_cell.get_quadrature_points(), convection_values);

local_bilinear_form_values = 0;

    for (unsigned int quad_point = 0; quad_point < no_of_quad_points_cell; ++quad_point)
	{
	Tensor<1,dim> convection;

	    for (unsigned int d = 0; d < dim; ++d)
	    {
	    convection[d] = convection_values[quad_point](d);
	    }

	    for (unsigned int k = 0; k < no_of_test_dofs_per_cell; ++k)
		{
		Vector<double> values(dim+1);

		values(0) =  -convection*fe_values_test_cell.shape_grad_component(k,quad_point,0);

	        for (unsigned int d = 0; d < dim; ++d)
		    {
			values(0) += fe_values_test_cell.shape_grad_component(k,quad_point,d+1)[d];
			values(d+1) = fe_values_test_cell.shape_value_component(k,quad_point,d+1) + epsilon*fe_values_test_cell.shape_grad_component(k,quad_point,0)[d];
		    }

        values *= fe_values_test_cell.JxW(quad_point);

		    for (unsigned int i = 0; i < no_of_interior_trial_dofs_per_cell; ++i)
			    for (unsigned int d = 0; d < dim + 1; ++d)
		        {
				cell_values(k + i*no_of_test_dofs_per_cell) += fe_values_trial_cell.shape_value_component(i,quad_point,d)*values(d);
				}
	    }
    }

	for (unsigned int face = 0; face < 2*dim; ++face)
    {
	fe_values_trial_face.reinit(trial_cell, face); fe_values_test_face.reinit(test_cell, face);
	
	const std::vector<Tensor<1,dim> > &normals = fe_values_test_face.get_normal_vectors();

    std::vector<unsigned int> trial_index, test_index;

	    for (unsigned int i = 0; i < no_of_boundary_trial_dofs_per_cell; ++i)
	    {
	    if (fabs(fe_values_trial_face.shape_value_component(i,0,0)) > 1e-14 || fabs(fe_values_trial_face.shape_value_component(i,0,1)) > 1e-14) {trial_index.push_back(i);}
	    }

	    for (unsigned int i = 0; i < no_of_test_dofs_per_cell; ++i)
	    {
	    bool index_check = false;

	        for (unsigned int d = 0; d < dim + 1; ++d)
	        {
	        if (fabs(fe_values_test_face.shape_value_component(i,0,d)) > 1e-14) {index_check = true;}
	        }

	    if (index_check == true) {test_index.push_back(i);}
	    } 

	const unsigned int no_of_trial_dofs_on_face = trial_index.size(); const unsigned int no_of_test_dofs_on_face = test_index.size();

	    for (unsigned int quad_point = 0; quad_point < no_of_quad_points_face; ++quad_point)
		    for (unsigned int k = 0; k < no_of_test_dofs_on_face; ++k)
            {
            double taudotnormal = 0; double test_face_value = fe_values_test_face.shape_value_component(test_index[k],quad_point,0);

                for (unsigned int d = 0; d < dim; ++d)
                {
                taudotnormal += fe_values_test_face.shape_value_component(test_index[k],quad_point,d+1)*normals[quad_point][d];
                }
                
            taudotnormal *= fe_values_test_face.JxW(quad_point); test_face_value *= std::pow(-1,face+1)*fe_values_test_face.JxW(quad_point);

		        for (unsigned int i = 0; i < no_of_trial_dofs_on_face; ++i)
                {
				face_values(test_index[k] + trial_index[i]*no_of_test_dofs_per_cell) += fe_values_trial_face.shape_value_component(trial_index[i],quad_point,1)*test_face_value - fe_values_trial_face.shape_value_component(trial_index[i],quad_point,0)*taudotnormal;
                }
            }
    }
    
    for (unsigned int i = 0; i < no_of_total_trial_dofs_per_cell; ++i)
    {
    unsigned int comp_i = fe_trial.system_to_base_index(i).first.first;
    unsigned int basis_i = fe_trial.system_to_base_index(i).second;
    
    if (comp_i == 0)
    {
        for (unsigned int k = 0; k < no_of_test_dofs_per_cell; ++k)
        {
        local_bilinear_form_values(k + i*no_of_test_dofs_per_cell) = cell_values(k + basis_i*no_of_test_dofs_per_cell);
        }
    }
    else
    {
        for (unsigned int k = 0; k < no_of_test_dofs_per_cell; ++k)
        {
        local_bilinear_form_values(k + i*no_of_test_dofs_per_cell) = face_values(k + basis_i*no_of_test_dofs_per_cell);
        }
    }
    }
}

// Intermediate function needed in assemble_system to compute the optimal test functions. 
// Takes current cell information and bilinear form values as an input and outputs a vector whose k + i*no_of_test_dofs_per_cell entry corresponds to the weighting of the {psi_k} local basis function in the local basis expansion of the ith test function in the full enriched space V_h.

template <int dim>
void ConvectionDiffusionDPG<dim>::compute_local_optimal_test_functions (const typename DoFHandler<dim>::active_cell_iterator &test_cell, const FEValues<dim> &fe_values_test_cell, const Vector<double> &local_bilinear_form_values, Vector<double> &local_optimal_test_functions)
{
const unsigned int no_quad_points_cell = fe_values_test_cell.get_quadrature().size();
const unsigned int no_of_trial_dofs_per_cell = fe_trial.dofs_per_cell;
const unsigned int no_of_test_dofs_per_cell = fe_test.dofs_per_cell;

FullMatrix<double> V_basis_matrix (no_of_test_dofs_per_cell, no_of_test_dofs_per_cell);
Vector<double> U_basis_rhs (no_of_test_dofs_per_cell);
Vector<double> optimal_test_function (no_of_test_dofs_per_cell);
std::vector<Vector<double> > convection_values (no_quad_points_cell, Vector<double>(dim));

Convection<dim>().vector_value_list (fe_values_test_cell.get_quadrature_points(), convection_values);

double vol = test_cell->measure ();

// Compute the V basis matrix whose (i,j) entry corresponds to the local inner product associated with the norm ||v||^2 + ||div(tau) - b*grad(v)||^2 + ||C_K*tau + sqrt(epsilon)*grad(v)||^2 where C_K := min(1/|K|,1/sqrt(epsilon)).
// This is an improvement (I think) upon Jesse Chan's proposed norm for Convection-Diffusion problems (see his PhD Thesis). 
// Note: The proposed test norm is NOT good for the hyperbolic problem (epsilon = 0). This is a possible avenue for future research.

    for (unsigned int quad_point = 0; quad_point < no_quad_points_cell; ++quad_point)
    {
	Tensor<1,dim> convection;

	    for (unsigned int d = 0; d < dim; ++d)
	    {
	    convection[d] = convection_values[quad_point](d);
	    }
            
        for (unsigned int k = 0; k < no_of_test_dofs_per_cell; ++k)
            for (unsigned int l = 0; l < k + 1; ++l)
            {
            V_basis_matrix(k,l) += fe_values_test_cell.shape_value_component(k,quad_point,0)*fe_values_test_cell.shape_value_component(l,quad_point,0)*fe_values_test_cell.JxW(quad_point);
            
			double divtau = 0; double divsigma = 0;

			    for (unsigned int d = 0; d < dim; ++d)
				{
				divtau += fe_values_test_cell.shape_grad_component(k,quad_point,d+1)[d];
				divsigma += fe_values_test_cell.shape_grad_component(l,quad_point,d+1)[d];

				V_basis_matrix(k,l) += ((1/epsilon)*fe_values_test_cell.shape_value_component(k,quad_point,d+1)+fe_values_test_cell.shape_grad_component(k,quad_point,0)[d])*((1/epsilon)*fe_values_test_cell.shape_value_component(l,quad_point,d+1)+fe_values_test_cell.shape_grad_component(l,quad_point,0)[d])*fe_values_test_cell.JxW(quad_point);
				}

			V_basis_matrix(k,l) += (divtau - fe_values_test_cell.shape_grad_component(k,quad_point,0)*convection)*(divsigma - fe_values_test_cell.shape_grad_component(l,quad_point,0)*convection)*fe_values_test_cell.JxW(quad_point);
            }
    } 
           
    for (unsigned int k = 0; k < no_of_test_dofs_per_cell; ++k)
        for (unsigned int l = 0; l < k + 1; ++l)
        {
        V_basis_matrix(l,k) = V_basis_matrix(k,l);
        }

V_basis_matrix.gauss_jordan(); // Invert the V basis matrix in preparation for finding the optimal test function basis coefficients.

    for (unsigned int i = 0; i < no_of_trial_dofs_per_cell; ++i)
    {  	
	// Set the right hand side whose kth entry is B(phi_i, psi_k) where {phi_i} is a local basis for U_h and {psi_k} is a local basis for the the full enriched space V_h.
        for (unsigned int k = 0; k < no_of_test_dofs_per_cell; ++k)
        {
        U_basis_rhs(k) = local_bilinear_form_values(k + i*no_of_test_dofs_per_cell);
        }

    // Solve the linear system corresponding to (varphi_{i,k}, psi_k)_V = B(phi_i, psi_k) for varphi_{i,k} where varphi_{i,k} is a vector of unknowns representing the coefficients of the ith local optimal test function in the local basis {psi_k} of the the full enriched space V_h and phi_i is a (fixed) local basis function for U_h.
    V_basis_matrix.vmult (optimal_test_function, U_basis_rhs); 

	    for (unsigned int k = 0; k < no_of_test_dofs_per_cell; ++k)
        {
        local_optimal_test_functions(k + i*no_of_test_dofs_per_cell) = optimal_test_function(k); 
        }
    }
}

template <int dim>
void ConvectionDiffusionDPG<dim>::run ()
{
    GridGenerator::hyper_cube (triangulation, -1, 1, true); triangulation.refine_global (2); // Creates the triangulation and globally refines n times.
    
	setup_system ();
    assemble_system ();
	trace_solve ();
	interior_solve ();
 	output_solution ();
}

int main ()
{
    deallog.depth_console (2);
	std::ofstream logfile ("deallog");
	deallog.attach (logfile);
	try
    {
      ConvectionDiffusionDPG<2> ConvDiffDPG;
      ConvDiffDPG.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    };

  return 0;
}