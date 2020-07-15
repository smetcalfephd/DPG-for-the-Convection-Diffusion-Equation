#include <deal.II/base/tensor_function.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_trace.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <fstream>

using namespace dealii;

// Convection function.

template <int dim> class Convection : public TensorFunction<1, dim>
{
public: Convection () : TensorFunction<1, dim>() {}
     
virtual void value_list (const std::vector<Point<dim>> &points, std::vector<Tensor<1, dim>> &values) const override;
};

template <int dim> void Convection<dim>::value_list (const std::vector<Point<dim>> &points, std::vector<Tensor<1, dim>> &values) const
{
const unsigned int no_of_points = points.size();
// double x = 0; double y = 0; double z = 0;

    for (unsigned int point = 0; point < no_of_points; ++point)
	{
	// x = points[point][0]; switch(dim) {case 2: y = points[point][1]; break; case 3: y = points[point][1]; z = points[point][2]; break;}

	values[point][0] = 1;

    switch(dim)
    {
    case 2: values[point][1] = 1; break;
	case 3: values[point][1] = 1; values[point][2] = 1; break;
    }
	}
}

// Forcing function.

template <int dim> class Forcing:  public Function<dim>
{
public: Forcing () : Function<dim>() {};

virtual void value_list (const std::vector<Point<dim>> &points, std::vector<double> &values, const unsigned int component = 0) const;
};

template <int dim>
void Forcing<dim>::value_list (const std::vector<Point<dim>> &points, std::vector<double> &values, const unsigned int) const
{
const unsigned int no_of_points = points.size();
// double x = 0; double y = 0; double z = 0;

    for (unsigned int point = 0; point < no_of_points; ++point)
	{
	// x = points[point][0]; switch(dim) {case 2: y = points[point][1]; break; case 3: y = points[point][1]; z = points[point][2]; break;}

    values[point] = 1;
	}
}

// Dirichlet boundary values function enforcing u = g to be applied to all boundary points with indicator 0.

template <int dim> class DirichletBoundaryValues : public Function<dim>
{
public: virtual double value (const Point<dim> &point, const unsigned int) const override
{
// double x = 0; double y = 0; double z = 0;

// x = point[0]; switch(dim) {case 2: y = point[1]; break; case 3: y = point[1]; z = point[2]; break;}

return 0;
}
};

// Robin boundary values function enforcing (b*u - epsilon*grad(u))*n = g to be applied to all boundary points with indicator 1.

template <int dim> class RobinBoundaryValues : public Function<dim>
{
public: virtual double value (const Point<dim> &point, const unsigned int) const override
{
// double x = 0; double y = 0; double z = 0;

// x = point[0]; switch(dim) {case 2: y = point[1]; break; case 3: y = point[1]; z = point[2]; break;}

return 0;
}
};

template <int dim>
class ConvectionDiffusionDPG
{
public:
  	ConvectionDiffusionDPG ();
    void run ();
    
    const double epsilon = 0.01; // Diffusion coefficient.
    const unsigned int degree = 3; // Polynomial degree of the trial space.
	const unsigned int degree_offset = 2; // The amount by which we offset the polynomial degree of the test space.
	const unsigned int no_of_cycles = 30; // The maximum number of solution cycles.
	unsigned int cycle = 1; // The current solution cycle.
    
private:
    void setup_system (); // Set up the system.
	void create_constraints (); // Creates the constraints.
	void assemble_system (); // Assemble the system.
	void compute_bilinear_form_values (const typename DoFHandler<dim>::active_cell_iterator &trial_cell, const typename DoFHandler<dim>::active_cell_iterator &test_cell, const FEValues<dim> &fe_values_trial_cell, const FEValues<dim> &fe_values_test_cell, FEFaceValues<dim> &fe_values_trial_face, FEFaceValues<dim> &fe_values_test_face, const std::vector<Tensor<1,dim> > &convection_values, const std::vector<unsigned int> &additional_data); // Intermediate function needed in assemble_system to assemble the system matrix.
    void compute_local_optimal_test_functions (const FEValues<dim> &fe_values_test_cell, const std::vector<Tensor<1,dim> > &convection_values, const std::vector<unsigned int> &additional_data, const double &cell_size, std::vector<double> &local_optimal_test_functions); // Intermediate function needed in assemble_system to compute the optimal test functions. 
	void solve (); // Solve the system. 
	void output_solution () const; // Output the solution.
	void compute_error_estimator (); // Computes the error estimator.
	void refine_grid (); // Refines the mesh.

	Triangulation<dim> triangulation;
    FESystem<dim> fe_trial_interior, fe_trial_trace, fe_trial, fe_test;
	DoFHandler<dim> dof_handler_trial, dof_handler_test, dof_handler_trial_interior, dof_handler_trial_trace;

	ConstraintMatrix trace_constraints;
	SparsityPattern trace_sparsity_pattern;

	SparseMatrix<double> trace_system_matrix;
	Vector<double> interior_solution, trace_solution, trace_right_hand_side, refinement_vector;
	std::vector<double> local_optimal_test_functions, bilinear_form_values_storage, V_basis_matrix_inverse_storage, estimator_right_hand_side_storage, intermediate_matrix_storage;
};

// The trial space consists of the following (in order): a scalar corresponding to u, a vector corresponding to grad(u), a trace term corresponding to trace(u) and a trace term corresponding to the flux.
// The test space consists of the following (in order): a scalar test function corresponding to u and a vector test function corresponding to grad(u).

template <int dim>
ConvectionDiffusionDPG<dim>::ConvectionDiffusionDPG ()
                :
				fe_trial_interior (FE_DGQ<dim>(degree), 1, FE_DGQ<dim>(degree), dim), fe_trial_trace (FE_TraceQ<dim>(degree + 1), 1, FE_FaceQ<dim>(degree + 1), 1),
                fe_trial (fe_trial_interior, 1, fe_trial_trace, 1), fe_test (FE_DGQ<dim>(degree + degree_offset), 1, FE_DGQ<dim>(degree + degree_offset), dim),
				dof_handler_trial (triangulation), dof_handler_test (triangulation), dof_handler_trial_interior (triangulation), dof_handler_trial_trace (triangulation)

{}

// Setup the system.

template <int dim>
void ConvectionDiffusionDPG<dim>::setup_system ()
{
dof_handler_trial.distribute_dofs (fe_trial); dof_handler_test.distribute_dofs (fe_test);
dof_handler_trial_interior.distribute_dofs (fe_trial_interior); dof_handler_trial_trace.distribute_dofs (fe_trial_trace);

const unsigned int no_of_cells = triangulation.n_active_cells();
const unsigned int no_of_interior_trial_dofs = dof_handler_trial_interior.n_dofs(); const unsigned int no_of_trace_trial_dofs = dof_handler_trial_trace.n_dofs();
const unsigned int no_of_trial_dofs_per_cell = fe_trial.dofs_per_cell; const unsigned int no_of_test_dofs_per_cell = fe_test.dofs_per_cell;

std::cout << "Trial degrees of freedom: " << dof_handler_trial.n_dofs() << " (" << no_of_interior_trial_dofs << " interior, " << no_of_trace_trial_dofs << " trace)" << std::endl;
std::cout << "Test degrees of freedom: " << dof_handler_test.n_dofs() << std::endl;

create_constraints();

DynamicSparsityPattern trace_dsp (no_of_trace_trial_dofs);
DoFTools::make_sparsity_pattern (dof_handler_trial_trace, trace_dsp);
trace_constraints.condense (trace_dsp);
trace_sparsity_pattern.copy_from (trace_dsp);

trace_system_matrix.reinit (trace_sparsity_pattern);
interior_solution.reinit (no_of_interior_trial_dofs);
trace_solution.reinit (no_of_trace_trial_dofs);
trace_right_hand_side.reinit (no_of_trace_trial_dofs);
refinement_vector.reinit (no_of_cells);

local_optimal_test_functions.resize (no_of_trial_dofs_per_cell*no_of_test_dofs_per_cell, 0); std::fill (local_optimal_test_functions.begin(), local_optimal_test_functions.end(), 0);
bilinear_form_values_storage.resize (no_of_trial_dofs_per_cell*no_of_test_dofs_per_cell*no_of_cells, 0); std::fill (bilinear_form_values_storage.begin(), bilinear_form_values_storage.end(), 0);
V_basis_matrix_inverse_storage.resize ((unsigned int)(0.5*no_of_test_dofs_per_cell*(no_of_test_dofs_per_cell + 1) + 0.1)*no_of_cells, 0); std::fill (V_basis_matrix_inverse_storage.begin(), V_basis_matrix_inverse_storage.end(), 0);
estimator_right_hand_side_storage.resize (no_of_test_dofs_per_cell*no_of_cells, 0); std::fill (estimator_right_hand_side_storage.begin(), estimator_right_hand_side_storage.end(), 0);
intermediate_matrix_storage.resize (no_of_trial_dofs_per_cell*no_of_test_dofs_per_cell*no_of_cells, 0); std::fill (intermediate_matrix_storage.begin(), intermediate_matrix_storage.end(), 0);
}

// Creates boundary and hanging node constraints.

template <int dim>
void ConvectionDiffusionDPG<dim>::create_constraints ()
{
const FEValuesExtractors::Scalar dirichlet_index (0); const ComponentMask dirichlet_comp_mask = fe_trial_trace.component_mask (dirichlet_index);
const FEValuesExtractors::Scalar robin_index (1); const ComponentMask robin_comp_mask = fe_trial_trace.component_mask (robin_index);

trace_constraints.clear ();
DoFTools::make_hanging_node_constraints (dof_handler_trial_trace, trace_constraints); // Hanging node constraints.
VectorTools::interpolate_boundary_values (dof_handler_trial_trace, 0, DirichletBoundaryValues<dim>(), trace_constraints, dirichlet_comp_mask); // Dirichlet boundary constraints.
VectorTools::interpolate_boundary_values (dof_handler_trial_trace, 1, RobinBoundaryValues<dim>(), trace_constraints, robin_comp_mask); // Robin boundary constraints.
trace_constraints.close ();
}

// Assemble the system. First note that we are going to use STATIC CONDENSATION to solve the linear system ( S_00  S_01 // S_10  S_11 ) ( U_INT // U_TR ) = ( F_INT // F_TR ) 
// First we loop over all cells and allocate the values from compute_bilinear_form_values to the correct submatrices. We also compute the right-hand side vectors.
// Expanding the system of equations implies that

// S_00 * U_INT + S_01 * U_TR = F_INT => U_INT = inv(S_00) * F_INT - inv(S_00) * S_01 * U_TR,

// S_10 * U_INT + S_11 * U_TR = F_TR => (S_11 - S_10 * inv(S_00) * S_01) * U_TR = F_TR - S_10 * inv(S_00) * F_INT.

// Strategy: S_00 can be inverted LOCALLY since it contains no coupling (trace) terms and contains only cell terms. Thus, S_11 - S_10 * inv(S_00) * S_01 and F_TR - S_10 * inv(S_00) * F_INT may all be computed locally as well and then assembed into the global (trace) system matrix.
// Once U_TR is known from a global solve, U_INT is then recovered from U_INT = inv(S_00) * F_INT - inv(S_00) * S_01 * U_TR. We can preallocate the portion inv(S_00) * F_INT locally as well but we must store the local matrix inv(S_00) * S_01 for each cell until U_TR is known.

// If during assembly the current element K is the same size as the previous element K' AND the convection a|_K is the same as the convection a|_K' then we do not need to recalculate any of the bilinear form values, optimal test functions or local matrices.
// We therefore check if the convection evaluated at the quadrature points on the current element is the same as on the previous element to within some degree of tolerance and we also check that diam(K) = diam(K'). If both true, we do not recalculate.

template <int dim>
void ConvectionDiffusionDPG<dim>::assemble_system ()
{
const QGauss<dim>  quadrature_formula_cell (degree+degree_offset+1);
const QGauss<dim-1>  quadrature_formula_face (degree+degree_offset);

const unsigned int no_of_quad_points_cell = quadrature_formula_cell.size(); const unsigned int no_of_quad_points_face = quadrature_formula_face.size();
const unsigned int no_of_trial_dofs_per_cell = fe_trial.dofs_per_cell; const unsigned int no_of_test_dofs_per_cell = fe_test.dofs_per_cell;
const unsigned int no_of_interior_trial_dofs_per_cell = fe_trial_interior.dofs_per_cell; const unsigned int no_of_trace_trial_dofs_per_cell = fe_trial_trace.dofs_per_cell;

typename DoFHandler<dim>::active_cell_iterator trial_cell = dof_handler_trial.begin_active(), final_cell = dof_handler_trial.end(); typename DoFHandler<dim>::active_cell_iterator test_cell = dof_handler_test.begin_active();
typename DoFHandler<dim>::active_cell_iterator trial_cell_interior = dof_handler_trial_interior.begin_active(); typename DoFHandler<dim>::active_cell_iterator trial_cell_trace = dof_handler_trial_trace.begin_active();

FEValues<dim> fe_values_trial_cell (fe_trial_interior, quadrature_formula_cell, update_values | update_quadrature_points);
FEValues<dim> fe_values_test_cell (fe_test, quadrature_formula_cell, update_values | update_gradients | update_JxW_values);
FEFaceValues<dim> fe_values_trial_face (fe_trial_trace, quadrature_formula_face, update_values);
FEFaceValues<dim> fe_values_test_face (fe_test, quadrature_formula_face, update_values | update_JxW_values | update_normal_vectors);
   
FullMatrix<double> local_trace_system_matrix (no_of_trace_trial_dofs_per_cell, no_of_trace_trial_dofs_per_cell);
FullMatrix<double> local_system_submatrix_00 (no_of_interior_trial_dofs_per_cell, no_of_interior_trial_dofs_per_cell);
FullMatrix<double> local_system_submatrix_01 (no_of_interior_trial_dofs_per_cell, no_of_trace_trial_dofs_per_cell);
FullMatrix<double> local_system_submatrix_10 (no_of_trace_trial_dofs_per_cell, no_of_interior_trial_dofs_per_cell);
FullMatrix<double> local_system_submatrix_11 (no_of_trace_trial_dofs_per_cell, no_of_trace_trial_dofs_per_cell);
FullMatrix<double> intermediate_matrix (no_of_trace_trial_dofs_per_cell, no_of_interior_trial_dofs_per_cell);
FullMatrix<double> intermediate_matrix1 (no_of_interior_trial_dofs_per_cell, no_of_trace_trial_dofs_per_cell);
Vector<double> local_interior_solution (no_of_interior_trial_dofs_per_cell);
Vector<double> local_interior_right_hand_side (no_of_interior_trial_dofs_per_cell);
Vector<double> local_trace_right_hand_side (no_of_trace_trial_dofs_per_cell);
Vector<double> intermediate_vector (no_of_trace_trial_dofs_per_cell);
std::vector<Tensor<1,dim,double> > convection_values (no_of_quad_points_cell);
std::vector<Tensor<1,dim,double> > previous_convection_values (no_of_quad_points_cell);
std::vector<double> forcing_values (no_of_quad_points_cell);
std::vector<types::global_dof_index> local_dof_indices_trial_cell (no_of_interior_trial_dofs_per_cell);
std::vector<types::global_dof_index> local_dof_indices_trial_trace (no_of_trace_trial_dofs_per_cell);

std::vector<unsigned int> additional_data (7); additional_data[0] = no_of_quad_points_cell; additional_data[1] = no_of_quad_points_face; additional_data[2] = no_of_trial_dofs_per_cell;
additional_data[3] = no_of_test_dofs_per_cell; additional_data[4] = no_of_interior_trial_dofs_per_cell; additional_data[5] = no_of_trace_trial_dofs_per_cell; 

unsigned int cell_no = 0; unsigned int previous_cell_no = 0; double cell_size = 0; double previous_cell_size = 0; 
double cell_size_check = 0; double convection_check = 0;
unsigned int index_no_1 = 0; unsigned int index_no_2 = 0; unsigned int index_no_3 = 0;

    for (; trial_cell != final_cell; ++trial_cell, ++trial_cell_interior, ++trial_cell_trace, ++test_cell)
    {
	fe_values_trial_cell.reinit (trial_cell_interior); fe_values_test_cell.reinit (test_cell);

	trial_cell_interior->get_dof_indices (local_dof_indices_trial_cell); trial_cell_trace->get_dof_indices (local_dof_indices_trial_trace);

	Convection<dim>().value_list (fe_values_trial_cell.get_quadrature_points(), convection_values); 
	Forcing<dim>().value_list (fe_values_trial_cell.get_quadrature_points(), forcing_values);

	cell_no = trial_cell->active_cell_index(); cell_size = trial_cell->diameter();
	index_no_1 = no_of_trial_dofs_per_cell*no_of_test_dofs_per_cell*cell_no;
	index_no_2 = no_of_test_dofs_per_cell*cell_no;
	index_no_3 = no_of_interior_trial_dofs_per_cell*no_of_trace_trial_dofs_per_cell*cell_no;

	cell_size_check = fabs(cell_size - previous_cell_size);

	if (cell_size_check < 1e-15)
	{
	convection_check = 0;

	    for (unsigned int quad_point = 0; quad_point < no_of_quad_points_cell; ++quad_point)
		    for (unsigned int d = 0; d < dim; ++d)
		    {
		    convection_check += (convection_values[quad_point][d] - previous_convection_values[quad_point][d])*(convection_values[quad_point][d] - previous_convection_values[quad_point][d]);
		    }

	convection_check = sqrt(convection_check);
    }

	if (cell_size_check < 1e-15 && convection_check < 1e-15)
	{
    const unsigned int index_no_4 = no_of_trial_dofs_per_cell*no_of_test_dofs_per_cell*previous_cell_no;
	const unsigned int index_no_5= (unsigned int)(0.5*no_of_test_dofs_per_cell*(no_of_test_dofs_per_cell + 1) + 0.1)*cell_no;
    const unsigned int index_no_6 = (unsigned int)(0.5*no_of_test_dofs_per_cell*(no_of_test_dofs_per_cell + 1) + 0.1)*previous_cell_no;

	    for (unsigned int k = 0; k < no_of_test_dofs_per_cell; ++k)
		{
		    for (unsigned int i = 0; i < no_of_trial_dofs_per_cell; ++i)
			{
			bilinear_form_values_storage[k + i*no_of_test_dofs_per_cell + index_no_1] = bilinear_form_values_storage[k + i*no_of_test_dofs_per_cell + index_no_4];
			}

			for (unsigned int l = 0; l < k + 1; ++l)
			{
            V_basis_matrix_inverse_storage[(unsigned int)(0.5*k*(k+1) + 0.1) + l + index_no_5] = V_basis_matrix_inverse_storage[(unsigned int)(0.5*k*(k+1) + 0.1) + l + index_no_6];
		    }
		}
	}
	else
	{
    additional_data[6] = cell_no;

	compute_bilinear_form_values (trial_cell_trace, test_cell, fe_values_trial_cell, fe_values_test_cell, fe_values_trial_face, fe_values_test_face, convection_values, additional_data);
	compute_local_optimal_test_functions (fe_values_test_cell, convection_values, additional_data, cell_size, local_optimal_test_functions);

	local_system_submatrix_00 = 0; local_system_submatrix_01 = 0; local_system_submatrix_10 = 0; local_system_submatrix_11 = 0;

	    for (unsigned int i = 0; i < no_of_trial_dofs_per_cell; ++i)
		{
		const unsigned int comp_i = fe_trial.system_to_base_index(i).first.first;
		const unsigned int basis_i = fe_trial.system_to_base_index(i).second;

	        for (unsigned int j = 0; j < no_of_trial_dofs_per_cell; ++j)
			{
			const unsigned int comp_j = fe_trial.system_to_base_index(j).first.first;
		    const unsigned int basis_j = fe_trial.system_to_base_index(j).second;

			if (comp_i == 0 && comp_j == 0)
			{
			    for (unsigned int k = 0; k < no_of_test_dofs_per_cell; ++k)
				{
				local_system_submatrix_00(basis_i,basis_j) += local_optimal_test_functions[k + i*no_of_test_dofs_per_cell]*bilinear_form_values_storage[k + j*no_of_test_dofs_per_cell + index_no_1];
				}
			}

			if (comp_i == 0 && comp_j == 1)
			{
			    for (unsigned int k = 0; k < no_of_test_dofs_per_cell; ++k)
				{
				local_system_submatrix_01(basis_i,basis_j) += local_optimal_test_functions[k + i*no_of_test_dofs_per_cell]*bilinear_form_values_storage[k + j*no_of_test_dofs_per_cell + index_no_1];
				}
			}

			if (comp_i == 1 && comp_j == 0)
			{
			    for (unsigned int k = 0; k < no_of_test_dofs_per_cell; ++k)
				{
				local_system_submatrix_10(basis_i,basis_j) += local_optimal_test_functions[k + i*no_of_test_dofs_per_cell]*bilinear_form_values_storage[k + j*no_of_test_dofs_per_cell + index_no_1];
				}
			}

			if (comp_i == 1 && comp_j == 1)
			{
			    for (unsigned int k = 0; k < no_of_test_dofs_per_cell; ++k)
				{
				local_system_submatrix_11(basis_i,basis_j) += local_optimal_test_functions[k + i*no_of_test_dofs_per_cell]*bilinear_form_values_storage[k + j*no_of_test_dofs_per_cell + index_no_1];
				}
			}
			}
		}

 	local_system_submatrix_00.gauss_jordan();
	local_system_submatrix_10.mmult(intermediate_matrix, local_system_submatrix_00);
	intermediate_matrix *= -1;

	intermediate_matrix.mmult(local_trace_system_matrix, local_system_submatrix_01);
	local_trace_system_matrix.add(1, local_system_submatrix_11);
	local_system_submatrix_00.mmult(intermediate_matrix1, local_system_submatrix_01); 
	}

		for (unsigned int k = 0; k < no_of_test_dofs_per_cell; ++k)
		{
		unsigned int comp_k = fe_test.system_to_base_index(k).first.first;

		if (comp_k == 0)
		{
		    for (unsigned int quad_point = 0; quad_point < no_of_quad_points_cell; ++quad_point)
			{
			double test_cell_value = forcing_values[quad_point]*fe_values_test_cell.shape_value_component(k,quad_point,0)*fe_values_test_cell.JxW(quad_point);

			estimator_right_hand_side_storage[k + index_no_2] += test_cell_value;

				for (unsigned int i = 0; i < no_of_trial_dofs_per_cell; ++i)
	            {
				unsigned int comp_i = fe_trial.system_to_base_index(i).first.first;
		        unsigned int basis_i = fe_trial.system_to_base_index(i).second;

				if (comp_i == 0) {local_interior_right_hand_side(basis_i) += local_optimal_test_functions[k + i*no_of_test_dofs_per_cell]*test_cell_value;} else {local_trace_right_hand_side(basis_i) += local_optimal_test_functions[k + i*no_of_test_dofs_per_cell]*test_cell_value;}
			    }
			}
		}
		}

	intermediate_matrix.vmult(intermediate_vector, local_interior_right_hand_side);
	local_trace_right_hand_side.add(1, intermediate_vector);

	    // Place the local system matrix contributions in the system matrix and the local right hand side contributions in the right hand side vector.
        for (unsigned int i = 0; i < no_of_trace_trial_dofs_per_cell; ++i)
	    {
		    for (unsigned int j = 0; j < no_of_trace_trial_dofs_per_cell; ++j)
		    {
		    trace_system_matrix(local_dof_indices_trial_trace[i], local_dof_indices_trial_trace[j]) += local_trace_system_matrix(i,j);
		    }

     	trace_right_hand_side(local_dof_indices_trial_trace[i]) += local_trace_right_hand_side(i);
	    }

	local_system_submatrix_00.vmult (local_interior_solution, local_interior_right_hand_side);

        for (unsigned int i = 0; i < no_of_interior_trial_dofs_per_cell; ++i)
	    {
     	interior_solution(local_dof_indices_trial_cell[i]) += local_interior_solution(i);
	    }

		for (unsigned int i = 0; i < no_of_interior_trial_dofs_per_cell; ++i)
		    for (unsigned int j = 0; j < no_of_trace_trial_dofs_per_cell; ++j)
			{
			intermediate_matrix_storage[i + j*no_of_interior_trial_dofs_per_cell + index_no_3] = intermediate_matrix1(i,j);
			}
    
	    for (unsigned int quad_point = 0; quad_point < no_of_quad_points_cell; ++quad_point)
		{
		previous_convection_values[quad_point] = convection_values [quad_point];
		}

	local_interior_right_hand_side = 0; local_trace_right_hand_side = 0; previous_cell_no = cell_no; previous_cell_size = cell_size; 
    }
}

// Intermediate function needed in assemble_system to assemble the system matrix. 
// Takes current cell information as an input and outputs a vector whose k + i*no_of_test_dofs_per_cell entry corresponds to B(phi_i, psi_k) where {phi_i} is a local basis for U_h and {psi_k} is a local basis for the the full enriched space V_h.
// B_K({u,sigma,trace(u),trace(sigma)},{v,tau}) := (u, div(tau) - b*grad(v))_K + (sigma, tau + epsilon*grad(v))_K - (trace(u), tau*n)_dK + (trace(sigma), v)_dK.

template <int dim>
void ConvectionDiffusionDPG<dim>::compute_bilinear_form_values (const typename DoFHandler<dim>::active_cell_iterator &trial_cell, const typename DoFHandler<dim>::active_cell_iterator &test_cell, const FEValues<dim> &fe_values_trial_cell, const FEValues<dim> &fe_values_test_cell, FEFaceValues<dim> &fe_values_trial_face, FEFaceValues<dim> &fe_values_test_face, const std::vector<Tensor<1,dim> > &convection_values, const std::vector<unsigned int> &additional_data)
{
const unsigned int no_of_quad_points_cell = additional_data[0];
const unsigned int no_of_quad_points_face = additional_data[1];
const unsigned int no_of_trial_dofs_per_cell = additional_data[2];
const unsigned int no_of_test_dofs_per_cell = additional_data[3];
const unsigned int no_of_interior_trial_dofs_per_cell = additional_data[4];
const unsigned int no_of_trace_trial_dofs_per_cell = additional_data[5];
const unsigned int index_no = no_of_trial_dofs_per_cell*no_of_test_dofs_per_cell*additional_data[6];

std::vector<double> cell_values (no_of_interior_trial_dofs_per_cell*no_of_test_dofs_per_cell);
std::vector<double> face_values (no_of_trace_trial_dofs_per_cell*no_of_test_dofs_per_cell);

    for (unsigned int quad_point = 0; quad_point < no_of_quad_points_cell; ++quad_point)
	    for (unsigned int k = 0; k < no_of_test_dofs_per_cell; ++k)
		{
		std::vector<double> values(dim+1, 0);

	        for (unsigned int d = 0; d < dim; ++d)
		    {
			values[0] += (fe_values_test_cell.shape_grad_component(k,quad_point,d+1)[d] - convection_values[quad_point][d]*fe_values_test_cell.shape_grad_component(k,quad_point,0)[d])*fe_values_test_cell.JxW(quad_point);
			values[d+1] = (fe_values_test_cell.shape_value_component(k,quad_point,d+1) + epsilon*fe_values_test_cell.shape_grad_component(k,quad_point,0)[d])*fe_values_test_cell.JxW(quad_point);
		    }

		    for (unsigned int i = 0; i < no_of_interior_trial_dofs_per_cell; ++i)
			    for (unsigned int d = 0; d < dim + 1; ++d)
		        {
				cell_values[k + i*no_of_test_dofs_per_cell] += fe_values_trial_cell.shape_value_component(i,quad_point,d)*values[d];
				}
        }
 

	for (unsigned int face = 0; face < 2*dim; ++face)
    {
	fe_values_trial_face.reinit (trial_cell, face); fe_values_test_face.reinit (test_cell, face);
	
	const std::vector<Tensor<1,dim> > &normals = fe_values_test_face.get_normal_vectors();

    std::vector<unsigned int> trial_index, test_index;

	    for (unsigned int i = 0; i < no_of_trace_trial_dofs_per_cell; ++i)
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
            double taudotnormal = 0;  double test_face_value = std::pow(-1,face+1)*fe_values_test_face.shape_value_component(test_index[k],quad_point,0)*fe_values_test_face.JxW(quad_point); 

                for (unsigned int d = 0; d < dim; ++d)
                {
				taudotnormal += fe_values_test_face.shape_value_component(test_index[k],quad_point,d+1)*normals[quad_point][d]*fe_values_test_face.JxW(quad_point);
                }

		        for (unsigned int i = 0; i < no_of_trial_dofs_on_face; ++i)
                {
				face_values[test_index[k] + trial_index[i]*no_of_test_dofs_per_cell] += fe_values_trial_face.shape_value_component(trial_index[i],quad_point,1)*test_face_value - fe_values_trial_face.shape_value_component(trial_index[i],quad_point,0)*taudotnormal;
                }
            }
    }
    
    for (unsigned int i = 0; i < no_of_trial_dofs_per_cell; ++i)
    {
    unsigned int comp_i = fe_trial.system_to_base_index(i).first.first;
    unsigned int basis_i = fe_trial.system_to_base_index(i).second;

    if (comp_i == 0)
    {
        for (unsigned int k = 0; k < no_of_test_dofs_per_cell; ++k)
        {
		bilinear_form_values_storage[k + i*no_of_test_dofs_per_cell + index_no] = cell_values[k + basis_i*no_of_test_dofs_per_cell];
        }
    }
    else
    {
        for (unsigned int k = 0; k < no_of_test_dofs_per_cell; ++k)
        {
		bilinear_form_values_storage[k + i*no_of_test_dofs_per_cell + index_no] = face_values[k + basis_i*no_of_test_dofs_per_cell];
        }
    }
    }
}

// Intermediate function needed in assemble_system to compute the optimal test functions. 
// Takes current cell information and bilinear form values as an input and outputs a vector whose k + i*no_of_test_dofs_per_cell entry corresponds to the weighting of the {psi_k} local basis function in the local basis expansion of the ith test function in the full enriched space V_h.

template <int dim>
void ConvectionDiffusionDPG<dim>::compute_local_optimal_test_functions (const FEValues<dim> &fe_values_test_cell, const std::vector<Tensor<1,dim> > &convection_values, const std::vector<unsigned int> &additional_data, const double &cell_size, std::vector<double> &local_optimal_test_functions)
{
const unsigned int no_of_quad_points_cell = additional_data[0];
const unsigned int no_of_trial_dofs_per_cell = additional_data[2];
const unsigned int no_of_test_dofs_per_cell = additional_data[3];
const unsigned int cell_no = additional_data[6];
const unsigned int index_no_1 = (unsigned int)(0.5*no_of_test_dofs_per_cell*(no_of_test_dofs_per_cell + 1) + 0.1)*cell_no;
const unsigned int index_no_2 = no_of_trial_dofs_per_cell*no_of_test_dofs_per_cell*cell_no;

FullMatrix<double> V_basis_matrix_inverse (no_of_test_dofs_per_cell, no_of_test_dofs_per_cell);
Vector<double> U_basis_rhs (no_of_test_dofs_per_cell);
Vector<double> optimal_test_function (no_of_test_dofs_per_cell);

// Compute the V basis matrix whose (i,j) entry corresponds to the local inner product associated with the norm ||div(tau)-b*grad(v)||^2 + ||C_K*(tau/epsilon + grad(v))||^2 + ||v||^2 + ||grad(v)||^2 with C_K = min(sqrt(epsilon/|K|), 1).
// Note: The proposed test norm is NOT good for the hyperbolic problem (epsilon = 0). This is a possible avenue for future research.

double C_K = fmin(epsilon/cell_size, 1);

    for (unsigned int k = 0; k < no_of_test_dofs_per_cell; ++k)
        for (unsigned int l = 0; l < k + 1; ++l)
            for (unsigned int quad_point = 0; quad_point < no_of_quad_points_cell; ++quad_point)    
            {
			double div_tau_k = 0; double div_tau_l = 0; double conv_grad_k = 0; double conv_grad_l = 0;

			    for (unsigned int d = 0; d < dim; ++d)
				{
				div_tau_k += fe_values_test_cell.shape_grad_component(k,quad_point,d+1)[d]; div_tau_l += fe_values_test_cell.shape_grad_component(l,quad_point,d+1)[d];
				conv_grad_k += convection_values[quad_point][d]*fe_values_test_cell.shape_grad_component(k,quad_point,0)[d]; conv_grad_l += convection_values[quad_point][d]*fe_values_test_cell.shape_grad_component(l,quad_point,0)[d];

				V_basis_matrix_inverse(k,l) += C_K*((1/epsilon)*fe_values_test_cell.shape_value_component(k,quad_point,d+1) + fe_values_test_cell.shape_grad_component(k,quad_point,0)[d])*((1/epsilon)*fe_values_test_cell.shape_value_component(l,quad_point,d+1) + fe_values_test_cell.shape_grad_component(l,quad_point,0)[d])*fe_values_test_cell.JxW(quad_point);
				}

			V_basis_matrix_inverse(k,l) += (fe_values_test_cell.shape_value_component(k,quad_point,0)*fe_values_test_cell.shape_value_component(l,quad_point,0) 
			                               + fe_values_test_cell.shape_grad_component(k,quad_point,0)*fe_values_test_cell.shape_grad_component(l,quad_point,0)
			                               + (div_tau_k - conv_grad_k)*(div_tau_l - conv_grad_l))*fe_values_test_cell.JxW(quad_point);
            }
           
    for (unsigned int k = 0; k < no_of_test_dofs_per_cell; ++k)
        for (unsigned int l = 0; l < k + 1; ++l)
        {
        V_basis_matrix_inverse(l,k) = V_basis_matrix_inverse(k,l);
        }

V_basis_matrix_inverse.gauss_jordan(); // Invert the V basis matrix in preparation for finding the optimal test function basis coefficients.
    
	for (unsigned int k = 0; k < no_of_test_dofs_per_cell; ++k)
	    for (unsigned int l = 0; l < k + 1; ++l)
		{
        V_basis_matrix_inverse_storage[(unsigned int)(0.5*k*(k+1) + 0.1) + l + index_no_1] = V_basis_matrix_inverse(k,l);
		}

    for (unsigned int i = 0; i < no_of_trial_dofs_per_cell; ++i)
    {  	
	// Set the right hand side whose kth entry is B(phi_i, psi_k) where {phi_i} is a local basis for U_h and {psi_k} is a local basis for the the full enriched space V_h.

        for (unsigned int k = 0; k < no_of_test_dofs_per_cell; ++k)
        {
		U_basis_rhs(k) = bilinear_form_values_storage[k + i*no_of_test_dofs_per_cell + index_no_2];
        }

    // Solve the linear system corresponding to (varphi_{i,k}, psi_k)_V = B(phi_i, psi_k) for varphi_{i,k} where varphi_{i,k} is a vector of unknowns representing the coefficients of the ith local optimal test function in the local basis {psi_k} of the the full enriched space V_h and phi_i is a (fixed) local basis function for U_h.
    V_basis_matrix_inverse.vmult (optimal_test_function, U_basis_rhs); 

	    for (unsigned int k = 0; k < no_of_test_dofs_per_cell; ++k)
        {
        local_optimal_test_functions[k + i*no_of_test_dofs_per_cell] = optimal_test_function(k); 
        }
    }
}

// Solve the system. 
// Recall from earlier that once we have solved for U_TR, we need to back substitute to solve for U_INT.

template <int dim>
void ConvectionDiffusionDPG<dim>::solve ()
{
trace_constraints.condense (trace_system_matrix, trace_right_hand_side);

SparseDirectUMFPACK trace_system_matrix_inverse;
trace_system_matrix_inverse.initialize (trace_system_matrix);
trace_system_matrix_inverse.vmult (trace_solution, trace_right_hand_side);

trace_constraints.distribute (trace_solution);

const unsigned int no_of_interior_trial_dofs_per_cell = fe_trial_interior.dofs_per_cell;
const unsigned int no_of_trace_trial_dofs_per_cell = fe_trial_trace.dofs_per_cell;

typename DoFHandler<dim>::active_cell_iterator trial_cell = dof_handler_trial_interior.begin_active(), final_cell = dof_handler_trial_interior.end();
typename DoFHandler<dim>::active_cell_iterator trial_cell_trace = dof_handler_trial_trace.begin_active();

std::vector<types::global_dof_index> local_dof_indices_trial_cell (no_of_interior_trial_dofs_per_cell);
std::vector<types::global_dof_index> local_dof_indices_trial_trace (no_of_trace_trial_dofs_per_cell);

std::vector<double> local_interior_solution (no_of_interior_trial_dofs_per_cell);

    for (; trial_cell != final_cell; ++trial_cell, ++trial_cell_trace)
    {
	trial_cell->get_dof_indices (local_dof_indices_trial_cell);
	trial_cell_trace->get_dof_indices (local_dof_indices_trial_trace);

	const unsigned int cell_no = trial_cell->active_cell_index();
	const unsigned int index_no = no_of_interior_trial_dofs_per_cell*no_of_trace_trial_dofs_per_cell*cell_no;

	std::fill (local_interior_solution.begin(), local_interior_solution.end(), 0);

        for (unsigned int i = 0; i < no_of_interior_trial_dofs_per_cell; ++i)
		    for (unsigned int j = 0; j < no_of_trace_trial_dofs_per_cell; ++j)
		    {
			local_interior_solution[i] += intermediate_matrix_storage[i + j*no_of_interior_trial_dofs_per_cell + index_no]*trace_solution(local_dof_indices_trial_trace[j]);
		    }

        for (unsigned int i = 0; i < no_of_interior_trial_dofs_per_cell; ++i)
		{
		interior_solution(local_dof_indices_trial_cell[i]) -= local_interior_solution[i];
		}
	}
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
data_out.attach_dof_handler (dof_handler_trial_interior);
data_out.add_data_vector (interior_solution, solution_names, DataOut<dim>::type_dof_data, data_component_interpretation);
data_out.build_patches ();

const std::string filename = "solution-" + Utilities::int_to_string (cycle, 2) + ".gnuplot";

std::ofstream gnuplot_output (filename.c_str());
data_out.write_gnuplot (gnuplot_output);
}

// Computes the error estimator needed for mesh refinement.
// We solve the system (res, v_h)_V = F(v_h) - B(u_h, v_h) for res in the enriched test space V_h. Then ||res||_V provides an a posteriori error bound. See DPG literature.
// NOTE: This system of equations can again be solved locally.

template <int dim>
void ConvectionDiffusionDPG<dim>::compute_error_estimator ()
{
const QGauss<dim>  quadrature_formula_cell (degree+degree_offset+1);

const unsigned int no_of_quad_points_cell = quadrature_formula_cell.size();
const unsigned int no_of_trial_dofs_per_cell = fe_trial.dofs_per_cell; const unsigned int no_of_test_dofs_per_cell = fe_test.dofs_per_cell;
const unsigned int no_of_interior_trial_dofs_per_cell = fe_trial_interior.dofs_per_cell; const unsigned int no_of_trace_trial_dofs_per_cell = fe_trial_trace.dofs_per_cell;

typename DoFHandler<dim>::active_cell_iterator test_cell = dof_handler_test.begin_active(), final_cell = dof_handler_test.end();
typename DoFHandler<dim>::active_cell_iterator trial_cell_interior = dof_handler_trial_interior.begin_active(); typename DoFHandler<dim>::active_cell_iterator trial_cell_trace = dof_handler_trial_trace.begin_active();

FEValues<dim> fe_values_test_cell (fe_test, quadrature_formula_cell, update_values | update_gradients | update_quadrature_points | update_JxW_values);

FullMatrix<double> V_basis_matrix_inverse (no_of_test_dofs_per_cell, no_of_test_dofs_per_cell);
Vector<double> local_residual_coefficients (no_of_test_dofs_per_cell);
Vector<double> local_right_hand_side (no_of_test_dofs_per_cell);
std::vector<Tensor<1,dim,double> > convection_values (no_of_quad_points_cell);
std::vector<double> v_values (no_of_quad_points_cell);
std::vector<double> div_tau_values (no_of_quad_points_cell);
std::vector<Tensor<1,dim,double> > tau_values (no_of_quad_points_cell);
std::vector<Tensor<1,dim,double> > grad_v_values (no_of_quad_points_cell);
std::vector<types::global_dof_index> local_dof_indices_trial_cell (no_of_interior_trial_dofs_per_cell);
std::vector<types::global_dof_index> local_dof_indices_trial_trace (no_of_trace_trial_dofs_per_cell);

unsigned int cell_no = 0; unsigned int index_no_1 = 0; unsigned int index_no_2 = 0; unsigned int index_no_3 = 0;

    for (; test_cell != final_cell; ++test_cell, ++trial_cell_interior, ++trial_cell_trace)
    {
	fe_values_test_cell.reinit (test_cell);

    trial_cell_interior->get_dof_indices (local_dof_indices_trial_cell); trial_cell_trace->get_dof_indices (local_dof_indices_trial_trace);

	cell_no = trial_cell_interior->active_cell_index();
	index_no_1 = (unsigned int)(0.5*no_of_test_dofs_per_cell*(no_of_test_dofs_per_cell + 1) + 0.1)*cell_no;
	index_no_2 = no_of_test_dofs_per_cell*cell_no;
	index_no_3 = cell_no*no_of_trial_dofs_per_cell*no_of_test_dofs_per_cell;

	Convection<dim>().value_list (fe_values_test_cell.get_quadrature_points(), convection_values);

	double C_K = fmin(epsilon/trial_cell_interior->diameter(), 1);

	    for (unsigned int k = 0; k < no_of_test_dofs_per_cell; ++k)
	    {
	        for (unsigned int l = 0; l < k + 1; ++l)
		    {
            V_basis_matrix_inverse(k,l) = V_basis_matrix_inverse_storage[(unsigned int)(0.5*k*(k+1) + 0.1) + l + index_no_1];
		    V_basis_matrix_inverse(l,k) = V_basis_matrix_inverse(k,l);
	     	}
        
		local_right_hand_side(k) = estimator_right_hand_side_storage[k + index_no_2];

	        for (unsigned int i = 0; i < no_of_trial_dofs_per_cell; ++i)
	    	{
			unsigned int comp_i = fe_trial.system_to_base_index(i).first.first;
            unsigned int basis_i = fe_trial.system_to_base_index(i).second;

			if (comp_i == 0)
			{
			local_right_hand_side(k) -= interior_solution(local_dof_indices_trial_cell[basis_i])*bilinear_form_values_storage[k + i*no_of_test_dofs_per_cell + index_no_3];
			}
			else
			{
			local_right_hand_side(k) -= trace_solution(local_dof_indices_trial_trace[basis_i])*bilinear_form_values_storage[k + i*no_of_test_dofs_per_cell + index_no_3];
			}
		    }
        }

	V_basis_matrix_inverse.vmult (local_residual_coefficients, local_right_hand_side);

	std::fill (v_values.begin(), v_values.end(), 0); std::fill (div_tau_values.begin(), div_tau_values.end(), 0); std::fill (tau_values.begin(), tau_values.end(), 0); std::fill (grad_v_values.begin(), grad_v_values.end(), 0);

	    for (unsigned int quad_point = 0; quad_point < no_of_quad_points_cell; ++quad_point)
			for (unsigned int k = 0; k < no_of_test_dofs_per_cell; ++k)
			{
			const unsigned int comp_k = fe_test.system_to_base_index(k).first.first;

			if (comp_k == 0)
			{
			    for (unsigned int d = 0; d < dim; ++d)
				{
				grad_v_values[quad_point][d] += local_residual_coefficients(k)*fe_values_test_cell.shape_grad_component(k,quad_point,0)[d];
				}
            
			v_values[quad_point] += local_residual_coefficients(k)*fe_values_test_cell.shape_value_component(k,quad_point,0);
			}
			else
			{
			double div_value = 0;

			    for (unsigned int d = 0; d < dim; ++d)
				{
				tau_values[quad_point][d] += local_residual_coefficients(k)*fe_values_test_cell.shape_value_component(k,quad_point,d+1);
				div_value += fe_values_test_cell.shape_grad_component(k,quad_point,d+1)[d];
				}

            div_tau_values[quad_point] += local_residual_coefficients(k)*div_value;  
			}
			}

		for (unsigned int quad_point = 0; quad_point < no_of_quad_points_cell; ++quad_point)
		{
		refinement_vector(cell_no) += (C_K*grad_v_values[quad_point]*grad_v_values[quad_point] + C_K*(1/epsilon)*(1/epsilon)*tau_values[quad_point]*tau_values[quad_point] + 2*C_K*(1/epsilon)*tau_values[quad_point]*grad_v_values[quad_point]
		                              + (div_tau_values[quad_point] - convection_values[quad_point]*grad_v_values[quad_point])*(div_tau_values[quad_point] - convection_values[quad_point]*grad_v_values[quad_point]) 
									  + v_values[quad_point]*v_values[quad_point] + grad_v_values[quad_point]*grad_v_values[quad_point])*fe_values_test_cell.JxW(quad_point);
		}

    refinement_vector(cell_no) = sqrt(refinement_vector(cell_no));
	}

std::cout << std::endl;
std::cout << "Error Estimator: " << refinement_vector.l2_norm() << std::endl;
std::cout << std::endl;
}

template <int dim>
void ConvectionDiffusionDPG<dim>::refine_grid ()
{
GridRefinement::refine_and_coarsen_fixed_number (triangulation, refinement_vector, 0.1, 0.05);
triangulation.execute_coarsening_and_refinement ();
}

template <int dim>
void ConvectionDiffusionDPG<dim>::run ()
{
GridGenerator::hyper_cube (triangulation, -1, 1); triangulation.refine_global (2); // Creates the triangulation and globally refines n times.
    
	for (; cycle < no_of_cycles + 1; ++cycle)
	{
	std::cout << "~~Solution Cycle " << cycle << "~~" << std::endl; std::cout << std::endl;

	setup_system ();
    assemble_system ();
	solve ();
	output_solution ();
	compute_error_estimator ();
	refine_grid ();
	}	
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
std::cerr << std::endl << std::endl << "----------------------------------------------------" << std::endl;
std::cerr << "Exception on processing: " << std::endl << exc.what() << std::endl << "Aborting!" << std::endl << "----------------------------------------------------" << std::endl;
return 1;
}
catch (...)
{
std::cerr << std::endl << std::endl << "----------------------------------------------------" << std::endl;
std::cerr << "Unknown exception!" << std::endl << "Aborting!" << std::endl << "----------------------------------------------------" << std::endl;
return 1;
};

return 0;
}
