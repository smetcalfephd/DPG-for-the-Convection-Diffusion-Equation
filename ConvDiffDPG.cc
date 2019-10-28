 #include <deal.II/base/function.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dg_vector.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>

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
	void solve (); // Solve the system. 
	void output_solution () const; // Output the solution.
    void compute_local_bilinear_form_values (const typename DoFHandler<dim>::active_cell_iterator &trial_cell, const typename DoFHandler<dim>::active_cell_iterator &test_cell, const FEValues<dim> &fe_values_trial_cell, const FEValues<dim> &fe_values_test_cell, FEFaceValues<dim> &fe_values_trial_face, FEFaceValues<dim> &fe_values_test_face, Vector<double> &local_bilinear_form_values); // Intermediate function needed in assemble_system to assemble the system matrix.
    void compute_local_optimal_test_functions (const typename DoFHandler<dim>::active_cell_iterator &trial_cell, const FEValues<dim> &fe_values_trial_cell, const FEValues<dim> &fe_values_test_cell, const Vector<double> &local_bilinear_form_values, Vector<double> &local_optimal_test_functions); // Intermediate function needed in assemble_system to compute the optimal test functions. 

    Triangulation<dim> triangulation;
    FESystem<dim> fe_trial, fe_test;
	DoFHandler<dim> dof_handler_trial, dof_handler_test;

	ConstraintMatrix constraints;
	SparsityPattern sparsity_pattern;

	SparseMatrix<double> system_matrix;
	Vector<double> solution, right_hand_side, local_optimal_test_functions, local_bilinear_form_values;
};

// The trial space consists of the following (in order): a scalar corresponding to u, a vector corresponding to grad(u), a face term corresponding to trace(u) and a face term corresponding to the flux.
// The test space consists of the following (in order): a scalar test function corresponding to u and a vector test function corresponding to grad(u).
// The method also seems to work if you replace {FE_RaviartThomas<dim>(degree - 1), 1} by {FE_Q<dim>(degree), dim} as long as you also replace {FE_DGRaviartThomas<dim>(degree - 1 + degree_offset), 1} by {FE_DGQ<dim>(degree), dim}.
// IMPORTANT: For dim = 1, FE_RaviartThomas<dim>(degree - 1) in the trial space and FE_DGRaviartThomas<dim>(degree + degree_offset - 1) need to be replaced by FE_Q<dim>(degree - 1) and FE_DGQ<dim>(degree + degree_offset - 1) respectively as RT spaces are undefined for dim = 1.

template <int dim>
ConvectionDiffusionDPG<dim>::ConvectionDiffusionDPG ()
                :
				fe_trial (FESystem<dim>(FE_Q<dim>(degree), 1, FE_RaviartThomas<dim>(degree - 1), 1), FESystem<dim>(FE_FaceQ<dim>(degree + 1), 1, FE_FaceQ<dim>(degree), 1)),
				fe_test (FE_DGQ<dim>(degree + degree_offset), 1, FE_DGRaviartThomas<dim>(degree - 1 + degree_offset), 1),
				dof_handler_trial (triangulation),
			    dof_handler_test (triangulation)

{}

// Setup the system

template <int dim>
void ConvectionDiffusionDPG<dim>::setup_system ()
{
dof_handler_trial.distribute_dofs (fe_trial);
dof_handler_test.distribute_dofs (fe_test);

const unsigned int no_of_trial_dofs = dof_handler_trial.n_dofs();
const unsigned int no_of_trial_dofs_per_cell = fe_trial.dofs_per_cell;
const unsigned int no_of_test_dofs_per_cell = fe_test.dofs_per_cell;

const FEValuesExtractors::Scalar index (dim+1);
const ComponentMask comp_mask = fe_trial.component_mask (index);

constraints.clear ();
DoFTools::make_hanging_node_constraints (dof_handler_trial, constraints);
DoFTools::make_zero_boundary_constraints (dof_handler_trial, constraints, comp_mask); // Apply zero boundary conditions to the trace of u.
constraints.close ();

DynamicSparsityPattern dsp (no_of_trial_dofs);
DoFTools::make_sparsity_pattern (dof_handler_trial, dsp);
constraints.condense (dsp);
sparsity_pattern.copy_from (dsp);

system_matrix.reinit (sparsity_pattern);

solution.reinit (no_of_trial_dofs);
right_hand_side.reinit (no_of_trial_dofs);
local_bilinear_form_values.reinit (no_of_trial_dofs_per_cell*no_of_test_dofs_per_cell);
local_optimal_test_functions.reinit (no_of_trial_dofs_per_cell*no_of_test_dofs_per_cell);
}

// Assemble the system

template <int dim>
void ConvectionDiffusionDPG<dim>::assemble_system ()
{
const QGauss<dim>  quadrature_formula_cell (degree+degree_offset+2);
const QGauss<dim-1>  quadrature_formula_face (degree+degree_offset+2);

const unsigned int no_of_quad_points_cell = quadrature_formula_cell.size();
const unsigned int no_of_trial_dofs_per_cell = fe_trial.dofs_per_cell;
const unsigned int no_of_test_dofs_per_cell = fe_test.dofs_per_cell;

typename DoFHandler<dim>::active_cell_iterator trial_cell = dof_handler_trial.begin_active(), final_cell = dof_handler_trial.end();
typename DoFHandler<dim>::active_cell_iterator test_cell = dof_handler_test.begin_active();

FEValues<dim> fe_values_trial_cell (fe_trial, quadrature_formula_cell, update_values | update_quadrature_points | update_JxW_values);
FEValues<dim> fe_values_test_cell (fe_test, quadrature_formula_cell, update_values | update_gradients | update_quadrature_points | update_JxW_values);
FEFaceValues<dim> fe_values_trial_face (fe_trial, quadrature_formula_face, update_values | update_quadrature_points | update_JxW_values);
FEFaceValues<dim> fe_values_test_face (fe_test, quadrature_formula_face, update_values | update_quadrature_points | update_JxW_values | update_normal_vectors);
   
FullMatrix<double> local_matrix (no_of_trial_dofs_per_cell, no_of_trial_dofs_per_cell);
Vector<double> local_rhs_values (no_of_trial_dofs_per_cell);
std::vector<double> forcing_values (no_of_quad_points_cell);
std::vector<types::global_dof_index> local_dof_indices_trial (no_of_trial_dofs_per_cell);

    for (; trial_cell!=final_cell; ++trial_cell, ++test_cell)
    {
    fe_values_trial_cell.reinit(trial_cell); fe_values_test_cell.reinit(test_cell);

    compute_local_bilinear_form_values (trial_cell, test_cell, fe_values_trial_cell, fe_values_test_cell, fe_values_trial_face, fe_values_test_face, local_bilinear_form_values);
	compute_local_optimal_test_functions (trial_cell, fe_values_trial_cell, fe_values_test_cell, local_bilinear_form_values, local_optimal_test_functions);

	Forcing<dim>().value_list (fe_values_test_cell.get_quadrature_points(), forcing_values);

	trial_cell->get_dof_indices (local_dof_indices_trial);

	local_matrix = 0; local_rhs_values = 0;

	// Assemble the local contributions to the system matrix.
	for (unsigned int i = 0; i < no_of_trial_dofs_per_cell; ++i)
	    for (unsigned int j = 0; j < no_of_trial_dofs_per_cell; ++j)
		    for (unsigned int k = 0; k < no_of_test_dofs_per_cell; ++k)
			{
			local_matrix(i,j) += local_optimal_test_functions(k + i*no_of_test_dofs_per_cell)*local_bilinear_form_values(k + j*no_of_test_dofs_per_cell);
			}

    // Place the local system matrix contributions in the system matrix.
	for (unsigned int i = 0; i < no_of_trial_dofs_per_cell; ++i)
	    for (unsigned int j = 0; j < no_of_trial_dofs_per_cell; ++j)
		{
		system_matrix(local_dof_indices_trial[i], local_dof_indices_trial[j]) += local_matrix(i,j);
		}

	    // Assemble the local contributions to the right hand side vector.
	    for (unsigned int quad_point = 0; quad_point < no_of_quad_points_cell; ++quad_point)
		{
    	    for (unsigned int k = 0; k < no_of_test_dofs_per_cell; ++k)
		    {
            unsigned int comp_k = fe_test.system_to_base_index(k).first.first;

            if (comp_k == 0)
			{
			double test_cell_value = forcing_values[quad_point]*fe_values_test_cell.shape_value_component(k,quad_point,0)*fe_values_test_cell.JxW(quad_point);

			    for (unsigned int i = 0; i < no_of_trial_dofs_per_cell; ++i)
	     		{
				local_rhs_values(i) += local_optimal_test_functions(k + i*no_of_test_dofs_per_cell)*test_cell_value;
				}
            }
		    }
        }

		// Place the local right hand side contributions in the right hand side vector.
		for (unsigned int i = 0; i < no_of_trial_dofs_per_cell; ++i)
		{
		right_hand_side(local_dof_indices_trial[i]) += local_rhs_values(i);
		}
    }
}

// Solve the system. 
// Using a CG solver since DPG yields a symmetric positive definite matrix. TODO: Work on preconditioner.

template <int dim>
void ConvectionDiffusionDPG<dim>::solve ()
{
SolverControl solver_control (2000, 1e-12);
SolverCG<> solver (solver_control);

constraints.condense (system_matrix, right_hand_side);

SparseILU<double> ilu;
ilu.initialize (system_matrix);
solver.solve (system_matrix, solution, right_hand_side, ilu);

constraints.distribute (solution);
}

// Output the solution. 
// Currently outputting: GNUPLOT.

template <int dim>
void ConvectionDiffusionDPG<dim>::output_solution () const
{
std::vector<std::string> solution_names (1, "solution");
for (unsigned int d = 0; d < dim; ++d) {solution_names.push_back ("gradient");}
solution_names.push_back ("skeleton solution");
solution_names.push_back ("flux");

std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation (1, DataComponentInterpretation::component_is_scalar);
for (unsigned int d = 0; d < dim; ++d) {data_component_interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);}
data_component_interpretation.push_back (DataComponentInterpretation::component_is_scalar);
data_component_interpretation.push_back (DataComponentInterpretation::component_is_scalar);

DataOut<dim> data_out;
data_out.attach_dof_handler (dof_handler_trial);
data_out.add_data_vector (solution, solution_names, DataOut<dim>::type_dof_data, data_component_interpretation);
data_out.build_patches ();

const std::string filename = "solution.gnuplot";

std::ofstream gnuplot_output (filename.c_str());
data_out.write_gnuplot (gnuplot_output);
}

// Intermediate function needed in assemble_system to assemble the system matrix. 
// Takes current cell information as an input and outputs a vector whose k + i*no_of_test_dofs_per_cell entry corresponds to B(phi_i, psi_k) where {phi_i} is a local basis for U_h and {psi_k} is a local basis for the the full enriched space V_h.
// B({u,sigma,trace(u),trace(sigma)},{v,tau}) := (u, div(tau) - b*grad(v))_K + (sigma, tau/epsilon + grad(v))_K - (trace(u), tau*n)_dK + (trace(sigma), v)_dK.

template <int dim>
void ConvectionDiffusionDPG<dim>::compute_local_bilinear_form_values (const typename DoFHandler<dim>::active_cell_iterator &trial_cell, const typename DoFHandler<dim>::active_cell_iterator &test_cell, const FEValues<dim> &fe_values_trial_cell, const FEValues<dim> &fe_values_test_cell, FEFaceValues<dim> &fe_values_trial_face, FEFaceValues<dim> &fe_values_test_face, Vector<double> &local_bilinear_form_values)
{
const unsigned int no_of_quad_points_cell = fe_values_trial_cell.get_quadrature().size();
const unsigned int no_of_quad_points_face = fe_values_trial_face.get_quadrature().size();
const unsigned int no_of_trial_dofs_per_cell = fe_trial.dofs_per_cell;
const unsigned int no_of_test_dofs_per_cell = fe_test.dofs_per_cell;

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
			values(d+1) = (1/epsilon)*fe_values_test_cell.shape_value_component(k,quad_point,d+1) + fe_values_test_cell.shape_grad_component(k,quad_point,0)[d];
		    }

        values *= fe_values_test_cell.JxW(quad_point);

		    for (unsigned int i = 0; i < no_of_trial_dofs_per_cell; ++i)
	        {
	        unsigned int comp_i = fe_trial.system_to_base_index(i).first.first;

	        if (comp_i == 0)
            {
			    for (unsigned int d = 0; d < dim + 1; ++d)
		        {
				local_bilinear_form_values(k + i*no_of_test_dofs_per_cell) += fe_values_trial_cell.shape_value_component(i,quad_point,d)*values(d);
				}
            }
	        }
	    }
    }

	for (unsigned int face = 0; face < 2*dim; ++face)
    {
	fe_values_trial_face.reinit(trial_cell, face); fe_values_test_face.reinit(test_cell, face);

    const std::vector<Tensor<1,dim> > &normals = fe_values_test_face.get_normal_vectors();

	    for (unsigned int quad_point = 0; quad_point < no_of_quad_points_face; ++quad_point)
		    for (unsigned int k = 0; k < no_of_test_dofs_per_cell; ++k)
            {
            double taudotnormal = 0; double test_face_value = fe_values_test_face.shape_value_component(k,quad_point,0);

                for (unsigned int d = 0; d < dim; ++d)
                {
                taudotnormal += fe_values_test_face.shape_value_component(k,quad_point,d+1)*normals[quad_point][d];
                }
                
            taudotnormal *= fe_values_test_face.JxW(quad_point); test_face_value *= std::pow(-1,face+1)*fe_values_test_face.JxW(quad_point);

		        for (unsigned int i = 0; i < no_of_trial_dofs_per_cell; ++i)		    		
                {
		        unsigned int comp_i = fe_trial.system_to_base_index(i).first.first;

				if (comp_i == 1)
				{
				local_bilinear_form_values(k + i*no_of_test_dofs_per_cell) += fe_values_trial_face.shape_value_component(i,quad_point,dim+2)*test_face_value - fe_values_trial_face.shape_value_component(i,quad_point,dim+1)*taudotnormal;
				}
                }
            }
    }
}

// Intermediate function needed in assemble_system to compute the optimal test functions. 
// Takes current cell information and bilinear form values as an input and outputs a vector whose k + i*no_of_test_dofs_per_cell entry corresponds to the weighting of the {psi_k} local basis function in the local basis expansion of the ith test function in the full enriched space V_h.

template <int dim>
void ConvectionDiffusionDPG<dim>::compute_local_optimal_test_functions (const typename DoFHandler<dim>::active_cell_iterator &trial_cell, const FEValues<dim> &fe_values_trial_cell, const FEValues<dim> &fe_values_test_cell, const Vector<double> &local_bilinear_form_values, Vector<double> &local_optimal_test_functions)
{
const unsigned int no_quad_points_cell = fe_values_trial_cell.get_quadrature().size();
const unsigned int no_of_trial_dofs_per_cell = fe_trial.dofs_per_cell;
const unsigned int no_of_test_dofs_per_cell = fe_test.dofs_per_cell;

FullMatrix<double> V_basis_matrix (no_of_test_dofs_per_cell, no_of_test_dofs_per_cell);
Vector<double> U_basis_rhs (no_of_test_dofs_per_cell);
Vector<double> optimal_test_function (no_of_test_dofs_per_cell);
std::vector<Vector<double> > convection_values (no_quad_points_cell, Vector<double>(dim));

Convection<dim>().vector_value_list (fe_values_test_cell.get_quadrature_points(), convection_values);

double vol = trial_cell->measure ();

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

				V_basis_matrix(k,l) += (fmin(1/sqrt(epsilon),1/sqrt(vol))*fe_values_test_cell.shape_value_component(k,quad_point,d+1)+sqrt(epsilon)*fe_values_test_cell.shape_grad_component(k,quad_point,0)[d])*(fmin(1/sqrt(epsilon),1/sqrt(vol))*fe_values_test_cell.shape_value_component(l,quad_point,d+1)+sqrt(epsilon)*fe_values_test_cell.shape_grad_component(l,quad_point,0)[d])*fe_values_test_cell.JxW(quad_point);
				}

			V_basis_matrix(k,l) += (divtau - fe_values_test_cell.shape_grad_component(k,quad_point,0)*convection)*(divsigma - fe_values_test_cell.shape_grad_component(l,quad_point,0)*convection)*fe_values_test_cell.JxW(quad_point);
            }
    } 
           
    for (unsigned int k = 0; k < no_of_test_dofs_per_cell; ++k)
        for (unsigned int l = 0; l < k + 1; ++l)
        {
        V_basis_matrix(l,k) = V_basis_matrix(k,l);
        }

V_basis_matrix.gauss_jordan (); // Invert the V basis matrix in preparation for finding the optimal test function basis coefficients.
 
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
    GridGenerator::hyper_cube (triangulation, -1, 1, true); triangulation.refine_global (5); // Creates the triangulation and globally refines n times.
    
	setup_system ();
    assemble_system ();
	solve ();
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
