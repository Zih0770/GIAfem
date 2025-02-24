// giafem.cpp
#include "giafem.hpp"
#include <cmath>
#include <limits>
#include <functional>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

namespace giafem
{

void TensorFieldCoefficient::Eval(mfem::DenseMatrix &M, mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
{
    int e = T.ElementNo;

    const mfem::FiniteElement *fe = fespace->GetFE(e);
    const mfem::IntegrationRule *ir = &mfem::IntRules.Get(fe->GetGeomType(), 2 * fe->GetOrder() + 2);

    int nq = ir->GetNPoints();
    int closest_idx = 0;
    double min_dist = std::numeric_limits<double>::max();

    for (int i = 0; i < nq; i++)
    {
        double dist = (ip.x - ir->IntPoint(i).x) * (ip.x - ir->IntPoint(i).x) +
                      (ip.y - ir->IntPoint(i).y) * (ip.y - ir->IntPoint(i).y) +
                      (ip.z - ir->IntPoint(i).z) * (ip.z - ir->IntPoint(i).z);
        if (dist < min_dist)
        {
            min_dist = dist;
            closest_idx = i;
        }
    }

    M = m_storage[e][closest_idx];
}

//ViscoelasticRHSIntegrator::ViscoelasticRHSIntegrator(mfem::PWConstCoefficient &mu, mfem::MatrixRestrictedCoefficient &m_)
//        : mu_func(mu), m(m_) {}

    void ViscoelasticRHSIntegrator::AssembleRHSElementVect(const mfem::FiniteElement &el,
                                                           mfem::ElementTransformation &Tr,
                                                           mfem::Vector &elvec)
    {
        int dof = el.GetDof();
        int dim = el.GetDim();
        elvec.SetSize(dof * dim);
        elvec = 0.0;

        const mfem::IntegrationRule *ir = &mfem::IntRules.Get(Tr.GetGeometryType(), 2 * el.GetOrder());
        mfem::DenseMatrix m_tensor(dim, dim);
        mfem::DenseMatrix dshape(dof, dim);
        mfem::Vector force(dof * dim);
        force = 0.0;

        for (int k = 0; k < ir->GetNPoints(); k++)
        {
            const mfem::IntegrationPoint &ip = ir->IntPoint(k);
            Tr.SetIntPoint(&ip);

            m.Eval(m_tensor, Tr, ip);
            real_t m_tr = m_tensor.Trace();
            real_t mu_val = mu.Eval(Tr, ip);
            real_t fac = Tr.Weight() * ip.weight * 2.0 * mu_val;

            el.CalcPhysDShape(Tr, dshape);

            for (int s = 0; s < dof; s++)
            {
                for (int i = 0; i < dim; i++)
                {
                    real_t val = 0.0;
                    for (int j = 0; j < dim; j++)
                    {
                        val += (m_tensor(i, j) + m_tensor(j, i)) / 2.0 * dshape(s, j) - m_tr / 3.0 * dshape(s, i);
                    }
                    force(s + i * dof) += val * fac;
                }
            }

        }

        elvec = force;
    }

    void MatrixGradLFIntegrator::AssembleRHSElementVect(const mfem::FiniteElement &el,
                                                           mfem::ElementTransformation &Tr,
                                                           mfem::Vector &elvec)
    {
        int dof = el.GetDof();
        int dim = el.GetDim();
        elvec.SetSize(dof * dim);
        elvec = 0.0;

        const mfem::IntegrationRule *ir = &mfem::IntRules.Get(Tr.GetGeometryType(), 2 * el.GetOrder());
        mfem::DenseMatrix m_tensor(dim, dim);
        mfem::DenseMatrix dshape(dof, dim);
        mfem::Vector force(dof * dim);
        force = 0.0;

        for (int k = 0; k < ir->GetNPoints(); k++)
        {
            const mfem::IntegrationPoint &ip = ir->IntPoint(k);
            Tr.SetIntPoint(&ip);

            m.Eval(m_tensor, Tr, ip);
            real_t fac = Tr.Weight() * ip.weight;

            el.CalcPhysDShape(Tr, dshape);

            for (int s = 0; s < dof; s++)
            {
                for (int i = 0; i < dim; i++)
                {
                    real_t val = 0.0;
                    for (int j = 0; j < dim; j++)
                    {
                        val += m_tensor(i, j) * dshape(s, j);
                    }
                    force(s + i * dof) += val * fac;
                }
            }

        }

        elvec = force;
    }


//Turn a grid function into a dense matrix
GF2Mat::GF2Mat(const mfem::GridFunction &grid_func, const mfem::Array<int> &dof_indices)
    : mfem::DenseMatrix(dof_indices.Size(), grid_func.FESpace()->GetVDim())
{
    int dim = grid_func.FESpace()->GetVDim();
    int ndofs = dof_indices.Size();
    mfem::Vector el_vals(ndofs * dim);
    
    grid_func.GetSubVector(dof_indices, el_vals);

    for (int d = 0; d < dim; d++)
    {
        for (int i = 0; i < ndofs; i++)
        {
            (*this)(i, d) = el_vals(i + d * ndofs);
        }
    }
}

void FieldUtils::Strain_ip(const mfem::GridFunction &u,
                        mfem::ElementTransformation &Tr,
                        const mfem::IntegrationPoint &ip,
                        mfem::DenseMatrix &strain)
{
    int dim = Tr.GetSpaceDim();
    strain.SetSize(dim, dim);
    strain = 0.0;

    Tr.SetIntPoint(&ip);
    const FiniteElement *el = u.FESpace()->GetFE(Tr.ElementNo);
    int ndofs = el->GetDof();

    Vector u_vec;
    u.GetElementDofValues(Tr.ElementNo, u_vec);
    
    DenseMatrix dshape(ndofs, dim);//First-derivatives of shape functions
    el->CalcDShape(ip, dshape);

    DenseMatrix u_mat(u_vec.GetData(), ndofs, dim);

    //compute grad_u in the reference element
    DenseMatrix grad_ref(dim, dim);
    MultAtB(u_mat, dshape, grad_ref);

    //convert to grad_u in the physical element
    const DenseMatrix &Jinv = Tr.InverseJacobian();
    DenseMatrix grad_phys(dim, dim); 
    MultAtB(grad_ref, Jinv, grad_phys);

    // Compute the symmetric part of grad_u (strain tensor)
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            strain(i, j) = 0.5 * (grad_phys(i, j) + grad_phys(j, i));
        }
    }
}

void FieldUtils::DevStrain_ip(const mfem::GridFunction &u,
                           mfem::ElementTransformation &Tr,
                           const mfem::IntegrationPoint &ip,
                           mfem::DenseMatrix &deviatoric_strain)
{
    DenseMatrix strain;
    Strain_ip(u, Tr, ip, strain); // Compute the strain tensor first

    int dim = strain.Width();
    deviatoric_strain.SetSize(dim, dim);
    deviatoric_strain = strain;

    // Compute the trace
    double trace = 0.0;
    for (int i = 0; i < dim; i++)
    {
        trace += strain(i, i);
    }
    trace /= dim;

    for (int i = 0; i < dim; i++)
    {
        deviatoric_strain(i, i) -= trace;
    }
}

void FieldUtils::Strain(const GridFunction &u, GridFunction &strain)
{
    const FiniteElementSpace *fes_u = u.FESpace();
    const FiniteElementSpace *fes_strain = strain.FESpace();
    
    int dim = fes_u->GetVDim();

    Array<int> vdofs;
    Vector strain_vec;
    DenseMatrix F;
    DofTransformation *Tr_dof = NULL;

    for (int i = 0; i < fes_strain->GetNE(); i++)
    {
        Tr_dof = fes_strain->GetElementVDofs(i, vdofs);
        strain_vec.SetSize(vdofs.Size());

        ElementTransformation *Tr = fes_strain->GetElementTransformation(i);
        const FiniteElement *el_strain = fes_strain->GetFE(i);
        const IntegrationRule& ir = el_strain->GetNodes();

        int ndofs = ir.GetNPoints();

        for (int j = 0; j < ndofs; j++)
        {
            const IntegrationPoint& ip = ir.IntPoint(j);
            Tr->SetIntPoint(&ip);
            u.GetVectorGradient(*Tr, F);

            constexpr auto half = static_cast<real_t>(1) / static_cast<real_t>(2);

            if (dim == 2) {
              strain_vec(j) = F(0, 0);
              strain_vec(j + ndofs) = half * (F(1, 0) + F(0, 1));
              strain_vec(j + 2 * ndofs) = F(1, 1);
            } else if (dim == 3) {
              strain_vec(j) = F(0, 0);
              strain_vec(j + ndofs) = half * (F(1, 0) + F(0, 1));
              strain_vec(j + 2 * ndofs) = half * (F(2, 0) + F(0, 2));
              strain_vec(j + 3 * ndofs) = F(1, 1);
              strain_vec(j + 4 * ndofs) = half * (F(1, 2) + F(2, 1));
              strain_vec(j + 5 * ndofs) = F(2, 2);
            }
        }
        if (Tr_dof) {
          Tr_dof->TransformPrimal(strain_vec);
        }
    
        strain.SetSubVector(vdofs, strain_vec);
    }
}

void FieldUtils::DevStrain(const GridFunction &u, GridFunction &dev_strain)
{
    const FiniteElementSpace *fes_u = u.FESpace();
    const FiniteElementSpace *fes_dev = dev_strain.FESpace();
    
    int dim = fes_u->GetVDim();

    Array<int> vdofs;
    Vector strain_vec;
    DenseMatrix F;
    DofTransformation *Tr_dof = NULL;

    for (int i = 0; i < fes_dev->GetNE(); i++)
    {
        Tr_dof = fes_dev->GetElementVDofs(i, vdofs);
        strain_vec.SetSize(vdofs.Size());

        ElementTransformation *Tr = fes_dev->GetElementTransformation(i);
        const FiniteElement *el_strain = fes_dev->GetFE(i);
        const IntegrationRule& ir = el_strain->GetNodes();

        int ndofs = ir.GetNPoints();

        for (int j = 0; j < ndofs; j++)
        {
            const IntegrationPoint& ip = ir.IntPoint(j);
            Tr->SetIntPoint(&ip);
            u.GetVectorGradient(*Tr, F);

            constexpr auto half = static_cast<real_t>(1) / static_cast<real_t>(2);
            constexpr auto third = static_cast<real_t>(1) / static_cast<real_t>(3);

            if (dim == 2) {
              auto trace = F(0, 0) + F(1, 1);
              strain_vec(j) = F(0, 0) - half * trace;
              strain_vec(j + ndofs) = half * (F(1, 0) + F(0, 1));
            } else if (dim == 3) {
              auto trace = F(0, 0) + F(1, 1) + F(2, 2);
              strain_vec(j) = F(0, 0) - third * trace;
              strain_vec(j + ndofs) = half * (F(1, 0) + F(0, 1));
              strain_vec(j + 2 * ndofs) = half * (F(2, 0) + F(0, 2));
              strain_vec(j + 3 * ndofs) = F(1, 1) - third * trace;
              strain_vec(j + 4 * ndofs) = half * (F(1, 2) + F(2, 1));
            }
        }
        if (Tr_dof) {
          Tr_dof->TransformPrimal(strain_vec);
        }
    
        dev_strain.SetSubVector(vdofs, strain_vec);
    }
}

/*
void ViscoelasticIntegrator::AssembleElementMatrix(const mfem::FiniteElement &el,
                                                   mfem::ElementTransformation &Tr,
                                                   mfem::DenseMatrix &elmat)
{
    int dof = el.GetDof();
    int dim = el.GetDim();
    const IntegrationRule *ir = &mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 2);

    elmat.SetSize(dof * dim, dof * dim);
    elmat = 0.0;

    DenseMatrix shape(dof), dshape_dx(dof, dim);
    DenseMatrix m_mat(dim), d_mat(dim), test_grad(dim), contrib(dim, dim);
    Vector stress_vec(dim), test_func(dim);

    for (int q = 0; q < ir->GetNPoints(); q++)
    {
        const IntegrationPoint &ip = ir->IntPoint(q);
        Tr.SetIntPoint(&ip);

        double weight = ip.weight * Tr.Weight();

        el.CalcPhysShape(Tr, shape);       // Shape functions
        el.CalcPhysDShape(Tr, dshape_dx); // Gradients of shape functions

        // Access the current values of m and d at the integration point
        m.GetVectorValue(Tr, ip, stress_vec);
        for (int i = 0; i < dim; i++)      // Convert stress_vec to matrix
            for (int j = 0; j < dim; j++)
                m_mat(i, j) = stress_vec[i * dim + j];

        d.GetVectorValue(Tr, ip, stress_vec);
        for (int i = 0; i < dim; i++)      // Convert stress_vec to matrix
            for (int j = 0; j < dim; j++)
                d_mat(i, j) = stress_vec[i * dim + j];

        // Contribution from the term -2 * mu * m : d'
        contrib.Set(0.0);
        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                contrib(i, j) += -2.0 * mu.Eval(Tr, ip) * m_mat(i, j);
            }
        }

        // Contribution from the term 2 * mu * (tau * \dot{m} + m - d) : m'
        DenseMatrix relaxation_term(dim, dim);
        relaxation_term.Set(0.0);
        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                relaxation_term(i, j) += mu.Eval(Tr, ip) * (m_mat(i, j) - d_mat(i, j));
            }
        }

        for (int i = 0; i < dof; i++)
        {
            for (int j = 0; j < dof; j++)
            {
                for (int k = 0; k < dim; k++)
                {
                    for (int l = 0; l < dim; l++)
                    {
                        elmat(i * dim + k, j * dim + l) += weight * contrib(k, l) * dshape_dx(i, k) * dshape_dx(j, l);
                        elmat(i * dim + k, j * dim + l) += weight * relaxation_term(k, l) * dshape_dx(i, k) * dshape_dx(j, l);
                    }
                }
            }
        }
    }
}
*/
/*
void ViscoelasticRHSIntegrator::AssembleRHSElementVector(const mfem::FiniteElement &el,
                                                         mfem::ElementTransformation &Tr,
                                                         mfem::Vector &elvec)
{
    int dof = el.GetDof();
    int dim = el.GetDim();
    const IntegrationRule *ir = &mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 2);

    elvec.SetSize(dof * dim);
    elvec = 0.0;

    DenseMatrix shape(dof), dshape_dx(dof, dim);
    DenseMatrix m_mat(dim), d_mat(dim), rhs_term(dim, dim);
    Vector stress_vec(dim);

    for (int q = 0; q < ir->GetNPoints(); q++)
    {
        const IntegrationPoint &ip = ir->IntPoint(q);
        Tr.SetIntPoint(&ip);

        double weight = ip.weight * Tr.Weight();

        el.CalcPhysShape(Tr, shape); // Shape functions

        // Access the current values of m and d at the integration point
        m.GetVectorValue(Tr, ip, stress_vec);
        for (int i = 0; i < dim; i++)      // Convert stress_vec to matrix
            for (int j = 0; j < dim; j++)
                m_mat(i, j) = stress_vec[i * dim + j];

        d.GetVectorValue(Tr, ip, stress_vec);
        for (int i = 0; i < dim; i++)      // Convert stress_vec to matrix
            for (int j = 0; j < dim; j++)
                d_mat(i, j) = stress_vec[i * dim + j];

        // Compute RHS term: 2 * mu * (tau * \dot{m} + m - d)
        rhs_term.Set(0.0);
        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                rhs_term(i, j) += 2.0 * mu.Eval(Tr, ip) * (m_mat(i, j) - d_mat(i, j));
            }
        }

        for (int i = 0; i < dof; i++)
        {
            for (int k = 0; k < dim; k++)
            {
                double val = 0.0;
                for (int l = 0; l < dim; l++)
                {
                    val += rhs_term(k, l) * shape(i);
                }
                elvec(i * dim + k) += weight * val;
            }
        }
    }
}
*/
    plot::plot(const GridFunction &sol, Mesh &msh)
        : solution(sol), mesh(msh) {}

    plot::~plot()
    {
        //delete[] data;
        std::cout << "Destroying plot object and releasing data." << std::endl;
    }

    parse::~parse()
    {
        std::cout << "Destroying parse object." << std::endl;
    }

    interp::~interp()
    {
        std::cout << "Destroying interp object." << std::endl;
    }


    void plot::EvaluateRadialSolution(int num_samples, double min_radius, double max_radius, 
		                        std::vector<double> &radii, std::vector<double> &values)
    {
        double mesh_max_radius = 0.0;
        
	for (int i = 0; i < mesh.GetNV(); i++)
        {
            Vector node;
            //mesh.GetVertex(i, node);
            const double* vertex_coords = mesh.GetVertex(i);
            node.SetSize(mesh.Dimension());
            for (int d = 0; d < mesh.Dimension(); d++)
            {
                node(d) = vertex_coords[d];
            }

	    double radius = node.Norml2();
            if (radius > mesh_max_radius)
                mesh_max_radius = radius;
        }
	max_radius = std::min(max_radius, mesh_max_radius);


	DenseMatrix points(mesh.Dimension(), num_samples);
        for (int i = 0; i < num_samples; i++)
        {
            double radius = min_radius + i * (max_radius - min_radius) / num_samples;
            points(0, i) = radius;
            if (mesh.Dimension() > 1) points(1, i) = 0.0;
            if (mesh.Dimension() > 2) points(2, i) = 0.0; 
	}
       

        Array<int> elem_ids(num_samples);
	Array<IntegrationPoint> ips(num_samples);
        
        mesh.FindPoints(points, elem_ids, ips);
        
	for (int i = 0; i < num_samples; i++)
        {
            double value = 0.0;
            int elem_id = elem_ids[i];
            if (elem_id >= 0) // Check if the point was found
            {
                ElementTransformation *trans = mesh.GetElementTransformation(elem_id);
                
		value = solution.GetValue(*trans, ips[i]);
                std::cout << "Point (" << points(0, i) << ", "
                          << (mesh.Dimension() > 1 ? points(1, i) : 0.0) << ") "
                          << "has value " << value << " in element " << elem_id << ".\n";
            }
            else
            {
                std::cerr << "Warning: Point (" << points(0, i) << ", "
                          << (mesh.Dimension() > 1 ? points(1, i) : 0.0) << ") "
                          << "is outside the mesh.\n";
            }

            radii.push_back(points(0, i));
            values.push_back(value);
	}
    }

    void plot::SaveRadialSolution(const std::string &filename, int num_samples, double min_radius, double max_radius)
    {
        std::vector<double> radii;
        std::vector<double> values;
        EvaluateRadialSolution(num_samples, min_radius, max_radius, radii, values);

        std::ofstream ofs(filename);
        //if (!outfile.is_open()) {
        //    std::cerr << "Error: Could not open the file " << filename << " for writing.\n";
        //    return;
        //}
	
	ofs << filename<<"\n";
        for (size_t i = 0; i < radii.size(); i++)
        {
            ofs << radii[i] << " " << values[i] << "\n";
        }
        ofs.close();
    }

    

    std::vector<std::vector<double>> parse::properties_1d(const std::string &filename)
    {
        std::ifstream infile(filename);
        if (!infile.is_open())
        {
            throw std::runtime_error("Unable to open file: " + filename);
        }

        std::vector<std::vector<double>> properties;
        std::string line;

        int num_columns = 0; // Local variable to track the number of columns

        std::getline(infile, line);
        std::getline(infile, line);
        std::getline(infile, line);

        // Parse the file
        while (std::getline(infile, line))
        {
            std::istringstream iss(line);
            std::vector<double> row;
            double value;

            // Read all columns in the row
            while (iss >> value)
            {
                row.push_back(value);
            }

            // Determine the number of columns from the first valid row
            if (properties.empty() && !row.empty())
            {
                num_columns = row.size();
            }

            // Ensure all rows have the same number of columns
            if (!row.empty() && row.size() != num_columns)
            {
                throw std::runtime_error("Inconsistent number of columns in file: " + filename);
            }

            if (!row.empty())
            {
                properties.push_back(row);
            }
        }

        infile.close();

        return properties;
    }


    mfem::Array<mfem::FunctionCoefficient *>
    interp::PWCoef_1D(const std::vector<std::pair<double, double>> &radius_property,
                                     int num_attributes,
                                     const std::string &method)
    {
        if (radius_property.empty())
        {
            throw std::runtime_error("Radius-property array is empty.");
        }

        // Define the interpolation function
        std::function<double(double)> interpolate;

        if (method == "linear")
        {
            interpolate = [radius_property](double radius) -> double {
                if (radius > radius_property.back().first)
                {
                    return 0.0; // Return zero outside the range
                }

                for (size_t i = 1; i < radius_property.size(); ++i)
                {
                    if (radius <= radius_property[i].first)
                    {
                        double r1 = radius_property[i - 1].first;
                        double r2 = radius_property[i].first;
                        double v1 = radius_property[i - 1].second;
                        double v2 = radius_property[i].second;

                        // Linear interpolation
                        return v1 + (v2 - v1) * (radius - r1) / (r2 - r1);
                    }
                }
                return 0.0; 
            };
        }
        else if (method == "spline")
        {
            throw std::runtime_error("Spline interpolation is not implemented yet.");
        }
        else
        {
            throw std::runtime_error("Unsupported interpolation method: " + method);
        }

        // Create FunctionCoefficients for each attribute
        mfem::Array<mfem::FunctionCoefficient *> coefficients;
        for (int i = 1; i <= num_attributes; ++i)
        {
            // Create a FunctionCoefficient for this attribute
            mfem::FunctionCoefficient *fc = new mfem::FunctionCoefficient([interpolate](const mfem::Vector &x) -> double {
                double r = x.Norml2();
                return interpolate(r);
            });

            coefficients.Append(fc);
        }

        return coefficients;
    }



} // namespace giafem

