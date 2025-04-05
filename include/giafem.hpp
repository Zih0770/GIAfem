#ifndef GIAFEM_HPP
#define GIAFEM_HPP

#include "mfem.hpp"  // Include MFEM for GridFunction, Mesh, etc.
#include <vector>
#include <string>
#include <utility>

namespace giafem
{
    using namespace mfem;

    class TensorFieldCoefficient : public mfem::MatrixCoefficient
    {
    private:
        const std::vector<std::vector<mfem::DenseMatrix>> &m_storage;
        mfem::FiniteElementSpace *fespace;

    public:
        TensorFieldCoefficient(const std::vector<std::vector<mfem::DenseMatrix>> &storage,
                               mfem::FiniteElementSpace *fes)
            : mfem::MatrixCoefficient(fes->GetMesh()->Dimension()), m_storage(storage), fespace(fes) {}

        virtual void Eval(mfem::DenseMatrix &M, mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip);
    };

    class ViscoelasticRHSIntegrator : public LinearFormIntegrator
    {
    private:
        Coefficient &mu;
        VectorCoefficient &m;

    public:
        ViscoelasticRHSIntegrator(Coefficient &mu_, VectorCoefficient &m_)
            : mu(mu_), m(m_) {}
        virtual void AssembleRHSElementVect(const mfem::FiniteElement &el,
                                           mfem::ElementTransformation &Tr,
                                           mfem::Vector &elvec);

        using LinearFormIntegrator::AssembleRHSElementVect;
    };

    class MatrixGradLFIntegrator : public mfem::LinearFormIntegrator
    {
    private:
        mfem::MatrixCoefficient &m;

    public:
        MatrixGradLFIntegrator(MatrixCoefficient &m_)
            : m(m_) {}
        virtual void AssembleRHSElementVect(const mfem::FiniteElement &el,
                                           mfem::ElementTransformation &Tr,
                                           mfem::Vector &elvec);

        using LinearFormIntegrator::AssembleRHSElementVect;
    };


    class FieldUtils
    {
    public:
        static void Strain_ip(const mfem::GridFunction &u,
                           mfem::ElementTransformation &Tr,
                           const mfem::IntegrationPoint &ip,
                           mfem::DenseMatrix &strain);

        static void DevStrain_ip(const mfem::GridFunction &u,
                              mfem::ElementTransformation &Tr,
                              const mfem::IntegrationPoint &ip,
                              mfem::DenseMatrix &deviatoric_strain);

        static void Strain(const mfem::GridFunction &u,
                           mfem::GridFunction &strain);

        static void DevStrain(const mfem::GridFunction &u,
                              mfem::GridFunction &dev_strain);
    };

    class ViscoelasticIntegrator : public mfem::BilinearFormIntegrator
    {
    private:
        double tau, dt;                    // Relaxation time
        mfem::Coefficient &mu;         // Second Lame parameter
 
    public:
        ViscoelasticIntegrator(double tau_, double dt_, mfem::Coefficient &mu_)
            : tau(tau_), dt(dt_), mu(mu_) {}

        virtual void AssembleElementMatrix(const mfem::FiniteElement &el,
                                           mfem::ElementTransformation &Tr,
                                           mfem::DenseMatrix &elmat) override;
    }; 

    class plot
    {
    public:
        // Constructor to initialize with solution and mesh
        plot(const GridFunction &solution, Mesh &mesh);

        ~plot();

        // Subroutine to evaluate the radial solution at evenly spaced points
        void EvaluateRadialSolution(int num_samples, double min_radius, double max_radius, std::vector<double> &radii, 
                                    std::vector<double> &values);

        // Optional subroutine to save the evaluated radial solution data to a file
        void SaveRadialSolution(const std::string &filename, int num_samples, double min_radius, double max_radius);

    private:
        const GridFunction &solution;  // Reference to the solution GridFunction
        Mesh &mesh;              // Reference to the mesh
    };


    class parse
    {
    public:
        ~parse();
        // Function to parse 1D properties from a file with dynamic columns
        static std::vector<std::vector<double>> properties_1d(const std::string &filename);
    };


    class interp
    {
    public:
        ~interp();
        // Function to interpolate a 1D property from dynamic data
        static mfem::Array<mfem::FunctionCoefficient *>
        PWCoef_1D(const std::vector<std::pair<double, double>> &radius_property,
                                 int num_attributes,
                                 const std::string &method = "linear");
    };

} // namespace giafem

#endif // GIAFEM_HPP

