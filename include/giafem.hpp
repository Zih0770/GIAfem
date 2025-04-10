#ifndef GIAFEM_HPP
#define GIAFEM_HPP

#include "mfem.hpp"  // Include MFEM for GridFunction, Mesh, etc.
#include <vector>
#include <string>
#include <utility>

namespace giafem
{
    using namespace mfem;

class VeOperator : public TimeDependentOperator
{
protected:
    Coefficient &tau, &lamb, &mu;
    Array<int> etl;
    FiniteElementSpace &fes_u;
    FiniteElementSpace &fes_m;
    FiniteElementSpace &fes_w;
    GridFunction &u_gf;
    GridFunction &m_gf;
    GridFunction &d_gf;
    GridFunction d0_gf;
    mutable GridFunction lamb_gf, mu_gf, tau_gf;
    mutable VectorArrayCoefficient force;
    BilinearForm *K;
    mutable SparseMatrix Kmat;
    SparseMatrix *T;
    mutable CGSolver K_solver;
    GSSmoother K_prec;
    real_t current_dt;

    mutable Vector z;

public:
    VeOperator(FiniteElementSpace &fes_u_, FiniteElementSpace &fes_m_, FiniteElementSpace &fes_w_, Coefficient &lamb_, Coefficient &mu_, Coefficient &tau_, const Vector &u_vec, const Vector &m_vec, GridFunction &u_gf_, GridFunction &m_gf_, GridFunction &d_gf_);

    void Mult(const Vector &m_vec, Vector &dm_dt_vec) const override;

    void ImplicitSolve(const real_t dt, const Vector &m_vec, Vector &k_vec) override;

    const GridFunction &GetTau() const { return tau_gf; }
    const GridFunction &GetLamb() const { return lamb_gf; }
    const GridFunction &GetMu()  const { return mu_gf; }
    const GridFunction &GetDev() const { return d_gf; }
    const GridFunction &GetDev0() const { return d0_gf; }

    ~VeOperator() override;
};


    class BaileySolver : public ODESolver
    {
    private:
        Vector dxdt;
        Vector d_vec_old, d_vec_new;

    public:
        void Init(TimeDependentOperator &f_) override;
        void Step(Vector &x, real_t &t, real_t &dt) override;
    };

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

