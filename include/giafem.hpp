#ifndef GIAFEM_HPP
#define GIAFEM_HPP

#include "mfem.hpp"  // Include MFEM for GridFunction, Mesh, etc.
#include "mfemElasticity.hpp"
#include <cmath>
#include <vector>
#include <string>
#include <utility>
#include <iostream>

namespace giafem
{
    using namespace std;
    using namespace mfem;

    //Utilities
    inline void Visualize(ostream &os, ParMesh *mesh, ParGridFunction *deformed_nodes,
                          ParGridFunction *field, const char *field_name, bool init_vis = false)
    {
        if (!os)
        {
            return;
        }

        GridFunction *nodes = deformed_nodes;

        int owns_nodes = 0;
        mesh->SwapNodes(nodes, owns_nodes);

        os << "parallel " << mesh->GetNRanks() << " " << mesh->GetMyRank() << "\n";
        os << "solution\n" << *mesh << *field;
        mesh->SwapNodes(nodes, owns_nodes);

        if (init_vis)
        {
            os << "window_size 800 800\n";
            os << "window_title '" << field_name << "'\n";
            if (mesh->SpaceDimension() == 2)
            {
                os << "view 0 0\n"; 
                os << "keys jl\n";  
            }
            os << "keys cm\n";        
            os << "autoscale value\n";
            os << "pause\n";
        }
        os << flush;

        delete nodes;
    }


//Operators
class ViscoelasticOperator : public TimeDependentOperator
{
protected:
    Coefficient &tau, &lamb, &mu, &loading;
    ParFiniteElementSpace &fes_u, &fes_m, &fes_w, &fes_properties;
    ParGridFunction &u_gf, &m_gf, &d_gf;
    mutable ParGridFunction lamb_gf, mu_gf, tau_gf;
    Array<int> ess_tdof_list;
    ParBilinearForm *K;
    mutable PetscParMatrix Kmat;
    PetscPreconditioner *K_prec = NULL;
    mutable PetscPCGSolver K_solver;
    mutable mfemElasticity::RigidBodySolver rigid_solver;
    MixedBilinearForm B;
    MixedBilinearForm B2;
    DiscreteLinearOperator Dev;
    real_t current_dt;
    real_t rel_tol = 1e-8;
    real_t res_max = 1e-3;

    mutable Vector u_vec, d_vec, tau_vec;
    mutable Vector x_vec, b_vec;

public:
    ViscoelasticOperator(ParFiniteElementSpace &fes_u_, ParFiniteElementSpace &fes_m_, ParFiniteElementSpace &fes_properties_, ParFiniteElementSpace &fes_w_, 
               ParGridFunction &u_gf_, ParGridFunction &m_gf_, ParGridFunction &d_gf_, Coefficient &lamb_, Coefficient &mu_, Coefficient &tau_, Coefficient &loading_);

    void Mult(const Vector &m_vec, Vector &dm_dt_vec) const override;

    void ImplicitSolve(const real_t dt, const Vector &m_vec, Vector &dm_dt_vec) override;

    void CalcStrainEnergyDensity(ParGridFunction &w_gf);

    const ParGridFunction &GetTau() const { return tau_gf; }
    const ParGridFunction &GetLamb() const { return lamb_gf; }
    const ParGridFunction &GetMu()  const { return mu_gf; }

    ~ViscoelasticOperator() override {delete K, K_prec;}
};










//Interpolators
class GradInterpolator : public DiscreteInterpolator
{
protected:
    int dim;
public:
    GradInterpolator(int dim_ = 3) : dim(dim_) {}

    void AssembleElementMatrix2(const FiniteElement &u_fe,
                                const FiniteElement &e_fe,
                                ElementTransformation &Trans,
                                DenseMatrix &elmat) override;
};

class StrainInterpolator : public DiscreteInterpolator
{
protected:
    int dim, vdim;
    std::vector<std::pair<int, int>> IndexMap;

public:
    StrainInterpolator(int dim_ = 3) : dim(dim_) {
        vdim = dim * (dim + 1) / 2;
        if (dim == 2){
            IndexMap = {{0, 0}, {1, 0}, {1, 1}};
        } else{
            IndexMap = {{0, 0}, {1, 0}, {2, 0}, {1, 1}, {2, 1}, {2,2}};
        }
    }

    void AssembleElementMatrix2(const FiniteElement &u_fe,
                                const FiniteElement &e_fe,
                                ElementTransformation &Trans,
                                DenseMatrix &elmat) override;
};

class DevStrainInterpolator : public DiscreteInterpolator
{
protected:
    int dim, vdim;
    std::vector<std::pair<int, int>> IndexMap;
public:
    DevStrainInterpolator(int dim_ = 3) : dim(dim_) {
        vdim = dim * (dim + 1) / 2 - 1;
        if (dim == 2){
            IndexMap = {{0, 0}, {1, 0}};
        } else{
            IndexMap = {{0, 0}, {1, 0}, {2, 0}, {1, 1}, {2, 1}};
        }
    }

    void AssembleElementMatrix2(const FiniteElement &u_fe,
                                const FiniteElement &e_fe,
                                ElementTransformation &Trans,
                                DenseMatrix &elmat) override;
};

//Integrators
inline void OperatorContractionTracefree(DenseMatrix &B0, int dof, int dim, DenseMatrix &B)
{
   if (dim == 2){
       std::vector<DenseMatrix> columns(4);
       for (int i = 0; i < 4; i++)
           B0.GetSubMatrix(0, B0.Height(), i*dof, (i+1)*dof, columns[i]);

           columns[0].AddMatrix(-1.0, columns[3], 0, 0);
           B.SetSubMatrix(0, 0, columns[0]);
           columns[1].AddMatrix(columns[2], 0, 0);
           B.SetSubMatrix(0, dof, columns[1]);
   } else {
       std::vector<DenseMatrix> columns(9);
       for (int i = 0; i < 9; i++){
           B0.GetSubMatrix(0, B0.Height(), i*dof, (i+1)*dof, columns[i]); //cout<<"Column "<<i<<": "<<columns[i].FNorm()<<endl;
       }
       columns[0].AddMatrix(-1.0, columns[8], 0, 0);
       B.SetSubMatrix(0, 0, columns[0]);
       columns[1].AddMatrix(columns[3], 0, 0);
       B.SetSubMatrix(0, dof, columns[1]);
       columns[2].AddMatrix(columns[6], 0, 0);
       B.SetSubMatrix(0, 2 * dof, columns[2]);
       columns[4].AddMatrix(-1.0, columns[8], 0, 0);
       B.SetSubMatrix(0, 3 * dof, columns[4]);
       columns[5].AddMatrix(columns[7], 0, 0);
       B.SetSubMatrix(0, 4 * dof, columns[5]);
   }
}


class ViscoelasticForcing : public BilinearFormIntegrator
{
protected:
    int dim, vdim, vdim_full;
    Coefficient &mu;
    std::vector<std::pair<int, int>> IndexMap;
public:
    ViscoelasticForcing(Coefficient &mu_, int dim_ = 3) : dim(dim_), mu(mu_) {
        vdim_full = dim * dim; 
        vdim = dim * (dim + 1) / 2 - 1;
        if (dim == 2){
            IndexMap = {{0, 0}, {1, 0}};
        } else{
            IndexMap = {{0, 0}, {1, 0}, {2, 0}, {1, 1}, {2, 1}};
        }
    }

    virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                        const FiniteElement &test_fe,
                                        ElementTransformation &Trans,
                                        DenseMatrix &elmat);
};


class ViscoelasticRHSIntegrator : public LinearFormIntegrator
{
    private:
        Coefficient &mu;
        VectorCoefficient &m;

    public:
        ViscoelasticRHSIntegrator(Coefficient &mu_, VectorCoefficient &m_) : mu(mu_), m(m_) { }

        void AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &Tr, Vector &elvec);

        using LinearFormIntegrator::AssembleRHSElementVect;
};


//Coefficients
class StrainEnergyCoefficient : public Coefficient
{
protected:
    GridFunction &u_gf;
    Coefficient &lambda, &mu;
    DenseMatrix grad_u;

public:
    StrainEnergyCoefficient(GridFunction &displacement, Coefficient &lambda_, Coefficient &mu_)
        : u_gf(displacement), lambda(lambda_), mu(mu_) {}

    real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override{
        real_t lambda_val = lambda.Eval(T, ip);
        real_t mu_val = mu.Eval(T, ip);

        u_gf.GetVectorGradient(T, grad_u);

        DenseMatrix strain(3, 3);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                strain(i, j) = 0.5 * (grad_u(i, j) + grad_u(j, i));

        real_t trace_eps = strain(0, 0) + strain(1, 1) + strain(2, 2);

        real_t strain_energy_density = 0.5 * lambda_val * trace_eps * trace_eps;

        real_t eps_inner = 0.0;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                eps_inner += strain(i, j) * strain(i, j);

        strain_energy_density += mu_val * eps_inner;

        return strain_energy_density;
    }

    ~StrainEnergyCoefficient() override { }
};



//Solver-related
class RigidTranslation : public mfem::VectorCoefficient {
    private:
        int _component;

    public:
        RigidTranslation(int dimension, int component)
            : VectorCoefficient(dimension), _component{component} {
                MFEM_ASSERT(component >= 0 && component < dimension,
                        "component out of range");
            }

        void SetComponent(int component) { _component = component; }

        void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip) override {
            V.SetSize(vdim);
            for (auto i = 0; i < vdim; i++) {
                V[i] = i == _component ? 1 : 0;
            }
        }
};

class RigidRotation : public mfem::VectorCoefficient {
    private:
        int _component;
        Vector _x;

    public:
        RigidRotation(int dimension, int component)
            : VectorCoefficient(dimension), _component{component} {
                MFEM_ASSERT(component >= 0 && component < dimension,
                        "component out of range");
                MFEM_ASSERT(dimension == 3 || component == 2,
                        "In two dimensions only z-rotation defined");
            }

        void SetComponent(int component) { _component = component; }

        void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip) override {
            V.SetSize(vdim);
            _x.SetSize(vdim);
            T.Transform(ip, _x);
            if (_component == 0) {
                V[0] = 0;
                V[1] = -_x[2];
                V[2] = _x[1];
            } else if (_component == 1) {
                V[0] = _x[2];
                V[1] = 0;
                V[2] = -_x[0];
            } else {
                V[0] = -_x[1];
                V[1] = _x[0];
                if (vdim == 3) V[2] = 0;
            }
        }
};

class RigidBodySolver : public Solver {
    private:
        FiniteElementSpace *_fes;
        std::vector<Vector *> _u;
        Solver *_solver = nullptr;
        mutable Vector _b;
        const bool _parallel;

#ifdef MFEM_USE_MPI
        ParFiniteElementSpace *_pfes;
        MPI_Comm _comm;
#endif

        real_t Dot(const Vector &x, const Vector &y) const {
#ifdef MFEM_USE_MPI
            return _parallel ? InnerProduct(_comm, x, y) : InnerProduct(x, y);
#else
            return InnerProduct(x, y);
#endif
        }

        real_t Norm(const Vector &x) const {
            return std::sqrt(Dot(x, x));
        }

        void GramSchmidt() {
            for (auto i = 0; i < GetNullDim(); i++) {
                auto &u = *_u[i];
                for (auto j = 0; j < i; j++) {
                    auto &v = *_u[j];
                    auto product = Dot(u, v);
                    u.Add(-product, v);
                }
                auto norm = Norm(u);
                u /= norm;
            }
        }

        int GetNullDim() const {
            auto vDim = _fes->GetVDim();
            return vDim * (vDim + 1) / 2;
        }

        void ProjectOrthogonalToRigidBody(const Vector &x, Vector &y) const {
            y = x;
            for (auto i = 0; i < GetNullDim(); i++) {
                auto &u = *_u[i];
                auto product = Dot(y, u);
                y.Add(-product, u);
            }
        }

    public:
        RigidBodySolver(FiniteElementSpace *fes) : Solver(0, false), _fes{fes}, _parallel{false} {
                auto vDim = _fes->GetVDim();
                MFEM_ASSERT(vDim == 2 || vDim == 3, "Dimensions must be two or three");

                // Set up a temporary gridfunction.
                auto u = GridFunction(_fes);

                // Set the translations.
                for (auto component = 0; component < vDim; component++) {
                    auto v = RigidTranslation(vDim, component);
                    u.ProjectCoefficient(v);
                    auto tv = new Vector();
                    u.GetTrueDofs(*tv);
                    _u.push_back(tv);
                }

                // Set the rotations.
                if (vDim == 2) {
                    auto v = RigidRotation(vDim, 2);
                    u.ProjectCoefficient(v);
                    auto tv = new Vector();
                    u.GetTrueDofs(*tv);
                    _u.push_back(tv);
                } else {
                    for (auto component = 0; component < vDim; component++) {
                        auto v = RigidRotation(vDim, component);
                        u.ProjectCoefficient(v);
                        auto tv = new Vector();
                        u.GetTrueDofs(*tv);
                        _u.push_back(tv);
                    }
                }

                GramSchmidt();
            }

#ifdef MFEM_USE_MPI
        RigidBodySolver(MPI_Comm comm, mfem::ParFiniteElementSpace *fes)
            : mfem::Solver(0, false),
            _fes{fes},
            _pfes{fes},
            _comm{comm},
            _parallel{true} {
                auto vDim = _fes->GetVDim();
                MFEM_ASSERT(vDim == 2 || vDim == 3, "Dimensions must be two or three");

                // Set up a temporary ParGridfunction.
                auto u = mfem::ParGridFunction(_pfes);

                // Set the translations.
                for (auto component = 0; component < vDim; component++) {
                    auto v = RigidTranslation(vDim, component);
                    u.ProjectCoefficient(v);
                    auto *tv = u.GetTrueDofs();
                    _u.push_back(tv);
                }

                // Set the rotations.
                if (vDim == 2) {
                    auto v = RigidRotation(vDim, 2);
                    u.ProjectCoefficient(v);
                    auto *tv = u.GetTrueDofs();
                    _u.push_back(tv);
                } else {
                    for (auto component = 0; component < vDim; component++) {
                        auto v = RigidRotation(vDim, component);
                        u.ProjectCoefficient(v);
                        auto *tv = u.GetTrueDofs();
                        _u.push_back(tv);
                    }
                }

                GramSchmidt();
            }
#endif

        ~RigidBodySolver() {
            for (auto i = 0; i < GetNullDim(); i++) {
                delete _u[i];
            }
        }

        void SetSolver(Solver &solver) {
            _solver = &solver;
            height = _solver->Height();
            width = _solver->Width();
            MFEM_VERIFY(height == width, "Solver must be a square operator");
        }

        void SetOperator(const Operator &op) {
            MFEM_VERIFY(_solver, "Solver hasn't been set, call SetSolver() first.");
            _solver->SetOperator(op);
            height = _solver->Height();
            width = _solver->Width();
            MFEM_VERIFY(height == width, "Solver must be a square Operator!");
        }

        void Mult(const Vector &b, Vector &x) const {
            ProjectOrthogonalToRigidBody(b, _b);
            _solver->iterative_mode = iterative_mode;
            _solver->Mult(_b, x);
            ProjectOrthogonalToRigidBody(x, x);
        }
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



//Plotting
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

//Legacy
void visualize(std::ostream &os, Mesh *mesh, GridFunction *deformed_nodes,
               GridFunction *field, const char *field_name = NULL, bool init_vis = false);

class VeOperator : public TimeDependentOperator
{
protected:
    Coefficient &tau, &lamb, &mu, &loading;
    FiniteElementSpace &fes_u, &fes_m, &fes_w, &fes_properties;
    GridFunction &u_gf, &m_gf, &d_gf;
    mutable GridFunction lamb_gf, mu_gf, tau_gf;
    Array<int> ess_tdof_list;
    BilinearForm *K;
    mutable SparseMatrix Kmat;
    GSSmoother K_prec;
    mutable CGSolver K_solver;
    mutable mfemElasticity::RigidBodySolver rigid_solver;
    MixedBilinearForm B;
    MixedBilinearForm B2;
    DiscreteLinearOperator Dev;
    real_t current_dt;
    real_t rel_tol = 1e-8;
    real_t res_max = 1e-3;

    mutable Vector u_vec, d_vec, tau_vec;
    mutable Vector x_vec, b_vec;

public:
    VeOperator(FiniteElementSpace &fes_u_, FiniteElementSpace &fes_m_, FiniteElementSpace &fes_properties_, FiniteElementSpace &fes_w_, GridFunction &u_gf_, GridFunction &m_gf_, GridFunction &d_gf_, Coefficient &lamb_, Coefficient &mu_, Coefficient &tau_, Coefficient &loading_);

    void Mult(const Vector &m_vec, Vector &dm_dt_vec) const override;

    void ImplicitSolve(const real_t dt, const Vector &m_vec, Vector &dm_dt_vec) override;

    void CalcStrainEnergyDensity(GridFunction &w_gf);

    const GridFunction &GetTau() const { return tau_gf; }
    const GridFunction &GetLamb() const { return lamb_gf; }
    const GridFunction &GetMu()  const { return mu_gf; }

    ~VeOperator() override {delete K;}
};

class VeOperator_beam : public TimeDependentOperator
{
protected:
    Coefficient &tau, &lamb, &mu;
    FiniteElementSpace &fes_u;
    FiniteElementSpace &fes_m;
    FiniteElementSpace &fes_w;
    FiniteElementSpace &fes_properties;
    GridFunction &u_gf;
    GridFunction &m_gf;
    GridFunction &d_gf;
    GridFunction d0_gf;
    mutable GridFunction lamb_gf, mu_gf, tau_gf;
    Array<int> etl;
    mutable VectorArrayCoefficient force;
    Coefficient *loading;
    LinearFormIntegrator *boundary_integ; 
    BilinearForm *K;
    mutable SparseMatrix Kmat;
    mutable CGSolver K_solver;
    GSSmoother K_prec;
    real_t current_dt;
    DiscreteLinearOperator Dev;
    std::function<LinearFormIntegrator*()> create_boundary_integ;

    mutable Vector u_vec;
    mutable Vector d_vec;
    mutable Vector tau_vec;
    mutable Vector z;

public:
    VeOperator_beam(FiniteElementSpace &fes_u_, FiniteElementSpace &fes_m_, FiniteElementSpace &fes_w_, Coefficient &lamb_, Coefficient &mu_, Coefficient &tau_, GridFunction &u_gf_, GridFunction &m_gf_, GridFunction &d_gf_);

    void Mult(const Vector &m_vec, Vector &dm_dt_vec) const override;

    void ImplicitSolve(const real_t dt, const Vector &m_vec, Vector &k_vec) override;

    const GridFunction &GetTau() const { return tau_gf; }
    const GridFunction &GetLamb() const { return lamb_gf; }
    const GridFunction &GetMu()  const { return mu_gf; }

    ~VeOperator_beam() override {delete K;}
};


class BaileySolver_test : public ODESolver
{
    private:
        Vector dxdt;
        Vector d_vec_old, d_vec_new;
        real_t tau = 1.0;

    public:
        void Init(TimeDependentOperator &f_) override;
        void Step(Vector &x, real_t &t, real_t &dt) override;
        void SetTau(real_t tau_) { tau = tau_; }

};



}

#endif // GIAFEM_HPP

