#ifndef GIAFEM_HPP
#define GIAFEM_HPP

#include "mfem.hpp"  // Include MFEM for GridFunction, Mesh, etc.
#include "mfemElasticity.hpp"
#include <cmath>
#include <vector>
#include <string>
#include <utility>
#include <iostream>
//#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>

namespace giafem
{
    using namespace std;
    using namespace mfem;

    //Utilities
    inline void Visualize(ostream &os, ParMesh *mesh, ParGridFunction *deformed_nodes,
                          ParGridFunction *field, const char *field_name, bool init_vis = false, real_t fac = 1000)
    {
        if (!os)
        {
            return;
        }

        //GridFunction *deformation = deformed_nodes;
        GridFunction *displaced_nodes = new GridFunction(deformed_nodes->FESpace());
        *displaced_nodes = *mesh->GetNodes();          // base geometry
        displaced_nodes->Add(fac, *deformed_nodes);

        int owns_nodes = 0;
        mesh->SwapNodes(displaced_nodes, owns_nodes);

        os << "parallel " << mesh->GetNRanks() << " " << mesh->GetMyRank() << "\n";
        os << "solution\n" << *mesh << *field;
        mesh->SwapNodes(displaced_nodes, owns_nodes);

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
    }

//Material Models
enum class ElasticityModel
{
    linear,
    neoHookean
};

enum class RheologyModel
{
    Maxwell,
    Maxwell_nonlinear,
    KelvinVoigt
};

ElasticityModel ParseElasticityModel(const char *str);
RheologyModel ParseRheologyModel(const char *str);


//Operators
class ViscoelasticOperator : public TimeDependentOperator
{
protected:
    MPI_Comm comm;
    int dim;
    ElasticityModel EM;
    RheologyModel RM;
    Coefficient &tau, &lamb, &mu, &loading;
    ParFiniteElementSpace &fes_u, &fes_m, &fes_w, &fes_properties;
    ParGridFunction &u_gf, &m_gf, &d_gf;
    mutable ParGridFunction lamb_gf, mu_gf, tau_gf;
    Array<int> ess_tdof_list;
    ParBilinearForm *K = nullptr;
    HypreParMatrix *hK = nullptr;
    mutable PetscParMatrix *Kmat = nullptr;
    PetscPreconditioner *K_prec = nullptr;
    PetscLinearSolver K_solver;
    mutable mfemElasticity::RigidBodySolver rigid_solver;
    ParMixedBilinearForm B;
    ParDiscreteLinearOperator Dev;
    real_t current_dt;
    real_t rel_tol;
    real_t implicit_scheme_res;
    bool nonlinear = false;

    mutable Vector u_vec, d_vec, tau_vec;
    mutable Vector x_vec, b_vec;

public:
    ViscoelasticOperator(ParFiniteElementSpace &fes_u_, ParFiniteElementSpace &fes_m_, ParFiniteElementSpace &fes_properties_, ParFiniteElementSpace &fes_w_, 
               ParGridFunction &u_gf_, ParGridFunction &m_gf_, ParGridFunction &d_gf_, Coefficient &lamb_, Coefficient &mu_, Coefficient &tau_, Coefficient &loading_,
               const real_t rel_tol_, const real_t implicit_scheme_res_,
               const char *elasticity_model_str = "linear", const char *rheology_model_str = "Maxwell");

    void Mult(const Vector &m_vec, Vector &dm_dt_vec) const override;

    void ImplicitSolve(const real_t dt, const Vector &m_vec, Vector &dm_dt_vec) override;

    void CalcStrainEnergyDensity(ParGridFunction &w_gf) const;

    const ParGridFunction &GetTau() const { return tau_gf; }
    const ParGridFunction &GetLamb() const { return lamb_gf; }
    const ParGridFunction &GetMu()  const { return mu_gf; }

    ~ViscoelasticOperator() override {delete K, hK, Kmat, K_prec;}
};


class ExtTrialMixedBilinearForm : public MixedBilinearForm
{
    protected:
        FiniteElementSpace *trial_fes_cond;
        SubMesh::From from;
        Array<int> parent_element_ids;
        Array<int> *vdof_to_vdof_map = nullptr;
    public:
        ExtTrialMixedBilinearForm(FiniteElementSpace *tr_fes,
                                  FiniteElementSpace *te_fes,
                                  FiniteElementSpace *tr_fes_cond, SubMesh *mesh_cond);

        void Assemble(int skip_zeros = 1);

        ~ExtTrialMixedBilinearForm() { delete vdof_to_vdof_map; }
};


class ExtTestMixedBilinearForm : public MixedBilinearForm
{
    protected:
        FiniteElementSpace *test_fes_cond;
        SubMesh::From from;
        Array<int> parent_element_ids;
        Array<int> *vdof_to_vdof_map = nullptr;
    public:
        ExtTestMixedBilinearForm(FiniteElementSpace *tr_fes,
                                 FiniteElementSpace *te_fes,
                                 FiniteElementSpace *te_fes_cond, SubMesh *mesh_cond);

        void Assemble(int skip_zeros = 1);

        ~ExtTestMixedBilinearForm() { delete vdof_to_vdof_map; }
};







//Interpolators
enum class VecMode {
    FULL,
    REDUCED
};

class GradInterpolator : public DiscreteInterpolator
{
protected:
    int dim, vdim;
    VecMode mode;
    std::vector<std::pair<int, int>> IndexMap;
public:
    GradInterpolator(int dim_ = 3, VecMode mode_ = VecMode::FULL) : dim(dim_), mode(mode_) {
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

class StrainInterpolator : public DiscreteInterpolator
{
protected:
    int dim, vdim;
    VecMode mode;
    std::vector<std::pair<int, int>> IndexMap;

public:
    StrainInterpolator(int dim_ = 3, VecMode mode_ = VecMode::REDUCED) : dim(dim_), mode(mode_) {
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
    VecMode mode;
    std::vector<std::pair<int, int>> IndexMap;
public:
    DevStrainInterpolator(int dim_ = 3, VecMode mode_ = VecMode::REDUCED) : dim(dim_), mode(mode_) {
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


class AdvectionIntegrator : public BilinearFormIntegrator {
private:
    Coefficient* Q = nullptr;
    VectorCoefficient* QV = nullptr;
    MatrixCoefficient* QM = nullptr;

#ifndef MFEM_THREAD_SAFE
    mfem::Vector trial_shape, qv;
    mfem::DenseMatrix test_dshape, pelmats, qm, tm;
#endif

public:
    AdvectionIntegrator(const IntegrationRule* ir = nullptr)
        : BilinearFormIntegrator(ir) {}

    AdvectionIntegrator(Coefficient& q, const IntegrationRule* ir = nullptr)
        : BilinearFormIntegrator(ir), Q{&q} {}

    AdvectionIntegrator(VectorCoefficient& qv, const IntegrationRule* ir = nullptr)
        : BilinearFormIntegrator(ir), QV{&qv} {}

    AdvectionIntegrator(MatrixCoefficient& qm, const IntegrationRule* ir = nullptr)
        : BilinearFormIntegrator(ir), QM{&qm} {}

    static const IntegrationRule& GetRule(const FiniteElement& trial_fe, const FiniteElement& test_fe,
            const ElementTransformation& Trans)
    {
        int order = trial_fe.GetOrder() + Trans.OrderGrad(&test_fe) + Trans.OrderJ();
        return IntRules.Get(trial_fe.GetGeomType(), order);
    }

    void AssembleElementMatrix2(const FiniteElement& trial_fe, const FiniteElement& test_fe,
                                ElementTransformation& Trans, DenseMatrix& elmat) override;

protected:
    const IntegrationRule* GetDefaultIntegrationRule(const FiniteElement& trial_fe, const FiniteElement& test_fe,
                                                     const ElementTransformation& trans) const override 
    { return &GetRule(trial_fe, test_fe, trans); }
};


class ProjectionGradientIntegrator : public BilinearFormIntegrator {
private:
    Coefficient *Q = nullptr;
    VectorCoefficient *QV =nullptr;
    MatrixCoefficient *QM = nullptr;

#ifndef MFEM_THREAD_SAFE
    Vector shape, g;
    DenseMatrix dshape, pelmat_l, pelmat_r, dg;
#endif

public:
    ProjectionGradientIntegrator(const IntegrationRule* ir = nullptr)
        : BilinearFormIntegrator(ir) {}

    ProjectionGradientIntegrator(Coefficient &q, VectorCoefficient &qv, MatrixCoefficient &qm, const IntegrationRule* ir = nullptr)
        : BilinearFormIntegrator(ir), Q{&q}, QV{&qv}, QM(&qm) {}

    void AssembleElementMatrix(const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat) override;
};


class AdvectionProjectionIntegrator : public BilinearFormIntegrator {
private:
    Coefficient *Q = nullptr;
    VectorCoefficient *QV =nullptr;
    MatrixCoefficient *QM = nullptr;

#ifndef MFEM_THREAD_SAFE
    Vector shape, g;
    DenseMatrix dshape, pelmat_l, pelmat_r, dg;
#endif

public:
    AdvectionProjectionIntegrator(const IntegrationRule* ir = nullptr)
        : BilinearFormIntegrator(ir) {}

    AdvectionProjectionIntegrator(Coefficient &q, VectorCoefficient &qv, MatrixCoefficient &qm, const IntegrationRule* ir = nullptr)
        : BilinearFormIntegrator(ir), Q{&q}, QV{&qv}, QM(&qm) {}

    void AssembleElementMatrix(const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat) override;
};


class DivergenceVectorIntegrator : public BilinearFormIntegrator {
private:
    Coefficient *Q = nullptr;
    VectorCoefficient *QV =nullptr;

#ifndef MFEM_THREAD_SAFE
    mfem::Vector shape, g;
    mfem::DenseMatrix dshape, pelmat;
#endif

public:
    DivergenceVectorIntegrator(const IntegrationRule* ir = nullptr)
        : BilinearFormIntegrator(ir) {}

    DivergenceVectorIntegrator(Coefficient &q, VectorCoefficient &qv, const IntegrationRule* ir = nullptr)
        : BilinearFormIntegrator(ir), Q{&q}, QV{&qv} {}

    void AssembleElementMatrix(const FiniteElement& el, ElementTransformation& Trans, DenseMatrix& elmat) override;
};


class ProjectionDivergenceIntegrator : public BilinearFormIntegrator {
private:
    Coefficient *Q = nullptr;
    VectorCoefficient *QV =nullptr;

#ifndef MFEM_THREAD_SAFE
    mfem::Vector shape, g;
    mfem::DenseMatrix dshape, pelmat;
#endif

public:
    ProjectionDivergenceIntegrator(const IntegrationRule* ir = nullptr)
        : BilinearFormIntegrator(ir) {}

    ProjectionDivergenceIntegrator(Coefficient &q, VectorCoefficient &qv, const IntegrationRule* ir = nullptr)
        : BilinearFormIntegrator(ir), Q{&q}, QV{&qv} {}

    void AssembleElementMatrix(const FiniteElement &el, ElementTransformation& Trans, DenseMatrix& elmat) override;
};


/*
class ProjDivIntegrator : public BilinearFormIntegrator {
private:
    Coefficient *Q = nullptr;
    VectorCoefficient *QV =nullptr;

#ifndef MFEM_THREAD_SAFE
    mfem::Vector trial_shape, g;
    mfem::DenseMatrix test_dshape, pelmats;
#endif

public:
    ProjDivIntegrator(const IntegrationRule* ir = nullptr)
        : BilinearFormIntegrator(ir) {}

    ProjDivIntegrator(Coefficient &q, VectorCoefficient &qv, const IntegrationRule* ir = nullptr)
        : BilinearFormIntegrator(ir), Q{&q}, QV{&qv} {}

    static const IntegrationRule& GetRule(const FiniteElement& trial_fe, const FiniteElement& test_fe,
                                          const ElementTransformation& Trans);

    void AssembleElementMatrix2(const FiniteElement& trial_fe, const FiniteElement& test_fe,
                                ElementTransformation& Trans, DenseMatrix& elmat) override;

protected:
    const IntegrationRule* GetDefaultIntegrationRule(const FiniteElement& trial_fe, const FiniteElement& test_fe,
                                                     const ElementTransformation& trans) const override 
    { return &GetRule(trial_fe, test_fe, trans); }
};
*/

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
    const GridFunction &u_gf;
    Coefficient &lambda, &mu;
    DenseMatrix grad_u;

public:
    StrainEnergyCoefficient(const GridFunction &displacement, Coefficient &lambda_, Coefficient &mu_)
        : u_gf(displacement), lambda(lambda_), mu(mu_) { }

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


class SphericalHarmonicCoefficient : public mfem::Coefficient {
private:
    int _l;
    int _m;
    bool _solid;
    const mfem::real_t sqrt2 = sqrt(2);
    mutable mfem::Vector _x;

public:
    SphericalHarmonicCoefficient(int l, int m, bool solid = false)
        : _l{l}, _m{m}, _solid{solid} {}

    mfem::real_t Eval(mfem::ElementTransformation &T,
            const mfem::IntegrationPoint &ip) {
        T.Transform(ip, _x);
        mfem::real_t theta = atan2(sqrt(_x[0] * _x[0] + _x[1] * _x[1]), _x[2]);
        mfem::real_t phi = atan2(_x[1], _x[0]);

        mfem::real_t rfac = _m == 0 ? 1 : sqrt2;
        if (_solid) {
            mfem::real_t r = _x.Norml2();
            rfac *= std::pow(r, _l);
        }

        if (_m < 0) {
            return rfac * boost::math::spherical_harmonic_r(_l, -_m, theta, phi);
        } else if (_m == 0) {
            return rfac * boost::math::spherical_harmonic_r(_l, 0, theta, phi);
        } else {
            return rfac * boost::math::spherical_harmonic_i(_l, _m, theta, phi);
        }
    }
};


class GradientVectorGridFunctionCoefficient : public MatrixCoefficient
{
protected:
   const GridFunction *GridFunc;

public:
   GradientVectorGridFunctionCoefficient(const GridFunction *gf);

   void SetGridFunction(const GridFunction *gf);

   const GridFunction *GetGridFunction() const { return GridFunc; }

   virtual void Eval(DenseMatrix &M,
                     ElementTransformation &T,
                     const IntegrationPoint &ip) override;
/*
   virtual void Eval(DenseTensor &T,
                     ElementTransformation &Tr,
                     const IntegrationRule &ir) override;
*/
   virtual ~GradientVectorGridFunctionCoefficient() {}
};


struct Constants {
    public:
        static constexpr real_t G = 6.6743e-11;
        static constexpr real_t c = 2.99792458e8;
        static constexpr real_t h = 6.62607015e-34;
        static constexpr real_t _h = 1.054571817e-34; 
        static constexpr real_t kB = 1.380649e-23;
        static constexpr real_t NA = 6.02214076e23;
        static constexpr real_t e = 1.602176634e-19;
        static constexpr real_t epi0 = 8.854187817e-12;
        static constexpr real_t mu0 = 1.25663706212e-6;


        static constexpr real_t R = 6371e3;
        static constexpr real_t R_ext = 8e6;
};


class Nondimensionalisation {
private:
    real_t L;   // Length scale [m]
    real_t T;   // Time scale [s]
    real_t RHO; // Density scale [kg/m^3]

public:
    Nondimensionalisation(real_t length_scale, real_t time_scale, real_t density_scale)
        : L(length_scale), T(time_scale), RHO(density_scale) {}

    // Accessors
    real_t Length() const { return L; }
    real_t Time() const { return T; }
    real_t Density() const { return RHO; }

    // Derived scales
    real_t Velocity() const { return L / T; }
    real_t Acceleration() const { return L / (T*T); }
    real_t Pressure() const { return RHO * L*L / (T*T); } // [Pa]
    real_t Gravity() const { return L / (T*T); }
    real_t Potential() const { return L*L / (T*T); }

    // Scaling functions for scalars
    real_t ScaleLength(real_t x) const { return x / L; }
    real_t UnscaleLength(real_t x_nd) const { return x_nd * L; }

    real_t ScaleDensity(real_t rho) const { return rho / RHO; }
    real_t UnscaleDensity(real_t rho_nd) const { return rho_nd * RHO; }

    real_t ScaleGravityPotential(real_t phi) const { return phi / Potential(); }
    real_t UnscaleGravityPotential(real_t phi_nd) const { return phi_nd * Potential(); }

    real_t ScaleStress(real_t sigma) const { return sigma / Pressure(); }
    real_t UnscaleStress(real_t sigma_nd) const { return sigma_nd * Pressure(); }

    // Scaling for GridFunction fields
    void UnscaleGravityPotential(GridFunction &phi_gf) const { phi_gf *= Potential(); }
    void UnscaleDisplacement(GridFunction &u_gf) const { u_gf *= L; }
    void UnscaleStress(GridFunction &sigma_gf) const { sigma_gf *= Pressure(); }

    // Create a scaled density coefficient from a dimensional one
    Coefficient *MakeScaledDensityCoefficient(Coefficient &rho_coeff) const {
        return new ProductCoefficient(1.0 / RHO, rho_coeff);
    }

    void Print() const {
        cout << "Scaling parameters:\n";
        cout << "  Length scale: " << L << " m\n";
        cout << "  Time scale: " << T << " s\n";
        cout << "  Density scale: " << RHO << " kg/m^3\n";
        cout << "  Gravity potential scale: " << Potential() << " m^2/s^2\n";
    }
};




//Solver-related
inline SparseMatrix* DenseToSparse(const DenseMatrix &dense_mat, real_t tol=1e-12)
{
    int m = dense_mat.Height();
    int n = dense_mat.Width();

    std::vector<int> I(m+1), J;
    std::vector<real_t> data;

    I[0] = 0;
    for (int row = 0; row < m; ++row)
    {
        for (int col = 0; col < n; ++col)
        {
            real_t val = dense_mat(row, col);
            if (std::abs(val) > tol) // consider as nonzero
            {
                J.push_back(col);
                data.push_back(val);
            }
        }
        I[row+1] = J.size();
    }

    return new SparseMatrix(I.data(), J.data(), data.data(), m, n);
}


class ParDirichletToNeumannOperator : public Operator {
private:
    ParFiniteElementSpace *fes;
    int lMax;
    real_t radius;
    Array<int> marker;
    vector<HypreParVector *> u;

    void SetMarker() {
        auto *pmesh = fes->GetParMesh();
        auto size = pmesh->bdr_attributes.Size(); //!
        marker = Array<int>(size);
        marker = 0;
        marker[size - 1] = 1;
    }

    void GetBoundingRadius() {
        radius = 0;
        auto *pmesh = fes->GetParMesh();
        auto x = Vector();
        for (int i = 0; i < fes->GetNBE(); i++) {
            const int bdr_attr = pmesh->GetBdrAttribute(i);
            if (marker[bdr_attr - 1] == 1) {
                const auto *el = fes->GetBE(i);
                auto *T = fes->GetBdrElementTransformation(i);
                const auto ir = el->GetNodes();
                for (auto j = 0; j < ir.GetNPoints(); j++) {
                    const IntegrationPoint &ip = ir.IntPoint(j);
                    T->SetIntPoint(&ip);
                    T->Transform(ip, x);
                    auto r = x.Norml2();
                    if (r > radius) radius = r;
                }
            }
        }
    }

public:
    ParDirichletToNeumannOperator(ParFiniteElementSpace *fes_, int lMax_, real_t radius_)
        : Operator(fes_->GetTrueVSize()), fes{fes_}, lMax{lMax_}, radius(radius_) {
            SetMarker();
            //GetBoundingRadius();
            //cout<<"Bounding radius: "<<radius<<endl;
            for (auto l = 0; l <= lMax; l++) {
                for (auto m = -l; m <= l; m++) {
                    auto f = SphericalHarmonicCoefficient(l, m);
                    auto b = ParLinearForm(fes);
                    b.AddBoundaryIntegrator(new BoundaryLFIntegrator(f), marker);
                    b.Assemble();
                    auto *tv = new HypreParVector();
                    tv = b.ParallelAssemble();
                    //b.ParallelAssemble(*tv);
                    u.push_back(tv);
                }
            }
        }

    ~ParDirichletToNeumannOperator() {
        for (auto i = 0; i < u.size(); i++) {
            delete u[i];
        }
    }

    void Mult(const Vector &x, Vector &y) const override {
        y.SetSize(x.Size());
        y = 0.0;
        auto ir3 = 1.0 / pow(radius, 3);
        auto i = 0;
        for (auto l = 0; l <= lMax; l++) {
            for (auto m = -l; m <= l; m++) {
                auto &_u = *u[i++];
                //auto product = (l + 1) * ir3 * (_u * x); //!
                auto product = (l + 1) * ir3 * InnerProduct(fes->GetComm(), _u, x);
                y.Add(product, _u);
            }
        }
    }

    void MultTranspose(const Vector &x, Vector &y) const override { Mult(x, y); }
};


class RigidTranslation : public VectorCoefficient {
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

class RigidRotation : public VectorCoefficient {
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
            //_solver->iterative_mode = iterative_mode;
            _solver->Mult(_b, x);
            ProjectOrthogonalToRigidBody(x, x);
        }
};


class BlockRigidBodySolver : public Solver {
    private:
        FiniteElementSpace *_fes_u;
        FiniteElementSpace *_fes_phi;
        Array<int> _block_offsets;
        std::vector<Vector *> _ns; //block?
        Solver *_solver = nullptr;
        mutable Vector _b;
        const bool _parallel; //

#ifdef MFEM_USE_MPI
        ParFiniteElementSpace *_pfes_u;
        ParFiniteElementSpace *_pfes_phi;
        MPI_Comm _comm;
#endif

        real_t Dot(const Vector &x, const Vector &y) const {
#ifdef MFEM_USE_MPI
            return _parallel ? InnerProduct(_comm, x, y) : InnerProduct(x, y);
#else
            return InnerProduct(x, y);
#endif
        }

        /*real_t BlockDot(const Vector &x, const Vector &y) const {
#ifdef MFEM_USE_MPI
            if (!_parallel)
            {
                real_t alpha_u = 1.0;
                //real_t alpha_phi = 1.0 / (9.8 * 9.8);  // example scaling
                real_t alpha_phi = 1.0;
                Vector x_u(const_cast<Vector&>(x), _block_offsets[0], _fes_u->GetTrueVSize()); 
                Vector y_u(const_cast<Vector&>(y), _block_offsets[0], _fes_u->GetTrueVSize());
                Vector x_phi(const_cast<Vector&>(x), _block_offsets[1], _fes_phi->GetTrueVSize());
                Vector y_phi(const_cast<Vector&>(y), _block_offsets[1], _fes_phi->GetTrueVSize());

                return alpha_u * InnerProduct(x_u, y_u) + alpha_phi * InnerProduct(x_phi, y_phi);
            }
            else
            {
                real_t alpha_u = 1.0;
                //real_t alpha_phi = 1.0 / (9.8 * 9.8);  // example scaling
                real_t alpha_phi = 1.0;
                Vector x_u(const_cast<Vector&>(x), _block_offsets[0], _fes_u->GetTrueVSize()); 
                Vector y_u(const_cast<Vector&>(y), _block_offsets[0], _fes_u->GetTrueVSize());
                Vector x_phi(const_cast<Vector&>(x), _block_offsets[1], _fes_phi->GetTrueVSize());
                Vector y_phi(const_cast<Vector&>(y), _block_offsets[1], _fes_phi->GetTrueVSize());

                return alpha_u * InnerProduct(_comm, x_u, y_u) + alpha_phi * InnerProduct(_comm, x_phi, y_phi);

            }
#else
            real_t alpha_u = 1.0;
            //real_t alpha_phi = 1.0 / (9.8 * 9.8);  // example scaling
            real_t alpha_phi = 1.0;
            Vector x_u(const_cast<Vector&>(x), _block_offsets[0], _fes_u->GetTrueVSize()); 
            Vector y_u(const_cast<Vector&>(y), _block_offsets[0], _fes_u->GetTrueVSize());
            Vector x_phi(const_cast<Vector&>(x), _block_offsets[1], _fes_phi->GetTrueVSize());
            Vector y_phi(const_cast<Vector&>(y), _block_offsets[1], _fes_phi->GetTrueVSize());

            return alpha_u * InnerProduct(x_u, y_u) + alpha_phi * InnerProduct(x_phi, y_phi);
#endif
        }*/

        real_t Norm(const Vector &x) const {
            return std::sqrt(Dot(x, x));
        }

        /*real_t BlockNorm(const Vector &x) const {
            return std::sqrt(BlockDot(x, x));
        }*/

        void GramSchmidt() {
            for (auto i = 0; i < GetNullDim(); i++) {
                auto &nv1 = *_ns[i];
                for (auto j = 0; j < i; j++) {
                    auto &nv2 = *_ns[j];
                    auto product = Dot(nv1, nv2);
                    nv1.Add(-product, nv2);
                }
                auto norm = Norm(nv1);
                nv1 /= norm;
            }
        }

        int GetNullDim() const {
            auto vDim = _fes_u->GetVDim();
            //return vDim * (vDim + 1) / 2 + 1; //
            return vDim * (vDim + 1) / 2;
        }

        void ProjectOrthogonalToRigidBody(const Vector &x, Vector &y) const {
            y = x;
            for (auto i = 0; i < GetNullDim(); i++) {
                auto &nv = *_ns[i];
                auto product = Dot(y, nv);
                y.Add(-product, nv);
            }
        }

    public:
        BlockRigidBodySolver(FiniteElementSpace *fes_u, FiniteElementSpace *fes_phi, VectorCoefficient *dphi0_coeff) 
            : Solver(0, false), _fes_u{fes_u}, _fes_phi{fes_phi}, _parallel{false} {
            auto vDim = _fes_u->GetVDim();
            MFEM_ASSERT(vDim == 2 || vDim == 3, "Dimensions must be two or three");

            int size_u = _fes_u->GetTrueVSize();
            int size_phi = _fes_phi->GetTrueVSize();

            _block_offsets.SetSize(3);
            _block_offsets[0] = 0;
            _block_offsets[1] = size_u;
            _block_offsets[2] = size_u + size_phi;

            height = width = _block_offsets[2];
            _b.SetSize(height);

            // Set up a temporary gridfunction.
            auto u_gf = GridFunction(_fes_u);
            auto phi_gf = GridFunction(_fes_phi);

            // Set the translations.
            for (auto component = 0; component < vDim; component++) {
                auto u_coeff = RigidTranslation(vDim, component);
                u_gf.ProjectCoefficient(u_coeff);
                InnerProductCoefficient phi_coeff(u_coeff, *dphi0_coeff);
                phi_gf.ProjectCoefficient(phi_coeff);
                phi_gf.Neg();

                Vector *nv = new Vector(height);
                //nv->SetSize(height);
                *nv = 0.0;

                Vector tu;
                u_gf.GetTrueDofs(tu);
                Vector tphi;
                phi_gf.GetTrueDofs(tphi);


                nv->AddSubVector(tu, _block_offsets[0]);
                nv->AddSubVector(tphi, _block_offsets[1]);
                _ns.push_back(nv);
            }

            // Set the rotations.
            if (vDim == 2) {
                auto u_coeff = RigidRotation(vDim, 2);
                u_gf.ProjectCoefficient(u_coeff);
                InnerProductCoefficient phi_coeff(u_coeff, *dphi0_coeff);
                phi_gf.ProjectCoefficient(phi_coeff);
                phi_gf.Neg();

                Vector *nv = new Vector(height);
                *nv = 0.0;

                Vector tu;
                u_gf.GetTrueDofs(tu);
                Vector tphi;
                phi_gf.GetTrueDofs(tphi);

                nv->AddSubVector(tu, _block_offsets[0]);
                nv->AddSubVector(tphi, _block_offsets[1]);
                _ns.push_back(nv);
            } else {
                for (auto component = 0; component < vDim; component++) {
                    auto u_coeff = RigidRotation(vDim, component);
                    u_gf.ProjectCoefficient(u_coeff);
                    InnerProductCoefficient phi_coeff(u_coeff, *dphi0_coeff);
                    phi_gf.ProjectCoefficient(phi_coeff);
                    phi_gf.Neg();

                    Vector *nv = new Vector(height);
                    *nv = 0.0;

                    Vector tu;
                    u_gf.GetTrueDofs(tu);
                    Vector tphi;
                    phi_gf.GetTrueDofs(tphi);

                    nv->AddSubVector(tu, _block_offsets[0]);
                    nv->AddSubVector(tphi, _block_offsets[1]);
                    _ns.push_back(nv);
                }
            }

            // Constant mode in phi
            /*phi_gf = 0.0;
            Vector tphi;
            phi_gf.GetTrueDofs(tphi);
            tphi = 1.0 / std::sqrt(tphi.Size()); 
            Vector *nv = new Vector(height);
            *nv = 0.0;
            nv->AddSubVector(tphi, _block_offsets[1]);
            _ns.push_back(nv);*/

            GramSchmidt();
        }

#ifdef MFEM_USE_MPI
        BlockRigidBodySolver(MPI_Comm comm, ParFiniteElementSpace *fes_u, ParFiniteElementSpace *fes_phi, VectorCoefficient *dphi0_coeff) : Solver(0, false),
            //_fes_u{fes_u},
            _pfes_u{fes_u},
            //_fes_phi{fes_phi},
            _pfes_phi{fes_phi},
            _comm{comm},
            _parallel{true} {
                auto vDim = _pfes_u->GetVDim();
                MFEM_ASSERT(vDim == 2 || vDim == 3, "Dimensions must be two or three");

                int size_u = _pfes_u->GetTrueVSize();
                int size_phi = _pfes_phi->GetTrueVSize();

                _block_offsets.SetSize(3);
                _block_offsets[0] = 0;
                _block_offsets[1] = size_u;
                _block_offsets[2] = size_u + size_phi;

                height = width = _block_offsets[2];
                _b.SetSize(height);

                // Set up a temporary ParGridfunction.
                auto u_gf = ParGridFunction(_pfes_u);
                auto phi_gf = ParGridFunction(_pfes_phi);

                // Set the translations.
                for (auto component = 0; component < vDim; component++) {
                    auto u_coeff = RigidTranslation(vDim, component);
                    u_gf.ProjectCoefficient(u_coeff);
                    InnerProductCoefficient phi_coeff(u_coeff, *dphi0_coeff);
                    phi_gf.ProjectCoefficient(phi_coeff);
                    phi_gf.Neg();

                    Vector *nv = new Vector(height);
                    //nv->SetSize(height);
                    *nv = 0.0;

                    Vector tu;
                    u_gf.GetTrueDofs(tu);
                    Vector tphi;
                    phi_gf.GetTrueDofs(tphi);


                    nv->AddSubVector(tu, _block_offsets[0]);
                    nv->AddSubVector(tphi, _block_offsets[1]);
                    _ns.push_back(nv);
                }

                // Set the rotations.
                if (vDim == 2) {
                    auto u_coeff = RigidRotation(vDim, 2);
                    u_gf.ProjectCoefficient(u_coeff);
                    InnerProductCoefficient phi_coeff(u_coeff, *dphi0_coeff);
                    phi_gf.ProjectCoefficient(phi_coeff);
                    phi_gf.Neg();

                    Vector *nv = new Vector(height);
                    *nv = 0.0;

                    Vector tu;
                    u_gf.GetTrueDofs(tu);
                    Vector tphi;
                    phi_gf.GetTrueDofs(tphi);


                    nv->AddSubVector(tu, _block_offsets[0]);
                    nv->AddSubVector(tphi, _block_offsets[1]);
                    _ns.push_back(nv);
                } else {
                    for (auto component = 0; component < vDim; component++) {
                        auto u_coeff = RigidRotation(vDim, component);
                        u_gf.ProjectCoefficient(u_coeff);
                        InnerProductCoefficient phi_coeff(u_coeff, *dphi0_coeff);
                        phi_gf.ProjectCoefficient(phi_coeff);
                        phi_gf.Neg();

                        Vector *nv = new Vector(height);
                        *nv = 0.0;

                        Vector tu;
                        u_gf.GetTrueDofs(tu);
                        Vector tphi;
                        phi_gf.GetTrueDofs(tphi);


                        nv->AddSubVector(tu, _block_offsets[0]);
                        nv->AddSubVector(tphi, _block_offsets[1]);
                        _ns.push_back(nv);
                    }
                }

                // Constant mode in phi
                //ConstantCoefficient one(1.0);
                //phi_gf.ProjectCoefficient(one);
                /*phi_gf = 0.0;
                Vector tphi;
                phi_gf.GetTrueDofs(tphi);
                tphi = 1.0 / std::sqrt(tphi.Size()); 
                Vector *nv = new Vector(height);
                *nv = 0.0;
                nv->AddSubVector(tphi, _block_offsets[1]);
                _ns.push_back(nv);*/

                GramSchmidt();
            }
#endif

        ~BlockRigidBodySolver() {
            for (auto i = 0; i < GetNullDim(); i++) {
                delete _ns[i];
            }
        }

        void SetSolver(Solver &solver) {
            _solver = &solver;
            height = _solver->Height();
            width = _solver->Width();
            MFEM_VERIFY(height == width, "Solver must be a square operator");
        }

        void SetOperator(const Operator &op) { //override
            MFEM_VERIFY(_solver, "Solver hasn't been set, call SetSolver() first.");
            _solver->SetOperator(op);
            height = _solver->Height();
            width = _solver->Width();
            MFEM_VERIFY(height == width, "Solver must be a square operator");
        }

        void Mult(const Vector &b, Vector &x) const { //override
            ProjectOrthogonalToRigidBody(b, _b);
            //_solver->iterative_mode = iterative_mode;
            Vector _x(x.Size());
            _solver->Mult(_b, _x);
            ProjectOrthogonalToRigidBody(const_cast<Vector&>(_x), x); //
            //_solver->Mult(_b, x);
            //ProjectOrthogonalToRigidBody(x, x); //
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


class DirichletToNeumannOperator : public Operator {
 private:
  FiniteElementSpace *_fes;
  int _lMax;
  real_t _radius;
  Array<int> _marker;
  vector<Vector *> _u;

  int NumberOfSphericalHarmonicCoefficients() const {
    return pow(_lMax + 1, 2);
  }

  void SetMarker() {
    auto *mesh = _fes->GetMesh();
    auto size = mesh->bdr_attributes.Size();
    _marker = Array<int>(size);
    _marker = 0;
    _marker[size - 1] = 1;
  }

  void GetBoundingRadius() {
    _radius = 0;
    auto *mesh = _fes->GetMesh();
    auto x = Vector();
    for (int i = 0; i < _fes->GetNBE(); i++) {
      const int bdr_attr = mesh->GetBdrAttribute(i);
      if (_marker[bdr_attr - 1] == 1) {
        const auto *el = _fes->GetBE(i);
        auto *T = _fes->GetBdrElementTransformation(i);
        const auto ir = el->GetNodes();
        for (auto j = 0; j < ir.GetNPoints(); j++) {
          const IntegrationPoint &ip = ir.IntPoint(j);
          T->SetIntPoint(&ip);
          T->Transform(ip, x);
          auto r = x.Norml2();
          if (r > _radius) _radius = r;
        }
      }
    }
  }

 public:
  DirichletToNeumannOperator(FiniteElementSpace *fes, int lMax)
      : Operator(fes->GetTrueVSize()), _fes{fes}, _lMax{lMax} {
    SetMarker();
    GetBoundingRadius();
    //_radius = Constants::R_ext;
    for (auto l = 0; l <= _lMax; l++) {
      for (auto m = -l; m <= l; m++) {
        auto f = SphericalHarmonicCoefficient(l, m);
        auto b = LinearForm(_fes);
        b.AddBoundaryIntegrator(new BoundaryLFIntegrator(f), _marker);
        b.Assemble();
        auto *tv = new Vector();
        *tv = b;
        _u.push_back(tv);
      }
    }
  }

  ~DirichletToNeumannOperator() {
    for (auto i = 0; i < _u.size(); i++) {
      delete _u[i];
    }
  }

  void Mult(const Vector &x, Vector &y) const override {
    y.SetSize(x.Size());
    y = 0.0;
    auto ir3 = 1.0 / pow(_radius, 3);
    auto i = 0;
    for (auto l = 0; l <= _lMax; l++) {
      for (auto m = -l; m <= l; m++) {
        auto &u = *_u[i++];
        auto product = (l + 1) * ir3 * (u * x);
        y.Add(product, u);
      }
    }
  }

  void MultTranspose(const Vector &x, Vector &y) const override { Mult(x, y); }
};



}

#endif // GIAFEM_HPP

