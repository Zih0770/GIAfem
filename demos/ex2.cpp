#include <mfem.hpp>
#include <giafem.hpp>
#include <iostream>
#include <cmath>

using namespace std;
using namespace mfem;
using namespace giafem;

real_t rho_func(const Vector &coord);
real_t mu_func(const Vector &coord);
real_t lamb_func(const Vector &coord);
real_t loading_func(const Vector &coord);

class RigidTranslation_test : public VectorCoefficient {
    private:
        int _component;

    public:
        RigidTranslation_test(int dimension, int component)
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

class RigidRotation_test : public VectorCoefficient {
    private:
        int _component;
        Vector _x;

    public:
        RigidRotation_test(int dimension, int component)
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


class BlockRigidBodySolver_test : public Solver {
    private:
        FiniteElementSpace *_fes_u;
        FiniteElementSpace *_fes_phi;
        Array<int> *_block_offsets;
        VectorCoefficient *_dphi0_coeff; //
        std::vector<BlockVector *> _ns; //block?
        Solver *_solver = nullptr;
        mutable BlockVector _b, _x;
        Array<int> _Earth_body_marker;


        real_t Dot(const Vector &x, const Vector &y) const {
            return InnerProduct(x, y);
        }

        real_t Norm(const Vector &x) const {
            return std::sqrt(Dot(x, x));
        }

        void GramSchmidt() {
            for (auto i = 0; i < GetNullDim(); i++) {
                auto &nv1 = *_ns[i];
                //auto &nv1 = *_ns_ext[i];
                for (auto j = 0; j < i; j++) {
                    auto &nv2 = *_ns[j];
                    //auto &nv2 = *_ns_ext[j];
                    //auto product = Dot(nv1.GetBlock(0), nv2.GetBlock(0));
                    auto product = Dot(nv1, nv2);
                    //nv1.GetBlock(0).Add(-product, nv2.GetBlock(0));
                    nv1.Add(-product, nv2);
                }
                //auto norm = Norm(nv1.GetBlock(0));
                auto norm = Norm(nv1);
                //nv1.GetBlock(0) /= norm;
                nv1 /= norm;
            }
        }

        int GetNullDim() const {
            auto vDim = _fes_u->GetVDim();
            return vDim * (vDim + 1) / 2;
        }

        void ProjectOrthogonalToRigidBody(const Vector &x, Vector &y) const {
            y = x;
            for (auto i = 0; i < GetNullDim(); i++) {
                auto &nv = *_ns[i];
                auto product = Dot(y, nv);
                y.Add(-product, nv); //
            }
        }

    public:
        BlockRigidBodySolver_test(FiniteElementSpace *fes_u, FiniteElementSpace *fes_phi, Array<int> *block_offsets, VectorCoefficient *dphi0_coeff) 
            : Solver(0, false), _fes_u{fes_u}, _fes_phi{fes_phi}, _block_offsets{block_offsets}, _dphi0_coeff{dphi0_coeff}, _b(*block_offsets), _x(*block_offsets) {
            auto vDim = _fes_u->GetVDim();
            MFEM_ASSERT(vDim == 2 || vDim == 3, "Dimensions must be two or three");

            width = (*_block_offsets)[2];
            height = width;

            // Set up a temporary gridfunction.
            auto u_gf = GridFunction(_fes_u); 
            auto phi_gf = GridFunction(_fes_phi);

            int num_attr = _fes_phi->GetMesh()->attributes.Size();
            _Earth_body_marker = Array<int>(num_attr);
            _Earth_body_marker = 0;
            _Earth_body_marker[num_attr - 1] = 0;

            ConstantCoefficient zero(0.0);

            // Set the translations.
            for (auto component = 0; component < vDim; component++) {
                auto u_coeff = RigidTranslation_test(vDim, component);
                u_gf.ProjectCoefficient(u_coeff);
                InnerProductCoefficient phi_coeff(u_coeff, *dphi0_coeff);
                //RestrictedCoefficient phi_coeff(phi_coeff_ext, _Earth_body_marker);
                phi_gf.ProjectCoefficient(phi_coeff);
                phi_gf.Neg();
                //phi_gf = 0.0;
                auto nv = new BlockVector(*_block_offsets);
                //nv->SetSize(height);
                *nv = 0.0;

                //u_gf.MakeRef(_fes_u, nv->GetBlock(0), 0);
                //phi_gf.MakeRef(_fes_phi, nv->GetBlock(1), 0);
                u_gf.GetTrueDofs(nv->GetBlock(0));
                phi_gf.GetTrueDofs(nv->GetBlock(1));
                //nv->AddSubVector(tu, _block_offsets[0]);
                //nv->AddSubVector(tphi, _block_offsets[1]);
                _ns.push_back(nv);
            }

            // Set the rotations.
            if (vDim == 2) {
                auto u_coeff = RigidRotation_test(vDim, 2);
                u_gf.ProjectCoefficient(u_coeff);
                InnerProductCoefficient phi_coeff_ext(u_coeff, *dphi0_coeff);
                RestrictedCoefficient phi_coeff(phi_coeff_ext, _Earth_body_marker);
                phi_gf.ProjectCoefficient(phi_coeff);
                phi_gf.Neg();

                auto nv = new BlockVector(*_block_offsets);
                *nv = 0.0;

                u_gf.GetTrueDofs(nv->GetBlock(0));
                phi_gf.GetTrueDofs(nv->GetBlock(1));
                _ns.push_back(nv);
            } else {
                for (auto component = 0; component < vDim; component++) {
                    auto u_coeff = RigidRotation_test(vDim, component);
                    u_gf.ProjectCoefficient(u_coeff);
                    InnerProductCoefficient phi_coeff(u_coeff, *dphi0_coeff);
                    //RestrictedCoefficient phi_coeff(phi_coeff_ext, _Earth_body_marker);
                    phi_gf.ProjectCoefficient(phi_coeff);
                    phi_gf.Neg();
                    //phi_gf = 0.0;

                    auto nv = new BlockVector(*_block_offsets);
                    *nv = 0.0;

                    u_gf.GetTrueDofs(nv->GetBlock(0));
                    phi_gf.GetTrueDofs(nv->GetBlock(1));
                    _ns.push_back(nv);
                }
            }

            GramSchmidt();
        }

        ~BlockRigidBodySolver_test() {
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

            //_solver->Mult(_b, _x);
            //ProjectOrthogonalToRigidBody(_x, x); //
            _solver->Mult(_b, x);
            ProjectOrthogonalToRigidBody(x, x); //
        }
};

int main(int argc, char *argv[])
{
    const char *mesh_file = "mesh/Earth_space.msh";
    const char *elasticity_model = "linear";
    real_t rel_tol = 1e-10;
    int order_u = 1;
    int lMax = 10;
    bool static_cond = false;
    bool pa = false;
    const char *device_config = "cpu";
    bool visualization = false;

    //Parsing
    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
            "Mesh file to use.");
    args.AddOption(&elasticity_model, "-em", "--elasticity-model",
            "Elasticity model to use: linear, neo-hookean, etc.");
    args.AddOption(&rel_tol, "-rt", "--rel-tol",
            "Relative tolerance for linear solving.");
    args.AddOption(&order_u, "-o", "--order",
            "Order (degree) of the finite elements.");
    args.AddOption(&lMax, "-l", "--lMax", "Truncation degree for the DtN map.");
    args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
            "--no-static-condensation", "Enable static condensation.");
    args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
            "--no-partial-assembly", "Enable Partial Assembly.");
    args.AddOption(&device_config, "-d", "--device",
            "Device configuration string, see Device::Configure().");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
            "--no-visualization",
            "Enable or disable GLVis visualization.");
    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(cout);
        return 1;
    }
    args.PrintOptions(cout);

    Device device(device_config);
    device.Print();

    //Mesh
    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();
    //mesh->SetAttributes();
    Array<int> attr_cond = mesh->attributes;
    attr_cond.DeleteLast();

    SubMesh mesh_cond(SubMesh::CreateFromDomain(*mesh, attr_cond));

/*
    Vector r_min, r_max;
    mesh->GetBoundingBox(r_min, r_max);
    r_min.Print();
    r_max.Print();
*/
    //FE Space
    int order_phi = order_u; int order_dphi = order_phi - 1; int order_prop = order_u; //int order_w = 2 * (order_u - 1); 
    H1_FECollection fec_u(order_u, dim), fec_phi(order_phi, dim);
    L2_FECollection fec_dphi(order_dphi, dim), fec_prop(order_prop, dim); //, fec_w(order_w, dim);
    FiniteElementSpace fes_phi(mesh, &fec_phi), fes_phi_cond(&mesh_cond, &fec_phi), fes_dphi(mesh, &fec_dphi), fes_dphi_cond(&mesh_cond, &fec_dphi);
    FiniteElementSpace fes_prop(&mesh_cond, &fec_prop);
    FiniteElementSpace fes_u(&mesh_cond, &fec_u, dim); //, fes_w(&mesh_cond, &fec_w);
    int u_size = fes_u.GetVSize();
    int phi_size = fes_phi.GetVSize();
    cout << "Number of u-unknowns: " << u_size << endl;
    cout << "Number of phi-unknowns: " << phi_size << endl;
    GridFunction u_gf(&fes_u); GridFunction phi_gf(&fes_phi); GridFunction phi_gf_cond(&fes_phi_cond); 
    GridFunction phi0_gf(&fes_phi); GridFunction phi0_gf_cond(&fes_phi_cond); GridFunction dphi0_gf(&fes_dphi); 
    GridFunction dphi0_gf_cond(&fes_dphi_cond); //GridFunction w_gf(&fes_w);
    GridFunction rho_gf(&fes_prop), lamb_gf(&fes_prop), mu_gf(&fes_prop);
    u_gf = 0.0; phi_gf = 0.0; phi_gf_cond = 0.0; phi0_gf = 0.0; phi0_gf_cond = 0.0; dphi0_gf = 0.0; dphi0_gf_cond = 0.0; //w_gf = 0.0;
    FunctionCoefficient rho_coeff(rho_func);
    FunctionCoefficient mu_coeff(mu_func);
    FunctionCoefficient lamb_coeff(lamb_func);
    FunctionCoefficient loading_coeff(loading_func);
    rho_gf.ProjectCoefficient(rho_coeff);
    lamb_gf.ProjectCoefficient(lamb_coeff);
    mu_gf.ProjectCoefficient(mu_coeff);

    Array<int> block_offsets(3);
    block_offsets[0] = 0;
    block_offsets[1] = fes_u.GetVSize();
    block_offsets[2] = fes_phi.GetVSize();
    block_offsets.PartialSum();

    std::cout << "***********************************************************\n";
    std::cout << "dim(u) = " << block_offsets[1] - block_offsets[0] << "\n";
    std::cout << "dim(phi) = " << block_offsets[2] - block_offsets[1] << "\n";
    std::cout << "dim(u+phi) = " << block_offsets.Last() << "\n";
    std::cout << "***********************************************************\n";

    MemoryType mt = device.GetMemoryType();
    BlockVector X(block_offsets, mt), Rhs(block_offsets, mt);
    X = 0.0;
    Rhs = 0.0;

    Array<int> ess_tdof_list;

    Array<int> Earth_body_marker;
    Earth_body_marker = Array<int>(mesh->attributes.Size());
    Earth_body_marker = 1;
    Earth_body_marker[mesh->attributes.Size() - 1] = 0;


    Array<int> bdr_marker;
    int size = mesh->bdr_attributes.Size();
    bdr_marker = Array<int>(size);
    bdr_marker = 0;
    bdr_marker[size - 2] = 1;


    Array<int> bdr_marker_cond;
    int size_cond = mesh_cond.bdr_attributes.Size();
    bdr_marker_cond = Array<int>(size_cond);
    bdr_marker_cond = 0;
    bdr_marker_cond[size - 1] = 1;

    //Compute the equilibrium state
    auto DtN = DirichletToNeumannOperator(&fes_phi, lMax);

    ProductCoefficient rhs_coeff(- 4.0 * M_PI * Constants::G, rho_coeff);
    LinearForm b0(&fes_phi);
    b0.AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
    b0.Assemble();

    BilinearForm a0(&fes_phi);
    auto one = ConstantCoefficient(1.0);
    a0.AddDomainIntegrator(new DiffusionIntegrator(one));
    a0.Assemble();

    OperatorPtr A0;
    Vector B0, Phi0;

    a0.FormLinearSystem(ess_tdof_list, phi0_gf, b0, A0, Phi0, B0);
    cout << "Size of linear system: " << A0->Height() << endl;

    auto S = SumOperator(A0.Ptr(), 1.0, &DtN, 1.0, false, false);

    GSSmoother M((SparseMatrix &)(*A0));
    //DSmoother M((SparseMatrix &)(*A));

    auto solver0 = CGSolver();
    //auto solver = BiCGSTABSolver();
    solver0.SetOperator(S);
    solver0.SetPreconditioner(M);
    solver0.SetRelTol(rel_tol);
    solver0.SetMaxIter(2000);
    solver0.SetPrintLevel(0);
    solver0.Mult(B0, Phi0);

    a0.RecoverFEMSolution(Phi0, b0, phi0_gf);

    DiscreteLinearOperator Grad(&fes_phi_cond, &fes_dphi_cond);
    Grad.AddDomainInterpolator(new GradientInterpolator);
    Grad.Assemble();

    //DiscreteLinearOperator Grad(&fes_phi, &fes_dphi);
    //Grad.AddDomainInterpolator(new GradInterpolator);
    //Grad.Assemble();
    //Grad.Finalize();
    //Grad.Mult(phi0_gf, dphi0_gf);
    //VectorGridFunctionCoefficient dphi0_coeff(&dphi0_gf);
    GridFunctionCoefficient phi0_coeff(&phi0_gf);
    mesh_cond.Transfer(phi0_gf, phi0_gf_cond);
    Grad.Mult(phi0_gf_cond, dphi0_gf_cond);

    GradientGridFunctionCoefficient dphi0_coeff(&phi0_gf);
    VectorRestrictedCoefficient dphi0_coeff_res(dphi0_coeff, Earth_body_marker);
    VectorGridFunctionCoefficient dphi0_cond_coeff(&dphi0_gf_cond);
    //GradientGridFunctionCoefficient dphi0_cond_coeff(&phi0_gf_cond);
    //dphi0_gf_cond.ProjectCoefficient(&dphi0_cond_coeff);
    GradientVectorGridFunctionCoefficient ddphi0_cond_coeff(&dphi0_gf_cond);
    ScalarVectorProductCoefficient dphi0_sig_cond_coeff(loading_coeff, dphi0_cond_coeff); //

    if (!visualization)
    {
        char vishost[] = "localhost";
        int  visport   = 19916;
        socketstream sol_sock(vishost, visport);
        sol_sock.precision(8);
        sol_sock << "solution\n" << *mesh << phi0_gf << flush;
    }


    //Coupled problem
    LinearForm *b1(new LinearForm); //
    b1->Update(&fes_u, Rhs.GetBlock(0), 0);
    b1->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(dphi0_sig_cond_coeff), bdr_marker_cond); //! by luck
    b1->Assemble();
    b1->SyncAliasMemory(Rhs);

    LinearForm *b2(new LinearForm);
    b2->Update(&fes_phi, Rhs.GetBlock(1), 0);
    b2->AddBoundaryIntegrator(new BoundaryLFIntegrator(loading_coeff), bdr_marker);
    b2->Assemble();
    b2->SyncAliasMemory(Rhs);

    BilinearForm *a11(new BilinearForm(&fes_u));
    BilinearForm *a22(new BilinearForm(&fes_phi));
    ExtTrialMixedBilinearForm *a12(new ExtTrialMixedBilinearForm(&fes_phi, &fes_u, &fes_phi_cond, &mesh_cond));
    ExtTestMixedBilinearForm *a21(new ExtTestMixedBilinearForm(&fes_u, &fes_phi, &fes_phi_cond, &mesh_cond));

    ConstantCoefficient c0(1.0 / (4.0 * M_PI * Constants::G));
    ProductCoefficient half_rho_coeff(0.5, rho_coeff);
    ProductCoefficient minus_half_rho_coeff(-0.5, rho_coeff);

    auto a11_integ_0 = ElasticityIntegrator(lamb_coeff, mu_coeff);
    auto a11_integ_1 = AdvectionProjectionIntegrator(half_rho_coeff, dphi0_cond_coeff, ddphi0_cond_coeff);
    //auto a11_integ_1 = mfemElasticity::DomainVectorGradVectorIntegrator(dphi0_cond_coeff, half_rho_coeff);
    auto a11_integ_2 = ProjectionDivergenceIntegrator(minus_half_rho_coeff, dphi0_cond_coeff);
    auto a11_integ_1_t = TransposeIntegrator(&a11_integ_1);
    auto a11_integ_2_t = TransposeIntegrator(&a11_integ_2);
    a11->AddDomainIntegrator(&a11_integ_0);
    a11->AddDomainIntegrator(&a11_integ_1);
    a11->AddDomainIntegrator(&a11_integ_2);
    a11->AddDomainIntegrator(&a11_integ_1_t);
    a11->AddDomainIntegrator(&a11_integ_2_t);
    //if (pa) { a11->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
    //a11->AddDomainIntegrator(new ElasticityIntegrator(lamb_coeff, mu_coeff));
    //a11->AddDomainIntegrator(new ProjectionGradientIntegrator(half_rho_coeff, dphi0_cond_coeff, ddphi0_cond_coeff));
    //a11->AddDomainIntegrator(new AdvectionProjectionIntegrator(half_rho_coeff, dphi0_cond_coeff, ddphi0_cond_coeff));
    //a11->AddDomainIntegrator(new DivergenceVectorIntegrator(minus_half_rho_coeff, dphi0_cond_coeff));
    //a11->AddDomainIntegrator(new ProjectionDivergenceIntegrator(minus_half_rho_coeff, dphi0_cond_coeff));
    a11->Assemble();
    a11->Finalize();

    a22->AddDomainIntegrator(new DiffusionIntegrator(c0));
    a22->Assemble();
    a22->Finalize();

    ConstantCoefficient zero(0.0);
    //a12->AddDomainIntegrator(new mfemElasticity::DomainVectorGradVectorIntegrator(dphi0_cond_coeff, half_rho_coeff));
    a12->AddDomainIntegrator(new GradientIntegrator(rho_coeff));
    //a12->AddDomainIntegrator(new GradientIntegrator(zero));
    //a12->AddDomainIntegrator(new mfemElasticity::DomainVectorGradScalarIntegrator(rho_coeff));
    a12->Assemble(); //by luck
    a12->Finalize();

    /*
    //ProductCoefficient minus_rho_coeff(-1.0, rho_coeff);
    a21->AddDomainIntegrator(new AdvectionIntegrator(rho_coeff));
    //a21->AddDomainIntegrator(new MixedScalarWeakDerivativeIntegrator(minus_rho_coeff), Earth_body_marker);
    //fes_u_ptr = &fes_u;
    a21->Assemble();
    a21->Finalize();
    */
    //FiniteElementSpace fes_u_ext(mesh, &fec_u, dim);

    BlockOperator EGOp(block_offsets);

    SparseMatrix &A11(a11->SpMat());
    SparseMatrix &A12(a12->SpMat());
    //SparseMatrix &A21(a21->SpMat());
    TransposeOperator A21(&A12);
    SparseMatrix &A22_0(a22->SpMat());
    auto A22 = SumOperator(&A22_0, 1.0, &DtN, 1.0 / (4.0 * M_PI * Constants::G), false, false);

    cout<<"Asymmetry tests: A11: "<<A11.IsSymmetric()<<", A12: "<<A12.IsSymmetric()
        <<", A22_0: "<<A22_0.IsSymmetric()<<endl;

    BilinearForm a11_0(&fes_u);
    a11_0.AddDomainIntegrator(&a11_integ_0);
    a11_0.Assemble();
    a11_0.Finalize();
    SparseMatrix &A11_0(a11_0.SpMat());

    //Testing the null space
    for (int i = 0; i < 3; i++)
    {
        auto u_null = GridFunction(&fes_u);
        auto phi_null = GridFunction(&fes_phi);
        auto u_null_coeff = RigidTranslation_test(3, i);
        u_null.ProjectCoefficient(u_null_coeff);
        InnerProductCoefficient phi_null_coeff(u_null_coeff, dphi0_coeff);
        phi_null.ProjectCoefficient(phi_null_coeff);
        //phi_null.Neg();
        //phi_gf = 0.0;
        //auto nv = BlockVector(*_block_offsets);
        //nv = 0.0;
        //u_null.GetTrueDofs(nv->GetBlock(0));
        //phi_null.GetTrueDofs(nv->GetBlock(1));
        
        Vector vec_11(u_size), vec_12(u_size), vec_21(phi_size), vec_22(phi_size);
        Vector diff_1(u_size), diff_2(phi_size);
        A11.Mult(u_null, vec_11); A12.Mult(phi_null, vec_12); 
        A21.Mult(u_null, vec_21); A22.Mult(phi_null, vec_22);
        diff_1 = vec_11; diff_1 -= vec_12;
        diff_2 = vec_21; diff_2 -= vec_22;

        cout<<"For the "<<i<<"th translational mode, the relative error between A11 u and A12 p is "<<diff_1.Norml2()/vec_11.Norml2()
            <<", the relative error between A21 u and A22 p is "<<diff_2.Norml2()/vec_22.Norml2()<<endl;

        Vector vec_00(u_size);
        A11_0.Mult(u_null, vec_00);
        cout<<"For pure elasticity, the l2-norm of A_11 u is "<<vec_00.Norml2()<<endl;
    }

    for (int i = 0; i < 3; i++)
    {
        auto u_null = GridFunction(&fes_u);
        auto phi_null = GridFunction(&fes_phi);
        auto u_null_coeff = RigidRotation_test(3, i);
        u_null.ProjectCoefficient(u_null_coeff);
        InnerProductCoefficient phi_null_coeff(u_null_coeff, dphi0_coeff);
        phi_null.ProjectCoefficient(phi_null_coeff);
        //phi_null.Neg();
        //phi_gf = 0.0;
        //auto nv = BlockVector(*_block_offsets);
        //nv = 0.0;
        //u_null.GetTrueDofs(nv->GetBlock(0));
        //phi_null.GetTrueDofs(nv->GetBlock(1));
        
        Vector vec_11(u_size), vec_12(u_size), vec_21(phi_size), vec_22(phi_size);
        Vector diff_1(u_size), diff_2(phi_size);
        A11.Mult(u_null, vec_11); A12.Mult(phi_null, vec_12); 
        A21.Mult(u_null, vec_21); A22.Mult(phi_null, vec_22);
        diff_1 = vec_11; diff_1 -= vec_12;
        diff_2 = vec_21; diff_2 -= vec_22;

        cout<<"For the "<<i<<"th rotational mode, the relative error between A11 u and A12 p is "<<diff_1.Norml2()/vec_11.Norml2()
            <<", the relative error between A21 u and A22 p is "<<diff_2.Norml2()/vec_22.Norml2()<<endl;

        Vector vec_00(u_size);
        A11_0.Mult(u_null, vec_00);
        cout<<"For pure elasticity, the l2-norm of A_11 u is "<<vec_00.Norml2()<<endl;
    }



    EGOp.SetBlock(0,0, &A11);
    EGOp.SetBlock(0,1, &A12);
    EGOp.SetBlock(1,0, &A21);
    EGOp.SetBlock(1,1, &A22);

    BlockDiagonalPreconditioner EGPrec(block_offsets);
    GSSmoother prec11(A11);
    GSSmoother prec22(A22_0);
    //DSmoother prec11(A11);
    //DSmoother prec22(A22_0);
    EGPrec.SetDiagonalBlock(0, &prec11);
    EGPrec.SetDiagonalBlock(1, &prec22);

    MINRESSolver solver;
    //CGSolver solver;
    solver.SetRelTol(rel_tol);
    solver.SetMaxIter(5000);
    solver.SetOperator(EGOp);
    solver.SetPreconditioner(EGPrec);
    solver.SetPrintLevel(1);
    //solver.Mult(Rhs, X);
    
    //BlockRigidBodySolver rigid_solver(&fes_u, &fes_phi);
    Vector zero_vec(3);
    zero_vec = 0.0;
    VectorConstantCoefficient zero_vec_coeff(zero_vec);
    //BlockRigidBodySolver rigid_solver(&fes_u, &fes_phi, &zero_vec_coeff);
    BlockRigidBodySolver_test rigid_solver(&fes_u, &fes_phi, &block_offsets, &dphi0_coeff);
    //BlockRigidBodySolver_test rigid_solver(&fes_u, &fes_phi, &block_offsets, &zero_vec_coeff);

    //_BlockRigidBodySolver rigid_solver(&fes_u, &fes_phi, &block_offsets, &dphi0_coeff);
    rigid_solver.SetSolver(solver);
    rigid_solver.Mult(Rhs, X);
    //rigid_solver.BlockMult(Rhs, X);

    if (device.IsEnabled()) { X.HostRead(); }


    if (solver.GetConverged())
    {
        std::cout << "MINRES converged in " << solver.GetNumIterations()
            << " iterations with a residual norm of "
            << solver.GetFinalNorm() << ".\n";
    }
    else
    {
        std::cout << "MINRES did not converge in " << solver.GetNumIterations()
            << " iterations. Residual norm is " << solver.GetFinalNorm()
            << ".\n";
    }

    u_gf.MakeRef(&fes_u, X.GetBlock(0), 0);
    phi_gf.MakeRef(&fes_phi, X.GetBlock(1), 0);
    mesh_cond.Transfer(phi_gf, phi_gf_cond);


    if (visualization)
    {
        char vishost[] = "localhost";
        int  visport   = 19916;
        socketstream u_sock(vishost, visport);
        u_sock.precision(8);
        u_sock << "solution\n" << mesh_cond << u_gf << "window_title 'Deformation'" << endl;
        socketstream phi_sock(vishost, visport);
        phi_sock.precision(8);
        phi_sock << "solution\n" << mesh_cond << phi_gf_cond << "window_title 'Gravity Potential'" << endl;

        /*socketstream rho_sock(vishost, visport);
        rho_sock.precision(8);
        rho_sock << "solution\n" << mesh_cond << rho_gf << "window_title 'rho'" << endl;
        socketstream lamb_sock(vishost, visport);
        lamb_sock.precision(8);
        lamb_sock << "solution\n" << mesh_cond << lamb_gf << "window_title 'lamb'" << endl;
        socketstream mu_sock(vishost, visport);
        mu_sock.precision(8);
        mu_sock << "solution\n" << mesh_cond << mu_gf << "window_title 'mu'" << endl;*/
    }


    {
        ofstream mesh_ofs("data/ex2.mesh");
        mesh_ofs.precision(8);
        mesh_cond.Print(mesh_ofs);

        ofstream u_ofs("data/ex2_u.gf");
        u_ofs.precision(8);
        u_gf.Save(u_ofs);

        ofstream phi_ofs("data/ex2_phi.gf");
        phi_ofs.precision(8);
        phi_gf_cond.Save(phi_ofs);
    }



    delete mesh; 
    delete b1;
    delete b2;
    delete a11;
    delete a12;
    delete a21;
    delete a22;

    return 0;
}

real_t rho_func(const Vector &coord)
{
    real_t r = coord.Norml2();
    if (r > Constants::R){
        return 0.0;
    } else{
        real_t r_norm = r / Constants::R;
        //real_t theta = acos(coord[2] / r); // polar angle
        //real_t phi = atan2(coord[1], coord[0]); // azimuthal angle
        real_t rho_surface = 2.6e3; 
        real_t rho_center = 1.3e4;   
        real_t base_rho =  rho_center + (rho_surface - rho_center) * r_norm;
        return base_rho;
    }
}


real_t mu_func(const Vector &coord)
{
    real_t r = coord.Norml2();
    real_t r_norm = r / 6371e3;
    real_t theta = acos(coord[2] / r); // polar angle
    real_t phi = atan2(coord[1], coord[0]); // azimuthal angle
    real_t mu_surface = 70e9;  // Pa
    real_t mu_center = 140e9;    // Pa
    real_t base_mu =  mu_center + (mu_surface - mu_center) * r_norm;
    real_t polar_perturb = 0.015 * (1.0 + cos(2.0 * theta));
    real_t azimuthal_perturb = 0.05 * sin(2.0 * phi);
    //return mu_surface;
    return base_mu * (1.0 + polar_perturb) * (1.0 + azimuthal_perturb);
}

real_t lamb_func(const Vector &coord)
{
    real_t r = coord.Norml2();
    real_t r_norm = r / 6371e3;
    real_t theta = acos(coord[2] / r);
    real_t phi = atan2(coord[1], coord[0]);
    real_t lamb_surface = 100e9;
    real_t lamb_center = 300e9;   
    real_t base_lamb = lamb_center + (lamb_surface - lamb_center) * r_norm;
    real_t polar_perturb = 0.015 * (1.0 + cos(2.0 * theta));
    real_t azimuthal_perturb = 0.05 * sin(2.0 * phi);
    //return lamb_surface;
    return base_lamb * (1.0 + polar_perturb) * (1.0 + azimuthal_perturb);
}

real_t loading_func(const Vector &coord)
{
    real_t factor = 1e-1;
    real_t r = coord.Norml2();
    real_t theta = acos(coord[2] / r);
    real_t phi = atan2(coord[1], coord[0]);
    // Max loading at poles (glaciers): e.g., 10 MPa (~1 km ice)
    const real_t polar_load = 10e6;
    // Equatorial loading (oceans): e.g., 1 MPa (~100 m water depth)
    const real_t equator_load = 1e6;
    real_t base_load = (equator_load + polar_load) / 2.0 + (polar_load - equator_load) / 2.0 * cos(2.0 * theta);
    real_t azimuthal_perturb = 0.2 * sin(2.0 * phi);
    return -base_load * (1.0 + azimuthal_perturb) * factor;
}
