#include "mfem.hpp"
#include "giafem.hpp"
#include <iostream>
#include <fstream>
#include <memory>

using namespace std;
using namespace mfem;
using namespace giafem;


class StrainCoefficient : public VectorCoefficient
{
public:
    enum Type { FULL, DEVIATORIC };

protected:
    const GridFunction *u;
    Type strain_type;
    int dim;

public:
    StrainCoefficient(const GridFunction *u_, Type type = FULL)
        : VectorCoefficient(type == FULL ? 6 : 5), u(u_), strain_type(type)
    {
        dim = u->FESpace()->GetMesh()->Dimension();
        MFEM_ASSERT(dim == 3, "Only implemented for 3D.");
    }

    virtual void Eval(Vector &v, ElementTransformation &T,
                      const IntegrationPoint &ip) override
    {
        DenseMatrix grad_u, symgrad;
        u->GetVectorGradient(T, grad_u); 

        symgrad.SetSize(3, 3);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                symgrad(i, j) = 0.5 * (grad_u(i, j) + grad_u(j, i));
            }
        }

        v.SetSize(GetVDim()); 
        if (strain_type == FULL)
        {
            v(0) = symgrad(0, 0);            
            v(1) = symgrad(1, 0);            
            v(2) = symgrad(2, 0);           
            v(3) = symgrad(1, 1);            
            v(4) = symgrad(2, 1);            
            v(5) = symgrad(2, 2);       
        }
        else if (strain_type == DEVIATORIC)
        {
            real_t trace = symgrad(0, 0) + symgrad(1, 1) + symgrad(2, 2);
            real_t third = trace / 3.0;
            v(0) = symgrad(0, 0) - third;
            v(1) = symgrad(1, 0);
            v(2) = symgrad(2, 0);
            v(3) = symgrad(1, 1) - third; 
            v(4) = symgrad(2, 1);
        }
    }
};

class StrainEnergyCoefficient : public Coefficient
{
private:
    GridFunction &u;
    real_t lambda, mu;
    DenseMatrix grad_u;

public:
    StrainEnergyCoefficient(GridFunction &displacement, double lambda_, double mu_);
    real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override;
    ~StrainEnergyCoefficient() override { }
};


void visualize(ostream &os, Mesh *mesh, GridFunction *deformed_nodes,
               GridFunction *field, const char *field_name = NULL,
               bool init_vis = false);


int main(int argc, char *argv[])
{
    const char *mesh_file = "mesh/examples/beam-tet.mesh";
    real_t tau = 5.0;
    real_t lamb = 15.0;
    real_t mu = 10.0;
    int ode_solver_type = 23;
    real_t t_final = 5.0;
    real_t dt = 0.02;
    //int ref_levels = 3;
    int order = 2;
    bool visualization = true;
    int vis_steps = 1;

    OptionsParser args(argc, argv);
    args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
    args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
    //args.AddOption(&ode_solver_type, "-s", "--ode-solver",
    //              ODESolver::Types.c_str());
    args.Parse();
    if (!args.Good())
    {
       args.PrintUsage(cout);
       return 1;
    }
    args.PrintOptions(cout);


    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();
    //for (int lev = 0; lev < ref_levels; lev++)
    //{
    //   mesh->UniformRefinement();
    //}
    {
        int ref_levels = (int) floor(log(5000./mesh->GetNE())/log(2.)/dim);
        for (int l = 0; l < ref_levels; l++)
        {
            mesh->UniformRefinement();
        }
    }


    H1_FECollection fec(order, dim);
    L2_FECollection fec_L2(order, dim); //
    FiniteElementSpace fes_u(mesh, &fec, 3);
    FiniteElementSpace fes_m(mesh, &fec_L2, 5);
    FiniteElementSpace fes_w(mesh, &fec_L2);
    int u_size = fes_u.GetTrueVSize();
    int m_size = fes_m.GetTrueVSize();
    cout << "Number of u-unknowns: " << u_size << endl;
    GridFunction u(&fes_u); GridFunction m(&fes_m); GridFunction d(&fes_m);
    //GridFunction u_ref(&fes_u);
    //mesh->GetNodes(u_ref);
    GridFunction w(&fes_w);
    u = 0.0; m = 0.0;
    ConstantCoefficient zero(0.0);
    Vector u_vec;
    u.GetTrueDofs(u_vec);
    Vector m_vec;
    m.GetTrueDofs(m_vec);
    ConstantCoefficient lamb_func(lamb), mu_func(mu), tau_func(tau);

    GridFunction *nodes = new GridFunction(&fes_u);
    VectorFunctionCoefficient identity(mesh->SpaceDimension(), [](const Vector &x, Vector &y) { y = x; });
    nodes->ProjectCoefficient(identity);
    mesh->NewNodes(*nodes, false);
    MFEM_VERIFY(mesh->GetNodes(), "Mesh has no nodal coordinates!");
                             
    VeOperator oper(fes_u, fes_m, fes_w, lamb_func, mu_func, tau_func, u_vec, m_vec, u, m, d);
    //unique_ptr<ODESolver> ode_solver = ODESolver::Select(ode_solver_type);
    //ODESolver *ode_solver = new ForwardEulerSolver();
    ODESolver *ode_solver = new BaileySolver();
    ode_solver->Init(oper);
    real_t t = 0.0;
    socketstream vis_w;
    u.SetFromTrueDofs(u_vec); //
    cout << "||u||_L2 = " << u.Norml2() << endl;
    StrainEnergyCoefficient strainEnergyDensity(u, lamb, mu);
    if (visualization)
    {
       char vishost[] = "localhost";
       int  visport   = 19916;
       vis_w.open(vishost, visport);
       w.ProjectCoefficient(strainEnergyDensity);
       cout<<"w_L2: "<<w.ComputeL2Error(zero)<<endl;
       vis_w.precision(8);
       visualize(vis_w, mesh, &u, &w, "Elastic energy density", true);
       cout << "GLVis visualization paused."
            << " Press space (in the GLVis window) to resume it.\n";
    }
    
    bool last_step = false;
    for (int ti = 1; !last_step; ti++)
    {
        cout<<"t = "<<t<<":"<<endl;
        real_t dt_real = min(dt, t_final - t);
        ode_solver->Step(m_vec, t, dt_real);
        last_step = (t >= t_final - 1e-8*dt);

        if (last_step || (ti % vis_steps) == 0)
        {
            if (visualization)
            {
                StrainEnergyCoefficient strainEnergyDensity(u, lamb, mu);
                w.ProjectCoefficient(strainEnergyDensity);
                std::cout << "||u||_L2 = " << u.Norml2() << std::endl;
                cout<<"w_L2: "<<w.Norml2()<<endl;
                visualize(vis_w, mesh, &u, &w, "Elastic energy density");
            }
        }

    }

    //GridFunction *nodes = &u;
    //int owns_nodes = 0;
    //mesh->SwapNodes(nodes, owns_nodes);
    ofstream omesh("ve.mesh");
    omesh.precision(8);
    mesh->Print(omesh);
    //mesh->SwapNodes(nodes, owns_nodes);
    ofstream osol("ve_final.gf");
    osol.precision(8);
    w.Save(osol);

    delete mesh;

    return 0;
}


void visualize(ostream &os, Mesh *mesh, GridFunction *deformed_nodes,
               GridFunction *field, const char *field_name, bool init_vis)
{
    if (!os)
    {
       return;
    }
    ConstantCoefficient zero(0.0);
   

    //GridFunction *displaced_nodes = new GridFunction(*mesh->GetNodes()); 
    //displaced_nodes->Add(1.0, *deformed_nodes);
    mfem::GridFunction *displaced_nodes = new mfem::GridFunction(deformed_nodes->FESpace());
    *displaced_nodes = *mesh->GetNodes();          // base geometry
    *displaced_nodes += *deformed_nodes;  

    int owns_nodes = 0;


    std::cout << "Displacement max = " << deformed_nodes->Max() << std::endl;
    std::cout << "Mesh node max before = " << mesh->GetNodes()->Max() << std::endl;
    
    mesh->SwapNodes(displaced_nodes, owns_nodes);
    
    os << "solution\n" << *mesh << *field;

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
    mesh->SwapNodes(displaced_nodes, owns_nodes);

    delete displaced_nodes;

    os << flush;
}


StrainEnergyCoefficient::StrainEnergyCoefficient(
    GridFunction &displacement, real_t lambda_, real_t mu_)
    : u(displacement), lambda(lambda_), mu(mu_) {}

real_t StrainEnergyCoefficient::Eval(ElementTransformation &T,
                                     const IntegrationPoint &ip)
{
    u.GetVectorGradient(T, grad_u);

    DenseMatrix strain(3, 3);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            strain(i, j) = 0.5 * (grad_u(i, j) + grad_u(j, i));

    real_t trace_eps = strain(0, 0) + strain(1, 1) + strain(2, 2);

    real_t strain_energy_density = 0.5 * lambda * trace_eps * trace_eps;

    real_t eps_inner = 0.0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            eps_inner += strain(i, j) * strain(i, j);

    strain_energy_density += mu * eps_inner;

    return strain_energy_density;
}


VeOperator::VeOperator(FiniteElementSpace &fes_u_, FiniteElementSpace &fes_m_, FiniteElementSpace &fes_w_, 
                       Coefficient &lamb_, Coefficient &mu_, Coefficient &tau_, const Vector &u_vec, const Vector &m_vec, GridFunction &u_gf_, GridFunction &m_gf_, GridFunction &d_gf_)   
    : TimeDependentOperator(fes_m_.GetTrueVSize(), (real_t) 0.0), fes_u(fes_u_), fes_m(fes_m_), fes_w(fes_w_), u_gf(u_gf_), m_gf(m_gf_), d_gf(d_gf_), lamb(lamb_), mu(mu_), tau(tau_), K(NULL), current_dt(0.0), force(3), lamb_gf(&fes_w), mu_gf(&fes_w), tau_gf(&fes_w) 
{
    Array<int> ess_bdr(fes_u.GetMesh()->bdr_attributes.Max());
    ess_bdr = 0;
    ess_bdr[0] = 1;
    fes_u.GetEssentialTrueDofs(ess_bdr, etl);
    
    int dim = fes_u.GetMesh()->Dimension();
    for (int i = 0; i < dim-1; i++)
    {
       force.Set(i, new ConstantCoefficient(0.0));
    }
    {
       Vector pull_force(fes_u.GetMesh()->bdr_attributes.Max());
       pull_force = 0.0;
       pull_force(1) = -1.0e-2;
       force.Set(dim-1, new PWConstCoefficient(pull_force));
    }

    const real_t rel_tol = 1e-6;

    auto b = new LinearForm(&fes_u);
    VectorGridFunctionCoefficient m_func(&m_gf);
    b->AddDomainIntegrator(new ViscoelasticRHSIntegrator(mu, m_func));
    b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(force));
    b->Assemble();
    
    K = new BilinearForm(&fes_u);
    K->AddDomainIntegrator(new ElasticityIntegrator(lamb, mu));
    K->Assemble();
    K->FormSystemMatrix(etl, Kmat);
    
    K_solver.iterative_mode = false;
    K_solver.SetRelTol(rel_tol);
    K_solver.SetAbsTol(0.0);
    K_solver.SetMaxIter(1000);
    K_solver.SetPrintLevel(0);
    K_solver.SetPreconditioner(K_prec);
}

void VeOperator::Mult(const Vector &m_vec, Vector &dm_dt_vec) const
{
    tau_gf.ProjectCoefficient(tau);
    //const Vector &tau_vals = tau_gf;
    Vector tau_vec;
    tau_gf.GetTrueDofs(tau_vec);

    auto b = new LinearForm(&fes_u);
    m_gf.SetFromTrueDofs(m_vec);
    VectorGridFunctionCoefficient m_func(&m_gf);
    b->AddDomainIntegrator(new ViscoelasticRHSIntegrator(mu, m_func));
    b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(force));
    b->Assemble();
    
    Vector X, B;
    K->FormLinearSystem(etl, u_gf, *b, Kmat, X, B);
    K_solver.SetOperator(Kmat);
    K_solver.Mult(B, X);
    K->RecoverFEMSolution(X, *b, u_gf);

    StrainCoefficient d_func(&u_gf, StrainCoefficient::DEVIATORIC);
    d_gf.ProjectCoefficient(d_func);

    Vector d_vec;
    d_gf.GetTrueDofs(d_vec);

    for (int i = 0; i < d_vec.Size(); i++)
    {
        dm_dt_vec[i] = (d_vec[i] - m_vec[i]) / tau_vec[i % tau_vec.Size()];
    }

    delete b;
}

void VeOperator::ImplicitSolve(const real_t dt,
                               const Vector &m, Vector &dm_dt)
{
}

VeOperator::~VeOperator()
{
    //delete T;
    delete K;
}
