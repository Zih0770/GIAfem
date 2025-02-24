#include "giafem.hpp"
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace giafem;


class VEOperator : public TimeDependentOperator
{
protected:
   FiniteElementSpace &fespace;

   BilinearForm a;
   real_t lamb;
   real_t mu;
   real_t tau;

   CGSolver u_solver; // Krylov solver for inverting the mass matrix M
   DSmoother prec;  // Preconditioner for the mass matrix M

public:
   VEOperator(FiniteElementSpace &f, Array<int> &ess_bdr,
                        real_t lamb, real_t mu, real_t tau);

   void Mult(const Vector &x, Vector &dx_dt) const override;
   void ImplicitSolve(const real_t dt, const Vector &x, Vector &k) override;

   ~VEOperator() override;
};


int main(int argc, char *argv[])
{
    //Configuration
    int order = 2;
    double tf = 10.0;
    double dt = 0.1;
    double tau = 1.0; //relaxation time
    int ode_solver_type = 23;
    bool visualization = 1;
    int vis_steps = 1;

    OptionsParser args(argc, argv);
    args.AddOption(&order, "-o", "--order", "Finite element order (polynomial degree).");
    args.AddOption(&t_final, "-tf", "--t-final", "Final simulation time.");
    args.AddOption(&dt, "-dt", "--time-step", "Time step size.");
    args.AddOption(&ode_solver_type, "-s", "--ode-solver", ODESolver::Types.c_str());
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis", "--no-visualization", "Enable or disable GLVis visualization.");
    args.AddOption(&vis_steps, "-vs", "--visualization-steps", "Visualize every n-th timestep.");
    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(cout);
        return 1;
    }
    args.PrintOptions(cout);


    //Mesh
    int nx = 64, ny = 16, nz = 16;
    double lx = 10.0, ly = 2.0, lz = 2.0;
    Mesh *mesh = new Mesh(Mesh::MakeCartesian3D(nx, ny, nz, Element::TETRAHEDRON, lx, ly, lz));
    int dim = mesh->Dimension();

    for (int i = 0; i < mesh->GetNE(); i++)
    {
        Element *el = mesh->GetElement(i);
        Array<int> vertices;
        el->GetVertices(vertices);
        double x_centroid = 0.0;
        for (int j = 0; j < vertices.Size(); j++)
        {
            x_centroid += mesh->GetVertex(vertices[j])[0];
        }
        x_centroid /= vertices.Size();
        el->SetAttribute(x_centroid < lx / 2.0 ? 1 : 2); // Divide at x = lx / 2.
    i}

    for (int i = 0; i < mesh->GetNBE(); i++)
    {
        Element *bdr_el = mesh->GetBdrElement(i);
        Array<int> vertices;
        bdr_el->GetVertices(verticies);
        bool is_x0 = true, is_xL = true;
        for (int j = 0; j < vertices.Size(); j++)
        {
            const double *v = mesh->GetVertex(vertices[j]);
            if (fabs(v[0]) > 1e-8) is_x0 = false;   // Not on x = 0
            if (fabs(v[0] - lx) > 1e-8) is_xL = false; // Not on x = lx
        }
        if (is_x0) bdr_el->SetAttribute(1);
        else if (is_xL) bdr_el->SetAttribute(2);
        else bdr_el->SetAttribute(3);
    }
    mesh->SetAttributes();

    {
        int ref_levels = (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
        for (int l = 0; l < ref_levels; l++)
        {
           mesh->UniformRefinement();
        }
    }



    //FE spaces
    FiniteElementCollection *fec = new H1_FECollection(order, dim);
    FiniteElementSpace *fes = new FiniteElementSpace(mesh, fec, dim);
    cout << "Number of finite element unknowns: " << fespace->GetTrueVSize() << endl;
    GridFunction x(fes);
    x = 0.0;


    //BCs
    Array<int> ess_tdof_list, ess_bdr(mesh->bdr_attributes.Max());
    ess_bdr = 0;
    ess_bdr[0] = 1;
    fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

    VectorArrayCoefficient f(dim);
    for (int i = 0; i < dim-1; i++)
    {
        f.Set(i, new ConstantCoefficient(0.0));
    }
    {
        Vector pull_force(mesh->bdr_attributes.Max());
        pull_force = 0.0;
        pull_force(1) = -3.0e-2;
        f.Set(dim-1, new PWConstCoefficient(pull_force));
    }

    //
    unique_ptr<ODESolver> ode_solver = ODESolver::Select(ode_solver_type);

    ViscoelasticOperator oper(fes, ess_bdr, lamb, mu, tau);

    socketstream vis_x;
    if (visualization)
    {
        char vishost[] = "localhost";
        int  visport   = 19916;
        vis_x.open(vishost, visport);
        vis_x.precision(8);
        GridFunction *nodes = &x;
        int owns_nodes = 0;
        mesh->SwapNodes(nodes, owns_nodes);
        vis_x << "solution\n" << *mesh << x;
        mesh->SwapNodes(nodes, owns_nodes);
        vis_x << "window_size 800 800\n";
        vis_x << "window_title '" << "Deformation\n";
        vis_x << "keys cm\n";   
        vis_x << "autoscale value\n";
        vis_x << "pause\n";
        vis_x << flush;
    }

    real_t t = 0.0;
    oper.SetTime(t);
    ode_solver->Init(oper);

    bool last_step = false;
    for (int ti = 1; !last_step; ti++)
    {
       real_t dt_real = min(dt, t_final - t);

       ode_solver->Step(x, t, dt_real);

       last_step = (t >= t_final - 1e-8*dt);

       if (last_step || (ti % vis_steps) == 0)
       {
           if (visualization)
           {
               GridFunction *nodes = &x;
               int owns_nodes = 0;
               mesh->SwapNodes(nodes, owns_nodes);
               vis_x << "solution\n" << *mesh << x;
               mesh->SwapNodes(nodes, owns_nodes);
               vis_x << flush;
           }
       }
    }


    {
        GridFunction *nodes = &x;
        int owns_nodes = 0;
        mesh->SwapNodes(nodes, owns_nodes);
        ofstream mesh_ofs("deformed.mesh");
        mesh_ofs.precision(8);
        mesh->Print(mesh_ofs);
        mesh->SwapNodes(nodes, owns_nodes);
        ofstream x_ofs("deformed.sol");
        x_ofs.precision(8);
        x.Save(x_ofs);
    }

    delete a;
    delete b;
    delete fes;
    delete fec;
    delete mesh;

    return 0;
}


ViscoelasticOperator::ViscoelasticOperator(FiniteElementSpace &f, Array<int> &ess_bdr, real_t lamb_, real_t mu_, real_t tau_)
    : TimeDependentOperator(f.GetTrueVSize(), (real_t) 0.0), fespace(f),
      a(&fespace),
      lamb(lamb_), mu(mu_), tau(tau_)
{
    const int skip_zero_entries = 0;

    Array<int> ess_tdof_list;
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

    ConstantCoefficient lamb_coef(lamb), mu_coef(mu);
    a.AddDomainIntegrator(new ElasticityIntegrator(lambda_func,mu_func));
    a.Assemble(skip_zero_entries);
    a.FormSystemMatrix(ess_tdof_list, tmp);

    K_prec = new DSmoother(1);
    MINRESSolver *K_minres = new MINRESSolver;
    K_minres->SetRelTol(rel_tol);
    K_minres->SetAbsTol(0.0);
    K_minres->SetMaxIter(300);
    K_minres->SetPrintLevel(1);
    K_minres->SetPreconditioner(*K_prec);
    K_solver = K_minres;
}

void ViscoelasticOperator::Mult(const Vector &x, Vector &dx_dt) const
{
   K_solver.Mult(x, dx_dt);
}

void ViscoelasticOperator::ImplicitSolve(const real_t dt,
                                         const Vector &x, Vector &dx_dt)
{
   Vector x_vec(x.GetData());
   Vector dx_dt_vec(dx_dt.GetData());

   add(x, dt, dx_dt);
}

ViscoelasticOperator::~ViscoelasticOperator()
{
   delete K_solver;
   delete K_prec;
}



/*
   LinearForm *b = new LinearForm(fespace);
   b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
   //b->AddDomainIntegrator(new ViscoelasticRHSIntegrator(tau, mu_func, m, d));
   b->AddDomainIntegrator(new ViscoelasticRHSIntegrator(mu_func, m_restricted, tau));
   b->Assemble();

   GridFunction x(fespace);
   x = 0.0;

   Vector lambda(mesh->attributes.Max());
   lambda = 1.0;
   lambda(0) = lambda(1)*50;
   PWConstCoefficient lambda_func(lambda);
   Vector mu(mesh->attributes.Max());
   mu = 1.0;
   mu(0) = mu(1)*50;
   PWConstCoefficient mu_func(mu);

   vector<vector<DenseMatrix>> m_storage(fespace->GetNE());

   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new ElasticityIntegrator(lambda_func,mu_func));
   a->Assemble();
*/
