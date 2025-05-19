#include "mfem.hpp"
#include "giafem.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>

using namespace std;
using namespace mfem;
using namespace giafem;

void visualize(ostream &os, Mesh *mesh, GridFunction *deformed_nodes,
               GridFunction *field, const char *field_name = NULL,
               bool init_vis = false);


int main(int argc, char *argv[])
{
    const char *mesh_file = "mesh/core_mantle.mesh";
    const char *output_file = "output.dat";
    int ode_solver_type = 23;
    real_t t_final = 5.0;
    real_t dt = 0.02;
    int order = 2;
    int ref_levels = 3;
    bool visualization = false;
    int vis_steps = 1;

    real_t tau = 5.0;
    real_t lamb = 15.0;
    real_t mu = 10.0;

    OptionsParser args(argc, argv);
    args.AddOption(&output_file, "-f", "--file",
                   "Output file.");
    args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
    args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
    args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  ODESolver::Types.c_str());
    args.Parse();
    if (!args.Good())
    {
       args.PrintUsage(cout);
       return 1;
    }
    args.PrintOptions(cout);


    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();
    {
        int ref_levels = (int) floor(log(10000./mesh->GetNE())/log(2.)/dim);
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
                             
    VeOperator oper(fes_u, fes_m, fes_w, lamb_func, mu_func, tau_func, u, m, d);
    //unique_ptr<ODESolver> ode_solver = ODESolver::Select(ode_solver_type);
    //ODESolver *ode_solver = new ForwardEulerSolver();
    //ODESolver *ode_solver = new RK2Solver;
    //ODESolver *ode_solver = new RK3SSPSolver;
    //ODESolver *ode_solver = new RK4Solver;
    //ODESolver *ode_solver = new RK6Solver;
    //ODESolver *ode_solver = new AB1Solver;
    //ODESolver *ode_solver = new AB2Solver;
    //ODESolver *ode_solver = new AB3Solver;
    //ODESolver *ode_solver = new AB4Solver;
    //ODESolver *ode_solver = new AB5Solver;
    ODESolver *ode_solver = new BaileySolver;

    //ODESolver *ode_solver = new BackwardEulerSolver;
    //ODESolver *ode_solver = new SDIRK23Solver(2);
    //ODESolver *ode_solver = new SDIRK33Solver;
    //ODESolver *ode_solver = new ImplicitMidpointSolver;
    //ODESolver *ode_solver = new SDIRK23Solver;
    //ODESolver *ode_solver = new SDIRK34Solver;
    //ODESolver *ode_solver = new AM2Solver;

    ode_solver->Init(oper);
    ofstream fout(output_file);
    fout << "# t   L2_u\n";
    real_t t = 0.0;
    socketstream vis_w;
    u.SetFromTrueDofs(u_vec); 
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
    auto start = std::chrono::high_resolution_clock::now();
    for (int ti = 1; !last_step; ti++)
    {
        cout<<"t = "<<t<<":"<<endl;
        auto current = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = current - start;
        fout << t << " " << elapsed.count() << " " << u.Norml2() << "\n";
        
        real_t dt_real = min(dt, t_final - t);
        ode_solver->Step(m_vec, t, dt_real);
        last_step = (t >= t_final - 1e-8*dt);

        if (last_step || (ti % vis_steps) == 0)
        {
            if (visualization)
            {
                StrainEnergyCoefficient strainEnergyDensity(u, lamb, mu);
                w.ProjectCoefficient(strainEnergyDensity);
                cout << "||u||_L2 = " << u.Norml2() << std::endl;
                cout<<"w_L2: "<<w.Norml2()<<endl;
                cout<<"m_msv: "<<m.Norml2()/sqrt(m.Size())<<endl;
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


