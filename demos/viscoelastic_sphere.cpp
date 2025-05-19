#include "mfem.hpp"
#include "giafem.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>

using namespace std;
using namespace mfem;
using namespace giafem;

real_t mu_func(const Vector &coord);
real_t lamb_func(const Vector &coord);
real_t tau_func(const Vector &coord);
real_t loading_func(const Vector &coord);

real_t R_earth = 6371e3;
int main(int argc, char *argv[])
{
    const char *mesh_file = "mesh/single_sphere.msh";
    const char *output_file = "output.dat";
    int ode_solver_type = 23;
    real_t t_final = 100; //year
    real_t dt = 1;
    int order = 2;
    int mesh_size = 10000;
    bool visualization = false;
    int vis_steps = 1;

    //Parsing
    OptionsParser args(argc, argv);
    args.AddOption(&output_file, "-f", "--file",
                   "Output file.");
    args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
    args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
    args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  ODESolver::Types.c_str());
    args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
    args.AddOption(&mesh_size, "-N", "--mesh-size",
                  "Maximum of elements");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
    args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
    args.Parse();
    if (!args.Good())
    {
       args.PrintUsage(cout);
       return 1;
    }
    args.PrintOptions(cout);

    //Mesh
    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();
    {
        int ref_levels = (int) floor(log(static_cast<real_t>(mesh_size)/mesh->GetNE())/log(2.)/dim);
        for (int l = 0; l < ref_levels; l++)
        {
            mesh->UniformRefinement();
        }
    }

    //FE Space
    int order_properties = order; int order_w = 2 * (order - 1); 
    H1_FECollection fec_u(order, dim);
    L2_FECollection fec_m(order - 1, dim);
    L2_FECollection fec_properties(order_properties, dim);
    L2_FECollection fec_w(order_w, dim);
    FiniteElementSpace fes_u(mesh, &fec_u, 3);
    FiniteElementSpace fes_m(mesh, &fec_m, 5);
    FiniteElementSpace fes_properties(mesh, &fec_properties);
    FiniteElementSpace fes_w(mesh, &fec_w);
    int u_size = fes_u.GetTrueVSize();
    cout << "Number of u-unknowns: " << u_size << endl;
    GridFunction u_gf(&fes_u); GridFunction m_gf(&fes_m); GridFunction d_gf(&fes_m); GridFunction w_gf(&fes_w);
    u_gf = 0.0; m_gf = 0.0;
    Vector u_vec, m_vec;
    u_gf.GetTrueDofs(u_vec);
    m_gf.GetTrueDofs(m_vec);
    FunctionCoefficient mu_coeff(mu_func);
    FunctionCoefficient lamb_coeff(lamb_func);
    FunctionCoefficient tau_coeff(tau_func);
    FunctionCoefficient loading_coeff(loading_func);

    GridFunction *nodes = new GridFunction(&fes_u);
    VectorFunctionCoefficient identity(dim, [](const Vector &x, Vector &y) { y = x; });
    nodes->ProjectCoefficient(identity);
    mesh->NewNodes(*nodes, false);
    MFEM_VERIFY(mesh->GetNodes(), "Mesh has no nodal coordinates!");
           
    //Time depednet operator
    VeOperator oper(fes_u, fes_m, fes_properties, fes_w, u_gf, m_gf, d_gf, lamb_coeff, mu_coeff, tau_coeff, loading_coeff);
    unique_ptr<ODESolver> ode_solver = ODESolver::Select(ode_solver_type);
    //ODESolver *ode_solver = new BaileySolver;
    ode_solver->Init(oper);

    //Time stepping
    ofstream fout(output_file);
    fout << "# t   L2_u\n";
    socketstream vis_w;
    if (visualization)
    {
       char vishost[] = "localhost";
       int  visport   = 19916;
       vis_w.open(vishost, visport);
       oper.CalcStrainEnergyDensity(w_gf);
       vis_w.precision(8);
       visualize(vis_w, mesh, &u_gf, &w_gf, "Elastic energy density", true);
       cout << "GLVis visualization paused."
            << " Press space (in the GLVis window) to resume it.\n";
    }
   
    real_t t = 0.0;
    t_final *= 3.1536e7; dt *= 3.1536e7;
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
                oper.CalcStrainEnergyDensity(w_gf);
                visualize(vis_w, mesh, &u_gf, &w_gf, "Elastic energy density");
            }
        }

    }

    *mesh->GetNodes() += u_gf;
    u_gf.Neg();
    ofstream omesh("data/ve_sphere.mesh");
    omesh.precision(8);
    mesh->Print(omesh);
    ofstream osol("data/ve_sphere.gf");
    osol.precision(8);
    u_gf.Save(osol);

    delete mesh;
    delete nodes;

    return 0;
}


real_t mu_func(const Vector &coord)
{
    real_t r = coord.Norml2();
    real_t r_norm = r / R_earth;
    real_t mu_surface = 70e9;  // Pa
    real_t mu_center = 140e9;    // Pa
    return mu_center + (mu_surface - mu_center) * r_norm;
}

real_t lamb_func(const Vector &coord)
{
    real_t r = coord.Norml2();
    real_t r_norm = r / R_earth;
    real_t lamb_surface = 100e9;
    real_t lamb_center = 300e9;   
    return lamb_center + (lamb_surface - lamb_center) * r_norm;
}

real_t tau_func(const Vector &coord)
{
    real_t r = coord.Norml2();
    real_t r_norm = r / R_earth;
    real_t tau_center  = 5000.0 * 3.15576e7;  // 5000 years in seconds
    real_t tau_surface = 500.0  * 3.15576e7;
    return tau_surface + (tau_surface - tau_center) * r_norm;;
}

real_t loading_func(const Vector &coord)
{
    real_t r = coord.Norml2();
    real_t z = coord[2];
    real_t latitude = asin(z / r);
    // Max loading at poles (glaciers): e.g., 10 MPa (~1 km ice)
    const real_t polar_load = -10e6;
    // Equatorial loading (oceans): e.g., 1 MPa (~100 m water depth)
    const real_t equator_load = -1e6;

    return equator_load + (polar_load - equator_load) * fabs(sin(latitude));
}

