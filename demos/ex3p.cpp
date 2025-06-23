#include <mfem.hpp>
#include <giafem.hpp>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;
using namespace mfem;
using namespace giafem;

real_t mu_func(const Vector &coord);
real_t lamb_func(const Vector &coord);
real_t tau_func(const Vector &coord);
real_t loading_func(const Vector &coord);

int main(int argc, char *argv[])
{
    Mpi::Init(argc, argv);
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();

    const char *mesh_file = "mesh/single_sphere_test.msh";
    const char *output_name = "single_sphere";
    const char *elasticity_model = "linear";
    const char *rheology_model = "Maxwell";
    int ode_solver_type = 1;
    real_t t_final = 10000; //year
    real_t dt = 10;
    real_t rel_tol = 1e-8;
    real_t dt_res = 1e-3;
    int order = 2;
    int ser_ref_levels = -1;
    int par_ref_levels = -1;
    bool amg_elast = 0;
    const char *petscrc_file = "demos/petscopts_viscoelastic";
    //bool petscmonitor = false;
    //bool forcewrap = false;
    //bool useh2 = false;
    //bool use_nonoverlapping = false;
    //bool petsc_use_jfnk = false;
    bool static_cond = false;
    const char *device_config = "cpu";
    bool visualization = false;
    int vis_steps = 1;

    //Parsing
    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
    args.AddOption(&output_name, "-f", "--file",
                   "Output file.");
    args.AddOption(&elasticity_model, "-em", "--elasticity-model",
                   "Elasticity model to use: linear, neo-hookean, etc.");
    args.AddOption(&rheology_model, "-rm", "--rheology-model",
                   "Rheology model to use: Maxwell, Kelvin-Voigt, etc.");
    args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
    args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
    args.AddOption(&rel_tol, "-rt", "--rel-tol",
                  "Relative tolerance for linear solving.");
    args.AddOption(&dt_res, "-im_res", "--implicit-res",
                  "Relative iteration residue in implicit time schemes.");
    args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  ODESolver::Types.c_str());
    args.AddOption(&order, "-o", "--order",
            "Order (degree) of the finite elements.");
    args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
            "Number of times to refine the mesh uniformly in serial.");
    args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
            "Number of times to refine the mesh uniformly in parallel.");
    args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                  "PetscOptions file to use.");
    args.AddOption(&amg_elast, "-elast", "--amg-for-elasticity", "-sys",
            "--amg-for-systems",
            "Use the special AMG elasticity solver (GM/LN approaches), "
            "or standard AMG for systems (unknown approach).");
    args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
            "--no-static-condensation", "Enable static condensation.");
    //args.AddOption(&use_nonoverlapping, "-nonoverlapping", "--nonoverlapping",
    //              "-no-nonoverlapping", "--no-nonoverlapping",
    //              "Use or not the block diagonal PETSc's matrix format "
    //              "for non-overlapping domain decomposition.");
    args.AddOption(&device_config, "-d", "--device",
            "Device configuration string, see Device::Configure().");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
            "--no-visualization",
            "Enable or disable GLVis visualization.");
    args.AddOption(&vis_steps, "-vs", "--visualization-steps",
            "Visualize every n-th timestep.");
    args.Parse();
    if (!args.Good())
    {
        if (myid == 0)
        {
            args.PrintUsage(cout);
        }
        return 1;
    }
    if (myid == 0)
    {
        args.PrintOptions(cout);
    }

    Device device(device_config);
    if (myid == 0) { device.Print(); }

    MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL);

    //Mesh
    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();
    for (int l = 0; l < ser_ref_levels; l++)
    {
        mesh->UniformRefinement();
    }

    ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;
    {
        for (int l = 0; l < par_ref_levels; l++)
        {
            pmesh->UniformRefinement();
        }
    }

    //FE Space
    int order_m = order - 1; int order_properties = order - 1; int order_w = 2 * (order - 1); 
    H1_FECollection fec_u(order, dim);
    L2_FECollection fec_m(order_m, dim);
    L2_FECollection fec_properties(order_properties, dim);
    L2_FECollection fec_w(order_w, dim);
    ParFiniteElementSpace fes_u(pmesh, &fec_u, 3);
    ParFiniteElementSpace fes_m(pmesh, &fec_m, 5);
    ParFiniteElementSpace fes_properties(pmesh, &fec_properties);
    ParFiniteElementSpace fes_w(pmesh, &fec_w);
    HYPRE_BigInt u_size = fes_u.GlobalTrueVSize();
    if (myid == 0)
    {
        cout << "Number of u-unknowns: " << u_size << endl;
    }
    ParGridFunction u_gf(&fes_u); ParGridFunction m_gf(&fes_m); ParGridFunction d_gf(&fes_m); ParGridFunction w_gf(&fes_w);
    u_gf = 0.0; m_gf = 0.0; d_gf = 0.0; w_gf = 0.0;
    Vector m_vec;
    m_gf.GetTrueDofs(m_vec);
    FunctionCoefficient mu_coeff(mu_func);
    FunctionCoefficient lamb_coeff(lamb_func);
    FunctionCoefficient tau_coeff(tau_func);
    FunctionCoefficient loading_coeff(loading_func);

    //ParGridFunction nodes(&fes_u);
    //pmesh->GetNodes(nodes);
    ParGridFunction *nodes = new ParGridFunction(&fes_u);
    VectorFunctionCoefficient identity(dim, [](const Vector &x, Vector &y) { y = x; });
    nodes->ProjectCoefficient(identity);
    pmesh->NewNodes(*nodes, false);
    MFEM_VERIFY(pmesh->GetNodes(), "Mesh has no nodal coordinates!");

    u_gf.SetTrueVector();
    u_gf.SetFromTrueVector();


    //Time-depednet operator
    ViscoelasticOperator oper(fes_u, fes_m, fes_properties, fes_w, u_gf, m_gf, d_gf, lamb_coeff, mu_coeff, tau_coeff, loading_coeff,
                              rel_tol, dt_res, elasticity_model, rheology_model);
    unique_ptr<ODESolver> ode_solver = ODESolver::Select(ode_solver_type);
    //ODESolver *ode_solver = new BaileySolver;
    ode_solver->Init(oper);

    //Time stepping
    socketstream vis_w;
    if (visualization)
    {
       char vishost[] = "localhost";
       int  visport   = 19916;
       vis_w.open(vishost, visport);
       vis_w.precision(8);
       w_gf = 0.0;
       Visualize(vis_w, pmesh, &u_gf, &w_gf, "Elastic energy density", true);
       MPI_Barrier(pmesh->GetComm());
    }
    real_t year2sec = 3.15576e7;
    real_t t = 0.0;
    t_final *= year2sec; dt *= year2sec;
    bool last_step = false;
    for (int ti = 1; !last_step; ti++)
    {
        if (myid == 0)
            cout<<"t = "<<round(t/year2sec)<<" year:"<<endl;
        real_t dt_real = min(dt, t_final - t);
        ode_solver->Step(m_vec, t, dt_real);
        last_step = (t >= t_final - 1e-8*dt);
        if (last_step || (ti % vis_steps) == 0 )
        {
            u_gf.SetFromTrueVector();
            if (visualization)
            {
                oper.CalcStrainEnergyDensity(w_gf);
                cout<<"u_max: "<<u_gf.Normlinf()<<endl;
                cout<<"W_max: "<<w_gf.Normlinf()<<endl;
                Visualize(vis_w, pmesh, &u_gf, &w_gf, "Elastic energy density");
            }
        }

    }

    //Saving
    //pmesh->SetNodalFESpace(fes_u);
    *pmesh->GetNodes() += u_gf;
    u_gf.Neg();
    ostringstream mesh_name;
    mesh_name << "data/mesh." << setfill('0') << setw(6) << myid;
    ofstream omesh(mesh_name.str().c_str());
    omesh.precision(8);
    pmesh->Print(omesh);
    ostringstream sol_name;
    sol_name << "data/mesh." << setfill('0') << setw(6) << myid;
    ofstream osol(sol_name.str().c_str());
    osol.precision(8);
    u_gf.Save(osol);

    delete pmesh;
    //delete nodes;

    MFEMFinalizePetsc();

    return 0;
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
    return base_lamb * (1.0 + polar_perturb) * (1.0 + azimuthal_perturb);
}

real_t tau_func(const Vector &coord)
{
    real_t r = coord.Norml2();
    real_t r_norm = r / 6371e3;
    real_t theta = acos(coord[2] / r);
    real_t phi = atan2(coord[1], coord[0]);
    real_t tau_center  = 5000.0 * 3.15576e7;  // 5000 years in seconds
    real_t tau_surface = 500.0  * 3.15576e7;
    real_t base_tau = tau_center + (tau_surface - tau_center) * r_norm;
    real_t polar_perturb = 15.0 * (1.0 + cos(2.0 * theta));
    real_t azimuthal_perturb = 10.0 * (1 + sin(2.0 * phi));
    return base_tau * (1.0 + polar_perturb) * (1.0 + azimuthal_perturb);
}

real_t loading_func(const Vector &coord)
{
    real_t r = coord.Norml2();
    real_t theta = acos(coord[2] / r);
    real_t phi = atan2(coord[1], coord[0]);
    // Max loading at poles (glaciers): e.g., 10 MPa (~1 km ice)
    const real_t polar_load = -10e6;
    // Equatorial loading (oceans): e.g., 1 MPa (~100 m water depth)
    const real_t equator_load = -1e6;
    real_t base_load = equator_load + (polar_load - equator_load) / 2.0 * cos(2.0 * theta);
    real_t azimuthal_perturb = 0.2 * sin(2.0 * phi);
    return base_load * (1.0 + azimuthal_perturb);
}

