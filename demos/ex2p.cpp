#include <mfem.hpp>
#include <giafem.hpp>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;
using namespace mfem;
using namespace giafem;

real_t rho_func(const Vector &coord);
real_t mu_func(const Vector &coord);
real_t lamb_func(const Vector &coord);
real_t loading_func(const Vector &coord);

int main(int argc, char *argv[])
{
    Mpi::Init(argc, argv);
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();

    const char *mesh_file = "mesh/Earth_space.msh";
    const char *output_name = "elasticity_gravity";
    const char *elasticity_model = "linear";
    real_t rel_tol = 1e-8;
    int order = 2;
    int lMax = 0;
    int ser_ref_levels = -1;
    int par_ref_levels = -1;
    bool amg_elast = 0;
    const char *petscrc_file = "demos/petscopts_elasticity_gravity";
    //bool petscmonitor = false;
    //bool forcewrap = false;
    //bool useh2 = false;
    //bool use_nonoverlapping = false;
    //bool petsc_use_jfnk = false;
    bool static_cond = false;
    const char *device_config = "cpu";
    bool visualization = false;

    real_t coeff_rho = - 4.0 * M_PI * Constants::G;


    //Parsing
    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
    args.AddOption(&output_name, "-f", "--file",
                   "Output file.");
    args.AddOption(&elasticity_model, "-em", "--elasticity-model",
                   "Elasticity model to use: linear, neo-hookean, etc.");
    args.AddOption(&rel_tol, "-rt", "--rel-tol",
                  "Relative tolerance for linear solving.");
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
    args.AddOption(&lMax, "-l", "--lMax", "Truncation degree for the DtN map.");
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
    mesh->SetAttributes();

    Array<int> attr0 = mesh->attributes;
    cout<<"Domain attributes: "<<attr0[0]<<endl;
    cout<<"Domain attributes: "<<attr0[1]<<endl;
    cout<<"Domain attributes: "<<attr0.Size()<<endl;
    attr0.DeleteLast();
    cout<<"Domain attributes: "<<attr0.Size()<<endl;

    ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;
    {
        for (int l = 0; l < par_ref_levels; l++)
        {
            pmesh->UniformRefinement();
        }
    }

    ParSubMesh pmesh0(ParSubMesh::CreateFromDomain(*pmesh, attr0));
    //ParSubMesh *pmesh0;
    cout<<"Domain attributes: "<<attr0.Max()<<endl;
    //pmesh0->CreateFromDomain(*pmesh, attr0);
    cout<<"Domain attributes: "<<attr0.Max()<<endl;

    //FE Space
    int order_phi = order; int order_properties = order - 1; int order_w = 2 * (order - 1); 
    H1_FECollection fec_u(order, dim);
    H1_FECollection fec_phi(order_phi, dim);
    L2_FECollection fec_properties(order_properties, dim);
    L2_FECollection fec_w(order_w, dim);
    ParFiniteElementSpace fes_phi(pmesh, &fec_phi);
    ParFiniteElementSpace fes_u(&pmesh0, &fec_u, 3);
    ParFiniteElementSpace fes_properties(&pmesh0, &fec_properties);
    ParFiniteElementSpace fes_w(&pmesh0, &fec_w);
    HYPRE_BigInt u_size = fes_u.GlobalTrueVSize();
    HYPRE_BigInt phi_size = fes_phi.GlobalTrueVSize();
    if (myid == 0)
    {
        cout << "Number of u-unknowns: " << u_size << endl;
        cout << "Number of phi-unknowns: " << phi_size << endl;
    }
    ParGridFunction u_gf(&fes_u); ParGridFunction phi_gf(&fes_phi); ParGridFunction Phi_gf(&fes_phi); ParGridFunction w_gf(&fes_w);
    u_gf = 0.0; phi_gf = 0.0; Phi_gf = 0.0; w_gf = 0.0;
    FunctionCoefficient rho_coeff(rho_func);
    FunctionCoefficient mu_coeff(mu_func);
    FunctionCoefficient lamb_coeff(lamb_func);
    FunctionCoefficient loading_coeff(loading_func);

    Array<int> block_offsets(3);
    block_offsets[0] = 0;
    block_offsets[1] = fes_u.GetVSize();
    block_offsets[2] = fes_phi.GetVSize();
    block_offsets.PartialSum();

    Array<int> block_trueOffsets(3);
    block_trueOffsets[0] = 0;
    block_trueOffsets[1] = fes_u.TrueVSize();
    block_trueOffsets[2] = fes_phi.TrueVSize();
    block_trueOffsets.PartialSum();

    MemoryType mt = device.GetMemoryType();
    BlockVector x(block_offsets, mt), rhs(block_offsets, mt);
    BlockVector X(block_trueOffsets, mt), Rhs(block_trueOffsets, mt);

    //Compute the equilibrium state




    delete pmesh, pmesh0;

    MFEMFinalizePetsc();

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
        real_t rho_surface = 2.6e3;  // Pa
        real_t rho_center = 1.3e4;    // Pa
        real_t base_rho =  rho_center + (rho_surface - rho_center) * r_norm;
        return - 4.0 * M_PI * Constants::G * base_rho;
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
