//Self-gravitation computation with the Dirichlet-to-Neumann approach
#include <mfem.hpp>
#include <giafem.hpp>
#include <iostream>
#include <cmath>

using namespace std;
using namespace mfem;
using namespace giafem;

real_t rho_func(const Vector &coord);

int main(int argc, char *argv[])
{
    Mpi::Init(argc, argv);
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();

    const char *mesh_file = "mesh/Earth_space.msh";
    real_t rel_tol = 1e-10;
    int order = 1;
    int lMax = 10;
    int ser_ref_levels = -1;
    int par_ref_levels = -1;
    bool use_petsc = false;
    const char *petscrc_file = "demos/petscopts_ex1p";
    bool static_cond = false;
    const char *device_config = "cpu";
    bool visualization = false;

    //Parsing
    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
    args.AddOption(&rel_tol, "-rt", "--rel-tol",
                  "Relative tolerance for linear solving.");
    args.AddOption(&order, "-o", "--order",
            "Order (degree) of the finite elements.");
    args.AddOption(&lMax, "-l", "--lMax", "Truncation degree for the DtN map.");
    args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
            "Number of times to refine the mesh uniformly in serial.");
    args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
            "Number of times to refine the mesh uniformly in parallel.");
    args.AddOption(&use_petsc, "-usepetsc", "--usepetsc", "-no-petsc",
                  "--no-petsc",
                  "Use or not PETSc to solve the linear system.");
    args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
            "PetscOptions file to use.");
    args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
            "--no-static-condensation", "Enable static condensation.");
    args.AddOption(&device_config, "-d", "--device",
            "Device configuration string, see Device::Configure().");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
            "--no-visualization",
            "Enable or disable GLVis visualization.");
    args.Parse();
    if (!args.Good())
    {
        if (myid == 0)
        {
            args.PrintUsage(cout);
            return 1;
        }
    }
    if (myid == 0)
    {
        args.PrintOptions(cout);
    }

    Device device(device_config);
    if (myid == 0) { device.Print(); }

    if (use_petsc) { MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL); }

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
    H1_FECollection fec_phi(order, dim);
    ParFiniteElementSpace fes_phi(pmesh, &fec_phi);
    if (myid == 0)
    {
        cout << "Number of phi-unknowns: " << fes_phi.GlobalTrueVSize() << endl;
    }
    ParGridFunction phi_gf(&fes_phi);    
    phi_gf = 0.0; 

    Array<int> ess_tdof_list;

    //DtN
    auto DtN = ParDirichletToNeumannOperator(&fes_phi, lMax, Constants::R_ext);

    //Construct the linear system
    FunctionCoefficient rho_coeff(rho_func);

    ParLinearForm b(&fes_phi);
    b.AddDomainIntegrator(new DomainLFIntegrator(rho_coeff));
    b.Assemble();

    ParBilinearForm a(&fes_phi);
    auto one = ConstantCoefficient(1.0);
    a.AddDomainIntegrator(new DiffusionIntegrator(one));
    a.Assemble();

    Vector B, Phi;

    if (!use_petsc)
    {
    //HypreParMatrix A;
    OperatorPtr A;
    a.FormLinearSystem(ess_tdof_list, phi_gf, b, A, Phi, B);
    //cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;

    auto S = SumOperator(A.Ptr(), 1, &DtN, 1, false, false);

    OperatorJacobiSmoother prec(a, ess_tdof_list);
    //HypreBoomerAMG prec((HypreParMatrix &)(*A));

    CGSolver solver(MPI_COMM_WORLD);
    //BiCGSTABSolver solver(MPI_COMM_WORLD);
    //GMRESSolver solver(MPI_COMM_WORLD);
    //MINRESSolver solver(MPI_COMM_WORLD);
    solver.SetOperator(S);
    solver.SetPreconditioner(prec);
    solver.SetRelTol(rel_tol);
    solver.SetMaxIter(3000);
    solver.SetPrintLevel(1);
    solver.Mult(B, Phi);
    } 
    else
    {
        PetscParMatrix A;
        a.SetOperatorType(Operator::PETSC_MATAIJ);
        a.FormLinearSystem(ess_tdof_list, phi_gf, b, A, Phi, B);
        if (myid == 0)
        {
            cout << "Size of linear system: " << A.M() << endl;
        }
        //A.SetBlockSize(dim);

        auto S = SumOperator(&A, 1, &DtN, 1, false, false);

        PetscLinearSolver solver(MPI_COMM_WORLD);
        PetscPreconditioner *prec = NULL;

        solver.SetOperator(S);
        solver.SetTol(rel_tol);
        solver.SetMaxIter(3000);
        solver.SetPrintLevel(1);
        solver.Mult(B, Phi);
    }
    a.RecoverFEMSolution(Phi, b, phi_gf);

    //Saving
    {
        ostringstream mesh_name, sol_name;
        mesh_name << "data/ex1p_mesh." << setfill('0') << setw(6) << myid;
        sol_name << "data/ex1p_sol." << setfill('0') << setw(6) << myid;

        ofstream mesh_ofs(mesh_name.str().c_str());
        mesh_ofs.precision(8);
        pmesh->Print(mesh_ofs);

        ofstream sol_ofs(sol_name.str().c_str());
        sol_ofs.precision(8);
        phi_gf.Save(sol_ofs);
    }

    if (visualization)
    {
        char vishost[] = "localhost";
        int  visport   = 19916;
        socketstream sol_sock(vishost, visport);
        sol_sock << "parallel " << num_procs << " " << myid << "\n";
        sol_sock.precision(8);
        sol_sock << "solution\n" << *pmesh << phi_gf << flush;
    }

    delete pmesh;

    if (use_petsc) { MFEMFinalizePetsc(); }

    return 0;
}

real_t rho_func(const Vector &coord)
{
    real_t r = coord.Norml2();
    if (r > Constants::R){
        return 0.0;
    } else{
        real_t r_norm = r / 6371e3;
        real_t theta = acos(coord[2] / r); // polar angle
        real_t phi = atan2(coord[1], coord[0]); // azimuthal angle
        real_t rho_surface = 2.6e3;  // Pa
        real_t rho_center = 1.3e4;    // Pa
        real_t base_rho =  rho_center + (rho_surface - rho_center) * r_norm;
        real_t polar_perturb = 0.015 * (1.0 + cos(2.0 * theta));
        real_t azimuthal_perturb = 0.03 * sin(2.0 * phi);
        return - 4.0 * M_PI * Constants::G * base_rho * (1.0 + polar_perturb) * (1.0 + azimuthal_perturb);
    }
}


