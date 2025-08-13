//Computing the self-gravitation from the PREM data file
#include <mfem.hpp>
#include <giafem.hpp>
#include <iostream>
#include <cmath>

using namespace std;
using namespace mfem;
using namespace giafem;


int main(int argc, char *argv[])
{
    Mpi::Init(argc, argv);
    Hypre::Init();
    int myid = Mpi::WorldRank();

    const char *mesh_file = "mesh/prem.msh";
    const char *density_file = "mesh/prem_density.gf";
    real_t rel_tol = 1e-10;
    int order = 1;
    int lMax = 10;
    bool visualization = false;

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
            "Mesh file to use.");
    args.AddOption(&density_file, "-df", "--density-file",
            "Density file to use.");
    args.AddOption(&rel_tol, "-rt", "--rel-tol",
            "Relative tolerance for linear solving.");
    args.AddOption(&order, "-o", "--order",
            "Order (degree) of the finite elements.");
    args.AddOption(&lMax, "-l", "--lMax", "Truncation degree for the DtN map.");
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

    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();
    ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);

    L2_FECollection fec_rho(1, dim);
    FiniteElementSpace fes_rho_ser(mesh, &fec_rho);
    GridFunction rho_gf_ser(&fes_rho_ser);
    ifstream ifstream_rho(density_file);
    rho_gf_ser.Load(ifstream_rho);
    GridFunctionCoefficient rho_coeff(&rho_gf_ser);

    H1_FECollection fec_phi(order, dim);
    ParFiniteElementSpace fes_phi(pmesh, &fec_phi);
    ParGridFunction phi_gf(&fes_phi);
    phi_gf = 0.0;
    Array<int> ess_tdof_list;

    // Define scales
    real_t L = 6368000.0; // [m]
    real_t T = 600.0; // [s]
    real_t RHO = 5000.0; // [kg/m^3]
    Nondimensionalisation scalings(L, T, RHO);
    if (myid == 0) scalings.Print();

    // DtN
    real_t R_ext = 1.2;
    auto DtN = ParDirichletToNeumannOperator(&fes_phi, lMax, R_ext);

    // Assemble system: \laplacian phi = c * rho_nd
    Coefficient *rho_nd = scalings.MakeScaledDensityCoefficient(rho_coeff);
    real_t c = 4.0 * M_PI * Constants::G * RHO * T * T; // dimensionless number
    if (myid == 0)
    {
        cout<<"RHS dimensionless number c = "<<c<<endl;
    }
    ProductCoefficient rhs_coeff(-c, *rho_nd);

    ParLinearForm b(&fes_phi);
    b.AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
    b.Assemble();

    ConstantCoefficient one(1.0);
    ParBilinearForm a(&fes_phi);
    a.AddDomainIntegrator(new DiffusionIntegrator(one));
    a.Assemble();

    Vector B, Phi;
    OperatorPtr A;
    a.FormLinearSystem(ess_tdof_list, phi_gf, b, A, Phi, B);

    auto S = SumOperator(A.Ptr(), 1.0, &DtN, 1.0, false, false);

    //HypreBoomerAMG prec((HypreParMatrix &)(*A));
    OperatorJacobiSmoother prec(a, ess_tdof_list);

    CGSolver solver(MPI_COMM_WORLD);
    solver.SetOperator(S);
    solver.SetPreconditioner(prec);
    solver.SetRelTol(rel_tol);
    solver.SetMaxIter(3000);
    solver.SetPrintLevel(1);
    solver.Mult(B, Phi);

    a.RecoverFEMSolution(Phi, b, phi_gf);

    // Unscale solution field
    scalings.UnscaleGravityPotential(phi_gf);

    {
        ostringstream mesh_name, sol_name;
        mesh_name << "data/ex6p_mesh." << setfill('0') << setw(6) << myid;
        sol_name << "data/ex6p_sol." << setfill('0') << setw(6) << myid;

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
        sol_sock << "parallel " << Mpi::WorldSize() << " " << myid << "\n";
        sol_sock.precision(8);
        sol_sock << "solution\n" << *pmesh << phi_gf << "window_title 'Dimensionless'" << flush;
    }

    delete mesh;
    delete pmesh;
    delete rho_nd;
    return 0;
}

