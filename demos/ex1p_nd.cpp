#include <mfem.hpp>
#include <giafem.hpp>
#include <iostream>
#include <cmath>

using namespace std;
using namespace mfem;
using namespace giafem;

real_t rho_func(const Vector &coord)
{
    real_t r = coord.Norml2();
    if (r > 1.0) { return 0.0; } // mesh is nondimensional
    real_t theta = acos(coord[2] / r);
    real_t phi = atan2(coord[1], coord[0]);
    real_t rho_surface = 2.6e3;
    real_t rho_center = 1.3e4;
    real_t base_rho = rho_center + (rho_surface - rho_center) * r;
    real_t polar_perturb = 0.015 * (1.0 + cos(2.0 * theta));
    real_t azimuthal_perturb = 0.03 * sin(2.0 * phi);
    return base_rho * (1.0 + polar_perturb) * (1.0 + azimuthal_perturb);
}

int main(int argc, char *argv[])
{
    Mpi::Init(argc, argv);
    Hypre::Init();
    int myid = Mpi::WorldRank();

    const char *mesh_file = "mesh/Earth_space_nd.msh";
    real_t rel_tol = 1e-10;
    int order = 1;
    int lMax = 10;
    bool visualization = false;

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
            "Mesh file to use.");
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

    // Define scales
    real_t L = 6371e3; // [m]
    real_t T = 600; // [s]
    real_t RHO = 5000.0; // [kg/m^3]
    Nondimensionalisation scalings(L, T, RHO);
    if (myid == 0) scalings.Print();

    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();
    ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;

    H1_FECollection fec_phi(order, dim);
    ParFiniteElementSpace fes_phi(pmesh, &fec_phi);
    ParGridFunction phi_gf(&fes_phi);
    phi_gf = 0.0;
    Array<int> ess_tdof_list;

    // DtN
    real_t R_ext_nd = 1.25568984461;
    auto DtN = ParDirichletToNeumannOperator(&fes_phi, lMax, R_ext_nd);

    // Assemble system: \laplacian phi = c * rho_nd
    FunctionCoefficient rho_coeff(rho_func); // dimensional
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

    auto S = SumOperator(A.Ptr(), 1, &DtN, 1, false, false);

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
        mesh_name << "data/ex1p_nd_mesh." << setfill('0') << setw(6) << myid;
        sol_name << "data/ex1p_nd_sol." << setfill('0') << setw(6) << myid;

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

    delete pmesh;
    delete rho_nd;
    return 0;
}

