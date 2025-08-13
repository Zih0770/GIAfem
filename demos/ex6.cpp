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
        args.PrintUsage(cout);
        return 1;
    }
    args.PrintOptions(cout);

    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();

    L2_FECollection fec_rho(1, dim);
    FiniteElementSpace fes_rho(mesh, &fec_rho);
    GridFunction rho_gf(&fes_rho);
    ifstream ifstream_rho(density_file);
    if (!ifstream_rho) {
        std::cerr << "Error: Cannot open density file: " << density_file << std::endl;
        return 1;
    }
    rho_gf.Load(ifstream_rho);
    GridFunctionCoefficient rho_coeff(&rho_gf);

    std::cout << "rho_gf.Min() = " << rho_gf.Min() << ", Max = " << rho_gf.Max() << std::endl;

    H1_FECollection fec_phi(order, dim);
    FiniteElementSpace fes_phi(mesh, &fec_phi);
    GridFunction phi_gf(&fes_phi);
    phi_gf = 0.0;
    Array<int> ess_tdof_list;

    // Define scales
    real_t L = 6368000.0; // [m]
    real_t T = 600.0; // [s]
    real_t RHO = 5000.0; // [kg/m^3]
    Nondimensionalisation scalings(L, T, RHO);
    scalings.Print();

    // DtN
    real_t R_ext = 1.2;
    auto DtN = DirichletToNeumannOperator(&fes_phi, lMax);

    // Assemble system: \laplacian phi = c * rho_nd
    Coefficient *rho_nd = scalings.MakeScaledDensityCoefficient(rho_coeff);
    real_t c = 4.0 * M_PI * Constants::G * RHO * T * T; // dimensionless number
    cout<<"RHS dimensionless number c = "<<c<<endl;

    ProductCoefficient rhs_coeff(-c, *rho_nd);

    LinearForm b(&fes_phi);
    b.AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
    b.Assemble();

    ConstantCoefficient one(1.0);
    BilinearForm a(&fes_phi);
    a.AddDomainIntegrator(new DiffusionIntegrator(one));
    a.Assemble();

    Vector B, Phi;
    OperatorPtr A;
    a.FormLinearSystem(ess_tdof_list, phi_gf, b, A, Phi, B);

    auto S = SumOperator(A.Ptr(), 1.0, &DtN, 1.0, false, false);

    GSSmoother prec((SparseMatrix &)(*A));

    CGSolver solver;
    solver.SetOperator(S);
    solver.SetPreconditioner(prec);
    solver.SetRelTol(rel_tol);
    solver.SetMaxIter(3000);
    solver.SetPrintLevel(1);
    solver.Mult(B, Phi);

    a.RecoverFEMSolution(Phi, b, phi_gf);

    // Unscale solution field
    scalings.UnscaleGravityPotential(phi_gf);

    ofstream mesh_ofs("data/ex6.mesh");
    mesh_ofs.precision(8);
    mesh->Print(mesh_ofs);
    ofstream sol_ofs("data/ex6.gf");
    sol_ofs.precision(8);
    phi_gf.Save(sol_ofs);

    if (visualization)
    {
        char vishost[] = "localhost";
        int  visport   = 19916;
        socketstream sol_sock(vishost, visport);
        sol_sock.precision(8);
        sol_sock << "solution\n" << *mesh << phi_gf << flush;
        sol_sock << "density\n" << *mesh << rho_gf << flush;
    }

    delete mesh;
    delete rho_nd;
    return 0;
}

