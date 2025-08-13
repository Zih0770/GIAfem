//Self-gravitation computation with the Dirichlet-to-Neumann approach
#include <mfem.hpp>
#include <giafem.hpp>
#include <mfemElasticity.hpp>
#include <iostream>
#include <cmath>

using namespace std;
using namespace mfem;
using namespace giafem;

real_t rho_func(const Vector &coord);

int main(int argc, char *argv[])
{
    const char *mesh_file = "mesh/Earth_space.msh";
    real_t rel_tol = 1e-10;
    int order = 1;
    int lMax = 10;
    int ref_levels = -1;
    bool static_cond = false;
    bool visualization = false;
    StopWatch chrono;

    //Parsing
    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
    args.AddOption(&rel_tol, "-rt", "--rel-tol",
                  "Relative tolerance for linear solving.");
    args.AddOption(&order, "-o", "--order",
            "Order (degree) of the finite elements.");
    args.AddOption(&lMax, "-l", "--lMax", "Truncation degree for the DtN map.");
    args.AddOption(&ref_levels, "-rs", "--refine-serial",
            "Number of times to refine the mesh uniformly in serial.");
    args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
            "--no-static-condensation", "Enable static condensation.");
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

    //Mesh
    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();
    for (int l = 0; l < ref_levels; l++)
    {
        mesh->UniformRefinement();
    }

    //FE Space
    H1_FECollection fec_phi(order, dim);
    FiniteElementSpace fes_phi(mesh, &fec_phi);
    cout << "Number of phi-unknowns: " << fes_phi.GetTrueVSize() << endl;
    GridFunction phi_gf(&fes_phi);    
    phi_gf = 0.0; 

    Array<int> ess_tdof_list;

    //DtN
    chrono.Clear();
    chrono.Start();
    auto DtN = DirichletToNeumannOperator(&fes_phi, lMax);
    chrono.Stop();
    cout << "Constructing the DtN operator (vectors) took " << chrono.RealTime() << "s.\n";

    chrono.Clear();
    chrono.Start();
    auto DtN_mat = mfemElasticity::PoissonDtNOperator(&fes_phi, lMax);
    chrono.Stop();
    cout << "Constructing the DtN operator (matrix) took " << chrono.RealTime() << "s.\n";

    //Construct the linear system
    FunctionCoefficient rho_coeff(rho_func);

    LinearForm b(&fes_phi);
    b.AddDomainIntegrator(new DomainLFIntegrator(rho_coeff));
    b.Assemble();

    BilinearForm a(&fes_phi);
    auto one = ConstantCoefficient(1.0);
    a.AddDomainIntegrator(new DiffusionIntegrator(one));
    a.Assemble();

    OperatorPtr A;
    Vector B, Phi;

    a.FormLinearSystem(ess_tdof_list, phi_gf, b, A, Phi, B);
    cout << "Size of linear system: " << A->Height() << endl;

    auto S = SumOperator(A.Ptr(), 1.0, &DtN, 1.0, false, false);

    GSSmoother M((SparseMatrix &)(*A));
    //DSmoother M((SparseMatrix &)(*A));

    auto solver = CGSolver();
    //auto solver = BiCGSTABSolver();
    solver.SetOperator(S);
    solver.SetPreconditioner(M);
    solver.SetRelTol(rel_tol);
    solver.SetMaxIter(2000);
    solver.SetPrintLevel(1);
    solver.Mult(B, Phi);

    a.RecoverFEMSolution(Phi, b, phi_gf);

    //Saving
    ofstream mesh_ofs("data/ex1.mesh");
    mesh_ofs.precision(8);
    mesh->Print(mesh_ofs);
    ofstream sol_ofs("data/ex1.gf");
    sol_ofs.precision(8);
    phi_gf.Save(sol_ofs);

    if (visualization)
    {
        char vishost[] = "localhost";
        int  visport   = 19916;
        socketstream sol_sock(vishost, visport);
        sol_sock.precision(8);
        sol_sock << "solution\n" << *mesh << phi_gf << flush;
    }

    delete mesh;

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


