/*
 * Test program using giafem and MFEM to verify nested Poisson problem in 3D using block system structure.
 *
 * We solve:
 *   -Laplace(psi) = g(r) in outer sphere (r < 1.2)
 *   -Laplace(phi) = psi(r) in inner sphere (r < 1.0)
 *
 * Manufactured solution:
 *   phi(r) = sin(pi r)
 *   psi(r) = pi^2 sin(pi r) - (2 pi / r) cos(pi r)
 *   g(r)   = pi^4 sin(pi r) - (4 pi^3 / r) cos(pi r)
 */

#include <mfem.hpp>
#include <giafem.hpp>
#include <cmath>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace giafem;


real_t phi_func(const Vector &coord)
{
    double r = coord.Norml2();
    return sin(M_PI * r);
}

real_t psi_func(const Vector &coord)
{
    double r = coord.Norml2();
    return M_PI*M_PI*sin(M_PI*r) - (2.0*M_PI/r)*cos(M_PI*r);
}

real_t g_func(const Vector &coord)
{
    double r = coord.Norml2();
    return pow(M_PI,4)*sin(M_PI*r) - (4.0*pow(M_PI,3)/r)*cos(M_PI*r);
}

/*real_t phi_func(const Vector &coord)
{
    const double x = coord[0], y = coord[1], z = coord[2];
    return -(x*x*x*x + y*y*y*y + z*z*z*z) / 72.0;
}

real_t psi_func(const Vector &coord)
{
    const real_t x = coord[0], y = coord[1], z = coord[2];
    return (x*x + y*y + z*z) / 6.0;
}

real_t g_func(const Vector &coord)
{
    return -1.0;
}*/


int main(int argc, char *argv[])
{
    const char *mesh_file = "mesh/Earth_space_nd.msh";
    real_t rel_tol = 1e-10;
    int order = 1;
    const char *device_config = "cpu";

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
    args.AddOption(&rel_tol, "-rt", "--rel-tol", "Relative tolerance for solving.");
    args.AddOption(&order, "-o", "--order", "Finite element order.");
    args.AddOption(&device_config, "-d", "--device", "Device configuration.");
    args.Parse();
    if (!args.Good()) { args.PrintUsage(cout); return 1; }
    args.PrintOptions(cout);

    Device device(device_config);
    device.Print();

    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();
    H1_FECollection fec_psi(order, dim);
    H1_FECollection fec_phi(order + 1, dim);
    FiniteElementSpace fes_psi(mesh, &fec_psi);

    Array<int> attr_cond = mesh->attributes;
    attr_cond.DeleteLast();
    SubMesh mesh_cond(SubMesh::CreateFromDomain(*mesh, attr_cond));

    FiniteElementSpace fes_phi(&mesh_cond, &fec_phi);
    FiniteElementSpace fes_psi_cond(&mesh_cond, &fec_psi);

    GridFunction psi_gf(&fes_psi);
    GridFunction phi_gf(&fes_phi);
    phi_gf = 0.0;
    psi_gf = 0.0;

    FunctionCoefficient phi_exact(phi_func);
    FunctionCoefficient psi_exact(psi_func);
    FunctionCoefficient g_exact(g_func);

    Array<int> bdr_marker_psi(mesh->bdr_attributes.Max());
    bdr_marker_psi = 0;
    bdr_marker_psi[mesh->bdr_attributes.Max() - 1] = 1;

    Array<int> bdr_marker_phi(mesh_cond.bdr_attributes.Max());
    bdr_marker_phi = 0;
    bdr_marker_phi[mesh_cond.bdr_attributes.Max() - 1] = 1;

    phi_gf.ProjectBdrCoefficient(phi_exact, bdr_marker_phi);
    psi_gf.ProjectBdrCoefficient(psi_exact, bdr_marker_psi);

    Array<int> block_offsets(3);
    block_offsets[0] = 0;
    block_offsets[1] = fes_phi.GetVSize(); //
    block_offsets[2] = fes_psi.GetVSize();
    block_offsets.PartialSum();

    std::cout << "***********************************************************\n";
    std::cout << "dim(phi) = " << block_offsets[1] - block_offsets[0] << "\n";
    std::cout << "dim(psi) = " << block_offsets[2] - block_offsets[1] << "\n";
    std::cout << "dim(phi+psi) = " << block_offsets.Last() << "\n";
    std::cout << "***********************************************************\n";

    MemoryType mt = device.GetMemoryType();
    BlockVector X(block_offsets, mt);
    BlockVector B(block_offsets, mt);
    X = 0.0;
    B = 0.0;
    phi_gf.GetTrueDofs(X.GetBlock(0));
    psi_gf.GetTrueDofs(X.GetBlock(1));

    auto zero = ConstantCoefficient(0.0);
    LinearForm *b0(new LinearForm);
    b0->Update(&fes_phi, B.GetBlock(0), 0); //
    b0->AddDomainIntegrator(new DomainLFIntegrator(zero));
    b0->Assemble();
    b0->SyncAliasMemory(B);

    LinearForm *b1(new LinearForm);
    b1->Update(&fes_psi, B.GetBlock(1), 0); //
    b1->AddDomainIntegrator(new DomainLFIntegrator(g_exact));
    b1->Assemble();
    b1->SyncAliasMemory(B);

    auto one = ConstantCoefficient(1.0);
    BilinearForm a00(&fes_phi);
    a00.AddDomainIntegrator(new DiffusionIntegrator(one));
    a00.Assemble();
    a00.EliminateEssentialBC(bdr_marker_phi, X.GetBlock(0), *b0);
    a00.Finalize();
    SparseMatrix &A00(a00.SpMat()); //
    A00.PrintInfo(mfem::out);
    BilinearForm a11(&fes_psi);
    a11.AddDomainIntegrator(new DiffusionIntegrator(one));
    a11.Assemble();
    a11.EliminateEssentialBC(bdr_marker_psi, X.GetBlock(1), *b1); //
    a11.Finalize();
    SparseMatrix &A11(a11.SpMat());
    A11.PrintInfo(mfem::out);
    ExtTrialMixedBilinearForm a01(&fes_psi, &fes_phi, &fes_psi_cond, &mesh_cond);
    a01.AddDomainIntegrator(new MixedScalarMassIntegrator(one));
    a01.Assemble();
    a01.EliminateTrialEssentialBC(bdr_marker_psi, X.GetBlock(1), *b1); //
    a01.EliminateTestEssentialBC(bdr_marker_phi); //
    a01.Finalize();
    SparseMatrix &A01(a01.SpMat());
    A01.PrintInfo(mfem::out);

    b0->SyncAliasMemory(B);
    b1->SyncAliasMemory(B);

    BlockOperator Op(block_offsets); //

    Op.SetBlock(0,0, &A00);
    Op.SetBlock(0,1, &A01);
    Op.SetBlock(1,1, &A11);


    GSSmoother prec_phi(A00);
    GSSmoother prec_psi(A11);
    BlockDiagonalPreconditioner prec(block_offsets);
    prec.SetDiagonalBlock(0, &prec_phi);
    prec.SetDiagonalBlock(1, &prec_psi);

    GMRESSolver solver;
    solver.SetRelTol(rel_tol);
    solver.SetMaxIter(3000);
    solver.SetPrintLevel(1);
    solver.SetOperator(Op);
    solver.SetPreconditioner(prec);
    solver.Mult(B, X);

    if (device.IsEnabled()) { X.HostRead(); }

    //phi_gf.SetFromTrueDofs(X.GetBlock(0));
    //psi_gf.SetFromTrueDofs(X.GetBlock(1));
    phi_gf.MakeRef(&fes_phi, X.GetBlock(0), 0);
    psi_gf.MakeRef(&fes_psi, X.GetBlock(1), 0);

    double phi_err = phi_gf.ComputeL2Error(phi_exact) / phi_gf.Norml2();
    double psi_err = psi_gf.ComputeL2Error(psi_exact)/ psi_gf.Norml2();

    cout << "L2 error in phi: " << phi_err << endl;
    cout << "L2 error in psi: " << psi_err << endl;

    delete mesh; 
    delete b0;
    delete b1;
    return 0;
}

