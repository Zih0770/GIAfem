#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
  Mpi::Init(argc, argv);
  Hypre::Init();


  string mesh_file = "../data/star.mesh";
  int order = 1;

  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
  args.ParseCheck();

  Mesh serial_mesh(mesh_file);
  ParMesh mesh(MPI_COMM_WORLD, serial_mesh); //
  serial_mesh.Clear(); //
  mesh.UniformRefinement();

  H1_FECollection fec(order, mesh.Dimension());
  FiniteElementSpace fespace(&mesh, &fec);  
  HYPRE_BigInt total_num_dofs = fespace.GlobalTrueVSize(); //
							   
  if (Mpi::Root()) //
  {
    cout << "Number of unknowns: " << total_num_dofs << endl;
  }

  Array<int> boundary_dofs;
  fespace.GetBoundaryTrueDofs(boundary_dofs);

  GridFunction x(&fespace);
  x = 0.0;

  ConstantCoefficient one(1.0);
  ParLinearForm b(&fespace); //
  b.AddDomainIntegrator(new DomainLFIntegrator(one));
  b.Assemble();

  ParBilinearForm a(&fespace);
  a.AddDomainIntegrator(new DiffusionIntegrator);
  a.Assemble();

  HypreParMatrix A;
  Vector B, X;
  a.FormLinearSystem(boundary_dofs, x, b, A, X, B);

  HypreBoomerAMG M(A); //GSSmoother M(A);
  CGSolver cg(MPI_COMM_WORLD); //PCG(A, M, B, X, 1, 200, 1e-12, 0.0);
  cg.SetRelTol(1e-12);
  cg.SetMaxIter(2000);
  cg.SetPrintLevel(1);
  cg.SetPreconditioner(M);
  cg.SetOperator(A);
  cg.Mult(B, X);

  a.RecoverFEMSolution(X, b, x);
  x.Save("sol");
  mesh.Save("mesh");  
 
  return 0;


}
