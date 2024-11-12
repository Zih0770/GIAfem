#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command line options.
   string mesh_file = "../data/star.mesh"; //const char *mesh_file = "../data/star.mesh";
					   //bool static_cond = false;
   //bool pa = false;
   // bool fa = false;
   //const char *device_config = "cpu";
   bool visualization = true;
   //bool algebraic_ceed = false;
   int order = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.ParseCheck();
   //args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc", "--no-static-condensation", "Enable static condensation.");
   //args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   //args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa",
                  "--no-full-assembly", "Enable Full Assembly.");
   //args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
//#ifdef MFEM_USE_CEED
//   args.AddOption(&algebraic_ceed, "-a", "--algebraic", "-no-a", "--no-algebraic",
                  "Use algebraic Ceed solver");
//#endif
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
   
   Device device(device_config);
   device.Print(); 

   // 2. Read the mesh from the given mesh file, and refine once uniformly.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   mesh.UniformRefinement();

   if (mesh->NURBSext)
   {
      mesh->DegreeElevate(order, order);
   }

   //{
   //   int ref_levels =
   //      (int)floor(log(50000./mesh.GetNE())/log(2.)/dim);
   //   for (int l = 0; l < ref_levels; l++)
   //   {
   //      mesh.UniformRefinement();
   //   }
   //}

   // 3. Define a finite element space on the mesh. Here we use H1 continuous
   //    high-order Lagrange finite elements of the given order.
   //H1_FECollection fec(order, mesh.Dimension());
   FiniteElementCollection *fec;
   bool delete_fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
      delete_fec = true;
   }
   else if (mesh.GetNodes())
   {
      fec = mesh.GetNodes()->OwnFEC();
      delete_fec = false;
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
      delete_fec = true;
   }
   
   //
   FiniteElementSpace *fespace;
   if (mesh->NURBSext)
   {
      fec = NULL;
      fespace = mesh->GetNodes()->FESpace();
   }
   else
   {
      fec = new H1_FECollection(order, dim);
      fespace = new FiniteElementSpace(mesh, fec, dim);
   }
   cout << "Number of finite element unknowns: " << fespace->GetTrueVSize()
        << endl << "Assembling: " << flush;
   //
   FiniteElementSpace fespace(&mesh, fec);
   cout << "Number of unknowns: " << fespace.GetTrueVSize() << endl;

   // 4. Extract the list of all the boundary DOFs. These will be marked as
   //    Dirichlet in order to enforce zero boundary conditions.
   //Array<int> boundary_dofs;
   //fespace.GetBoundaryTrueDofs(boundary_dofs);
   Array<int> ess_tdof_list;
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      //
      ess_bdr = 0;
      ess_bdr[0] = 1;
      //
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 5. Define the solution x as a finite element grid function in fespace. Set
   //    the initial guess to zero, which also sets the boundary conditions.
   GridFunction x(&fespace);
   x = 0.0;

   // 6. Set up the linear form b(.) corresponding to the right-hand side.
   ConstantCoefficient one(1.0);
   //
   VectorArrayCoefficient f(dim);
   for (int i = 0; i < dim-1; i++)
   {
      f.Set(i, new ConstantCoefficient(0.0));
   }
   {
      Vector pull_force(mesh->bdr_attributes.Max());
      pull_force = 0.0;
      pull_force(1) = -1.0e-2;
      f.Set(dim-1, new PWConstCoefficient(pull_force));
   }
   //
   LinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   //
   b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
   cout << "r.h.s. ... " << flush;
   //
   b.Assemble();

   // 7. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
   BilinearForm a(&fespace);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   if (fa)
   {
      a.SetAssemblyLevel(AssemblyLevel::FULL);
      a.EnableSparseMatrixSorting(Device::IsEnabled());
   }
   a.AddDomainIntegrator(new DiffusionIntegrator(one));

   if (static_cond) { a.EnableStaticCondensation(); }
   a.Assemble();
   //
   Vector lambda(mesh->attributes.Max());
   lambda = 1.0;
   lambda(0) = lambda(1)*50;
   PWConstCoefficient lambda_func(lambda);
   Vector mu(mesh->attributes.Max());
   mu = 1.0;
   mu(0) = mu(1)*50;
   PWConstCoefficient mu_func(mu);
   //

   // 8. Form the linear system A X = B. This includes eliminating boundary
   //    conditions, applying AMR constraints, and other transformations.
   OperatorPtr A; //SparseMatrix A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   cout << "Size of linear system: " << A->Height() << endl;

   // 9. Solve the system using PCG with symmetric Gauss-Seidel preconditioner.
   //GSSmoother M(A);
   //PCG(A, M, B, X, 1, 200, 1e-12, 0.0);
   if (!pa)
   {
#ifndef MFEM_USE_SUITESPARSE
      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 1, 200, 1e-12, 0.0);
#else
      UMFPackSolver umf_solver;
      umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      umf_solver.SetOperator(*A);
      umf_solver.Mult(B, X);
#endif
   }
   else
   {
      if (UsesTensorBasis(fespace))
      {
         if (algebraic_ceed)
         {
            ceed::AlgebraicSolver M(a, ess_tdof_list);
            PCG(*A, M, B, X, 1, 400, 1e-12, 0.0);
         }
         else
         {
            OperatorJacobiSmoother M(a, ess_tdof_list);
            PCG(*A, M, B, X, 1, 400, 1e-12, 0.0);
         }
      }
      else
      {
         CG(*A, B, X, 1, 400, 1e-12, 0.0);
      }
   }   


   // 10. Recover the solution x as a grid function and save to file. The output
   //     can be viewed using GLVis as follows: "glvis -m mesh.mesh -g sol.gf"
   a.RecoverFEMSolution(X, b, x);
   //
   if (!mesh->NURBSext)
   {
      mesh->SetNodalFESpace(fespace);
   }
   //
   //
   GridFunction *nodes = mesh->GetNodes();
   *nodes += x;
   x *= -1;
   //
   ofstream mesh_ofs("mesh.mesh");
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << x << flush;
   }

   if (delete_fec)
   {
      delete fec;
   }

   return 0;
}
