#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   //Mpi::Init(argc, argv);
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool static_cond = false;
   bool pa = false;
   bool fa = false;
   const char *device_config = "cpu";
   bool visualization = true;
   bool algebraic_ceed = false;
   //bool amg_elast = 0;
   //bool reorder_space = false;


   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   //args.ParseCheck();
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa",
                  "--no-full-assembly", "Enable Full Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
#ifdef MFEM_USE_CEED
   args.AddOption(&algebraic_ceed, "-a", "--algebraic",
                  "-no-a", "--no-algebraic",
                  "Use algebraic Ceed solver");
#endif
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
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

   // 3. Read the serial mesh from the given mesh file.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   {
      int ref_levels =
         (int)floor(log(10000./mesh.GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   // 4. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh once in parallel to increase the resolution.
   ParMesh pmesh(MPI_COMM_WORLD, serial_mesh);
   mesh.Clear(); // the serial mesh is no longer needed

   //
   {
      int par_ref_levels = 1;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }
   //
   // 5. Define a finite element space on the mesh. Here we use H1 continuous
   //    high-order Lagrange finite elements of the given order.
   //H1_FECollection fec(order, mesh.Dimension());
   FiniteElementCollection *fec;
   bool delete_fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
      delete_fec = true;
   }
   else if (pmesh.GetNodes())
   {
      fec = pmesh.GetNodes()->OwnFEC();
      delete_fec = false;
      if (myid == 0)
      {
         cout << "Using isoparametric FEs: " << fec->Name() << endl;
      }
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
      delete_fec = true;
   }
   //
   ParFiniteElementSpace *fespace;
   const bool use_nodal_fespace = pmesh->NURBSext && !amg_elast;
   if (use_nodal_fespace)
   {
      fec = NULL;
      fespace = (ParFiniteElementSpace *)pmesh->GetNodes()->FESpace();
   }
   else
   {
      fec = new H1_FECollection(order, dim);
      if (reorder_space)
      {
         fespace = new ParFiniteElementSpace(pmesh, fec, dim, Ordering::byNODES);
      }
      else
      {
         fespace = new ParFiniteElementSpace(pmesh, fec, dim, Ordering::byVDIM);
      }
   }
   //
   ParFiniteElementSpace fespace(&mesh, fec);
   HYPRE_BigInt total_num_dofs = fespace.GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of unknowns: " << total_num_dofs << endl;
   }

   // 6. Extract the list of all the boundary DOFs. These will be marked as
   //    Dirichlet in order to enforce zero boundary conditions.
   Array<int> ess_tdof_list;
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 7. Define the solution x as a finite element grid function in fespace. Set
   //    the initial guess to zero, which also sets the boundary conditions.
   ParGridFunction x(&fespace);
   x = 0.0;

   // 8. Set up the linear form b(.) corresponding to the right-hand side.
   ConstantCoefficient one(1.0);
   ParLinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(one)); // a->AddDomainIntegrator(new ElasticityIntegrator(lambda_func,mu_func));
   //
   ...
   f.Set(dim-1, new PWConstCoefficient(pull_force));
   //
   b.Assemble();

   // 9. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
   //...(vector coef)
   ParBilinearForm a(&fespace);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   if (fa)
   {
      a.SetAssemblyLevel(AssemblyLevel::FULL);
      a.EnableSparseMatrixSorting(Device::IsEnabled());
   }
   a.AddDomainIntegrator(new DiffusionIntegrator(one));

   if (static_cond) { a.EnableStaticCondensation(); }
   a.Assemble();

   // 10. Form the linear system A X = B. This includes eliminating boundary
   //     conditions, applying AMR constraints, parallel assembly, etc.
   OperatorPtr A; //HypreParMatrix A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   if (myid == 0)
   {
      cout << "done." << endl;
      cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
   }

   // 11. Solve the system using PCG with hypre's BoomerAMG preconditioner.
   //HypreBoomerAMG M(A);
   //CGSolver cg(MPI_COMM_WORLD);
   //cg.SetRelTol(1e-12);
   //cg.SetMaxIter(2000);
   //cg.SetPrintLevel(1);
   //cg.SetPreconditioner(M);
   //cg.SetOperator(A);
   //cg.Mult(B, X);
   //
   if (amg_elast && !a->StaticCondensationIsEnabled())
   {
      amg->SetElasticityOptions(fespace);
   }
   else
   {
      amg->SetSystemsOptions(dim, reorder_space);
   }
   HyprePCG *pcg = new HyprePCG(A);
   pcg->SetTol(1e-8);
   pcg->SetMaxIter(500);
   pcg->SetPrintLevel(2);
   pcg->SetPreconditioner(*amg);
   pcg->Mult(B, X);
   //   
   Solver *prec = NULL;
   if (pa)
   {
      if (UsesTensorBasis(fespace))
      {
         if (algebraic_ceed)
         {
            prec = new ceed::AlgebraicSolver(a, ess_tdof_list);
         }
         else
         {
            prec = new OperatorJacobiSmoother(a, ess_tdof_list);
         }
      }
   }
   else
   {
      prec = new HypreBoomerAMG;
   }
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   if (prec) { cg.SetPreconditioner(*prec); }
   cg.SetOperator(*A);
   cg.Mult(B, X);
   delete prec;

   // 12. Recover the solution x as a grid function and save to file. The output
   //     can be viewed using GLVis as follows: "glvis -np <np> -m mesh -g sol"
   a.RecoverFEMSolution(X, b, x);
   //
   if (!use_nodal_fespace)
   {
      pmesh->SetNodalFESpace(fespace);
   }
   //
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << x << flush;
   }

   if (delete_fec)
   {
      delete fec;
   }

   return 0;
}
