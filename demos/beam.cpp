#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   int order = 1;
   bool static_cond = false;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
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

   int nx = 64, ny = 16, nz = 16;
   double lx = 10.0, ly = 2.0, lz = 2.0;
   Mesh *mesh = new Mesh(Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON, lx, ly, lz));
   int dim = mesh->Dimension();

   for (int i = 0; i < mesh->GetNE(); i++)
   {
      Element *el = mesh->GetElement(i);
      Array<int> vertices;
      el->GetVertices(vertices);
      double x_centroid = 0.0;
      for (int j = 0; j < vertices.Size(); j++)
      {
         x_centroid += mesh->GetVertex(vertices[j])[0];
      }
      x_centroid /= vertices.Size();
      el->SetAttribute(x_centroid < lx / 2.0 ? 1 : 2); // Divide at x = lx / 2.
   }
   for (int i = 0; i < mesh->GetNBE(); i++)
   {
       Element *bdr_el = mesh->GetBdrElement(i);
       Array<int> vertices;
       bdr_el->GetVertices(vertices);

       bool is_x0 = true, is_xL = true;
       for (int j = 0; j < vertices.Size(); j++)
       {
           const double *v = mesh->GetVertex(vertices[j]);
           if (fabs(v[0]) > 1e-8) is_x0 = false;   // Not on x = 0
           if (fabs(v[0] - lx) > 1e-8) is_xL = false; // Not on x = lx
       }

       if (is_x0) bdr_el->SetAttribute(1); // Fixed boundary at x = 0
       else if (is_xL) bdr_el->SetAttribute(2); // Pull boundary at x = lx
       else bdr_el->SetAttribute(3); // Other boundaries
   }
   mesh->SetAttributes();

   {
      int ref_levels =
         (int)floor(log(5000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec, dim);
   cout << "Number of finite element unknowns: " << fespace->GetTrueVSize() << endl;

   Array<int> ess_tdof_list, ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   std::cout<<"boundary attributes: "<<mesh->bdr_attributes.Max()<<std::endl;
   VectorArrayCoefficient f(dim);
   for (int i = 0; i < dim-1; i++)
   {
      f.Set(i, new ConstantCoefficient(0.0));
   }
   {
      Vector pull_force(mesh->bdr_attributes.Max());
      pull_force = 0.0;
      pull_force(1) = -3.0e-2;
      f.Set(dim-1, new PWConstCoefficient(pull_force));
   }

   LinearForm *b = new LinearForm(fespace);
   b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
   b->Assemble();

   GridFunction x(fespace);
   x = 0.0;

   Vector lambda(mesh->attributes.Max());
   lambda = 1.0;
   lambda(0) = lambda(1)*50;
   PWConstCoefficient lambda_func(lambda);
   Vector mu(mesh->attributes.Max());
   mu = 1.0;
   mu(0) = mu(1)*50;
   PWConstCoefficient mu_func(mu);

   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new ElasticityIntegrator(lambda_func,mu_func));

   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   SparseMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
   cout << "done." << endl;

   cout << "Size of linear system: " << A.Height() << endl;

#ifndef MFEM_USE_SUITESPARSE
   GSSmoother M(A);
   PCG(A, M, B, X, 1, 500, 1e-8, 0.0);
#else
   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   umf_solver.SetOperator(A);
   umf_solver.Mult(B, X);
#endif

   a->RecoverFEMSolution(X, *b, x);

   {
      mesh->SetNodalFESpace(fespace);
      GridFunction *nodes = mesh->GetNodes();
      *nodes += x;
      x *= -1;
      ofstream mesh_ofs("0.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
      ofstream sol_ofs("0.gf");
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }
/*
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }
*/
   // 16. Free the used memory.
   delete a;
   delete b;
   delete fespace;
   delete fec;
   delete mesh;

   return 0;
}
