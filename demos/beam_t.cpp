#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   int order = 1;
   double t_final = 1.0;
   double dt = 0.01;
   bool static_cond = false;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&t_final, "-tf", "--t-final", "Final simulation time.");
   args.AddOption(&dt, "-dt", "--time-step", "Time step size.");
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

   Vector material_density(mesh->attributes.Max());
   material_density = 1.0;
   material_density(0) = 10.0;
   PWConstCoefficient density_coeff(material_density);

   BilinearForm mass_form(fespace);
   mass_form.AddDomainIntegrator(new MassIntegrator(density_coeff));
   mass_form.Assemble();
   SparseMatrix M;
   mass_form.FormSystemMatrix(ess_tdof_list, M);

   Vector lambda(mesh->attributes.Max()), mu(mesh->attributes.Max());
   lambda = 1.0; mu = 1.0;
   lambda(0) = lambda(1) * 50;
   mu(0) = mu(1) * 50;
   PWConstCoefficient lambda_func(lambda), mu_func(mu);

   BilinearForm stiffness_form(fespace);
   stiffness_form.AddDomainIntegrator(new ElasticityIntegrator(lambda_func, mu_func));
   stiffness_form.Assemble();
   SparseMatrix K;
   stiffness_form.FormSystemMatrix(ess_tdof_list, K);

   // Time-stepping parameters for Newmark-beta method.
   double beta = 0.25, gamma = 0.5; // Newmark-beta coefficients.
   double t = 0.0;

   GridFunction x(fespace), v(fespace), a(fespace);
   x = 0.0; v = 0.0; a = 0.0;

   Vector X;

   VectorArrayCoefficient f(dim);
   for (int i = 0; i < dim - 1; i++) { f.Set(i, new ConstantCoefficient(0.0)); }
   Vector pull_force(mesh->bdr_attributes.Max());
   pull_force = 0.0;
   pull_force(1) = -3.0e-2; // Pull force in the z-direction.
   f.Set(dim - 1, new PWConstCoefficient(pull_force));

   LinearForm b(fespace);
   b.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
   b.Assemble();

   socketstream sol_sock("localhost", 19916); // Connect to GLVis server.
   sol_sock.precision(8);
   mesh->SetNodalFESpace(fespace);
   GridFunction *nodes = mesh->GetNodes();

   while (t < t_final)
   {
      t += dt;

      SparseMatrix M_eff = M;
      M_eff *= 1.0 / (beta * dt * dt);
      M_eff.Add(1.0, K);
      GSSmoother M_solver(M_eff);

      Vector B = b;
      Vector temp(fespace->GetTrueVSize()), temp_v(fespace->GetTrueVSize()), temp_a(fespace->GetTrueVSize());
      M.Mult(x, temp);
      B.Add(1.0 / (beta * dt * dt), temp);
      M.Mult(v, temp_v);
      B.Add(1.0 / (beta * dt), temp_v);
      M.Mult(a, temp_a);
      B.Add((1.0 - 2.0 * beta) / (2.0 * beta), temp_a);
    
      PCG(M_eff, M_solver, B, X, 1, 500, 1e-8, 0.0);

      v.Add((1.0 - gamma) * dt, a);
      a = X;
      a.Add(-1.0, x);
      a.Add(-dt, v);
      a *= (1.0 / (beta * dt * dt));
      a.Add(-((1.0 - 2.0 * beta) / (2.0 * beta)), a);
      v.Add(gamma * dt, a);

      x = X;

      *nodes += x;
      sol_sock << "solution\n" << *mesh << x << flush;
      *nodes -= x;
   }

   ofstream mesh_ofs("0.mesh"), sol_ofs("0.gf");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   // Cleanup.
   delete fec;
   delete fespace;
   delete mesh;

   return 0;
}
