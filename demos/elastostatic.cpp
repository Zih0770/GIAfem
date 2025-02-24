#include "mfem.hpp"
#include "giafem.hpp"
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>

using namespace std;
using namespace mfem;


int main(int argc, char *argv[])
{
   string mesh_file = "mesh/2S.msh";
   bool static_cond = false;
   bool pa = false;
   bool fa = false;
   const char *device_config = "cpu";
   bool visualization = true;
   bool algebraic_ceed = false;
   int order = 1;
   int adaptive_int = 0;
   string parameter_file = "data/prem.200.noiso";
   string output_file = "data/output.txt";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc", "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&adaptive_int, "-a", "--adaptive", "Adaptive meshing");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa", "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa", "--no-full-assembly", "Enable Full Assembly.");
   args.AddOption(&device_config, "-d", "--device", "Device configuration string, see Device::Configure().");
#ifdef MFEM_USE_CEED
   args.AddOption(&algebraic_ceed, "-a", "--algebraic", "-no-a", "--no-algebraic", "Use algebraic Ceed solver");
#endif
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis", "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&parameter_file, "-p", "--para", "Parameter file to use.");
   args.AddOption(&output_file, "-f", "--output-file",
               "Output file name for saving data");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);
   bool adaptive = static_cast<bool>(adaptive_int);
   Device device(device_config);
   device.Print(); 



   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();
   int num_attributes = mesh->attributes.Max();
   std::cout<<"Number of attributes: "<<num_attributes<<std::endl;
   //if (mesh->NURBSext)
   //{
   //   mesh->DegreeElevate(order, order);
   //}

   if (!adaptive)
   {
      int ref_levels =
         (int)floor(log(50000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   cout << "Volume attributes in the mesh: ";
   for (int i = 0; i < mesh->attributes.Size(); i++)
   {
       cout << mesh->attributes[i] << " ";
   }
   cout << endl;
   cout << "Boundary attributes: ";
   for (int i = 0; i < mesh->bdr_attributes.Size(); i++)
   {
       cout << mesh->bdr_attributes[i] << " ";
   }
   cout << endl;
 
   
   FiniteElementCollection *fec;
   bool delete_fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
      delete_fec = true;
   }
   else if (mesh->GetNodes())
   {
      fec = mesh->GetNodes()->OwnFEC();
      delete_fec = false;
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
      delete_fec = true;
   }
   
   //
   //FiniteElementSpace *fespace;
   //if (mesh->NURBSext)
   //{
   //   fec = NULL;
   //   fespace = mesh->GetNodes()->FESpace();
   //}
   //else
   //{
   //   fec = new H1_FECollection(order, dim);
   //   fespace = new FiniteElementSpace(mesh, fec, dim);
   //}
   
   FiniteElementSpace fespace(mesh, fec, dim);
   cout << "Number of finite element unknowns: " << fespace.GetTrueVSize()
        << endl << "Assembling: " << flush;

   //
   //Array<int> boundary_dofs;
   //fespace.GetBoundaryTrueDofs(boundary_dofs);
   Array<int> ess_tdof_list;
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   if (mesh->bdr_attributes.Size())
   {
      ess_bdr = 0;
      //ess_bdr[0] = 0;
      //ess_bdr[num_attributes - 1] = 1;
      //
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   //
   GridFunction x(&fespace);
   x = 0.0;

   //ConstantCoefficient u_dirichlet(1.0);
   //x.ProjectBdrCoefficient(u_dirichlet, ess_bdr);
   //
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   //
   //VectorArrayCoefficient f(dim);
   //for (int i = 0; i < dim-1; i++)
   //{
   //   f.Set(i, new ConstantCoefficient(0.0));
   //}
   //{
   //   Vector pull_force(mesh->bdr_attributes.Max());
   //   pull_force = 0.0;
   //   pull_force(1) = -1.0e-2;
   //   f.Set(dim-1, new PWConstCoefficient(pull_force));
   //}
   //


   double G = 6.6743e-11 * 1e-18;
   double R = 6.38;
   double rho0 = 13000 * 1e+18;
   double rho1 = 2835 * 1e+18;

   auto inside_coef = [=](const Vector &x) -> double
   {
      double r = x.Norml2();
      double rho = rho0 - (rho0 - rho1) * r / R;
      return -4.0 * M_PI * G * rho;
   };

   //giafem::parse parser;
   //std::vector<std::vector<double>> data = parser.properties_1d(parameter_file);

   //std::vector<std::pair<double, double>> radius_rho;
   //std::vector<std::pair<double, double>> radius_lamb;
   //std::vector<std::pair<double, double>> radius_mu;
   //for (const auto &row : data)
   //{
   //   radius_rho.emplace_back(row[0] * 1e-6, row[1] * 1e+18);
   //   radius_lamb.emplace_back(row[0] * 1e-6, (row[4] - 2/3 *row[5]) * 1e+9 * 1e+6);
   //   radius_mu.emplace_back(row[0] * 1e-6, row[5] * 1e+9 * 1e+6);
   //}

   //giafem::interp interpolator;
   //mfem::Array<mfem::FunctionCoefficient *> rho_coefficients =
   //   interpolator.PWCoef_1D(radius_rho, num_attributes, "linear");
   //mfem::Array<mfem::FunctionCoefficient *> lamb_coefficients =
   //   interpolator.PWCoef_1D(radius_lamb, num_attributes, "linear");
   //mfem::Array<mfem::FunctionCoefficient *> mu_coefficients =
   //   interpolator.PWCoef_1D(radius_mu, num_attributes, "linear");

   Array<int> pw_attributes;
   //Array<Coefficient *> pw_coefficients_rho;
   //Array<Coefficient *> pw_coefficients_lamb;
   //Array<Coefficient *> pw_coefficients_mu;
   Array<FunctionCoefficient *> pw_coefficients_rho;
   Array<FunctionCoefficient *> pw_coefficients_lamb;
   Array<FunctionCoefficient *> pw_coefficients_mu;

   std::vector<std::pair<double, double>> radius_pair;
   std::vector<std::pair<double, double>> rho_pair;
   std::vector<std::pair<double, double>> lamb_pair;
   std::vector<std::pair<double, double>> mu_pair;
   radius_pair.emplace_back(0, 3.48);
   radius_pair.emplace_back(3.48, 6.38);
   rho_pair.emplace_back(13000 * 1e+18, 10000 * 1e+18);
   rho_pair.emplace_back(5500 * 1e+18, 3300 * 1e+18);
   lamb_pair.emplace_back(1100 * 1e+9 * 1e+6, 500 * 1e+9 * 1e+6);
   lamb_pair.emplace_back(90 * 1e+9 * 1e+6, 40 * 1e+9 * 1e+6);
   mu_pair.emplace_back(500 * 1e+9 * 1e+6, 0 * 1e+9 * 1e+6);
   mu_pair.emplace_back(26 * 1e+9 * 1e+6, 36 * 1e+9 * 1e+6);
   //for (int i = 1; i < num_attributes + 1; i++){
   //    pw_attributes.Append(i);
   //    pw_coefficients_rho.Append(rho_coefficients[i - 1]);
   //    pw_coefficients_lamb.Append(lamb_coefficients[i - 1]);
   //    pw_coefficients_mu.Append(mu_coefficients[i - 1]);
   //}

   for (int i = 0; i < num_attributes; i++)
   {
       pw_attributes.Append(i + 1); 

       double r0 = radius_pair[i].first;
       double r1 = radius_pair[i].second;
       double rho0 = rho_pair[i].first;
       double rho1 = rho_pair[i].second;
       std::cout<<r0<<" "<<r1<<" "<<rho0<<" "<<rho1<<endl;
       // Define linear function for rho in this layer
       pw_coefficients_rho.Append(new FunctionCoefficient([=](const Vector &x) -> double
       {
           double r = x.Norml2(); // Radius
          // std::cout<<r<<" "<<r0<<" "<<r1<<endl;
           //std::cout<<"density: "<<rho0<<" "<<rho1<<endl;
           if (r >= r0 && r <= r1)
           {
               return rho0 + (rho1 - rho0) * (r - r0) / (r1 - r0); // Linear interpolation
           }
           return 0.0;
       }));

       double lamb0 = lamb_pair[i].first;
       double lamb1 = lamb_pair[i].second;

       pw_coefficients_lamb.Append(new FunctionCoefficient([=](const Vector &x) -> double
       {
           double r = x.Norml2(); // Radius
           if (r >= r0 && r <= r1)
           {
               return lamb0 + (lamb1 - lamb0) * (r - r0) / (r1 - r0); // Linear interpolation
           }
           return 0.0;
       }));

       double mu0 = lamb_pair[i].first;
       double mu1 = lamb_pair[i].second;

       pw_coefficients_mu.Append(new FunctionCoefficient([=](const Vector &x) -> double
       {
           double r = x.Norml2(); // Radius
           if (r >= r0 && r <= r1)
           {
               return mu0 + (mu1 - mu0) * (r - r0) / (r1 - r0); // Linear interpolation
           }
           return 0.0;
       }));


   }
      
   PWCoefficient rho_func(pw_attributes, pw_coefficients_rho);
   PWCoefficient lamb_func(pw_attributes, pw_coefficients_lamb);
   PWCoefficient mu_func(pw_attributes, pw_coefficients_mu);

   auto scalar_fec = L2_FECollection(order, dim);
   FiniteElementSpace scalarFES(mesh, &scalar_fec, 1);
   GridFunction scalar(&scalarFES); 
   scalar.ProjectCoefficient(rho_func);


   LinearForm b(&fespace);
   //FunctionCoefficient f_in(inside_coef);
   FunctionCoefficient loading([](const Vector &x) -> double {
        double r = x.Norml2(); // Radius
        double theta = acos(x(2) / r); // Polar angle
        return -30 * 1e+6 * 1e+6 * abs(cos(theta));
    });
   b.AddBoundaryIntegrator(new BoundaryLFIntegrator(loading));
   cout << "r.h.s. ... " << flush;
   b.Assemble();



   BilinearForm *a = new BilinearForm(&fespace);
   if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   if (fa)
   {
      a->SetAssemblyLevel(AssemblyLevel::FULL);
      a->EnableSparseMatrixSorting(Device::IsEnabled());
   }
   BilinearFormIntegrator *integ = new ElasticityIntegrator(lamb_func,mu_func);
   a->AddDomainIntegrator(integ);
   if (static_cond) { a->EnableStaticCondensation(); }
  
   const int tdim = dim*(dim+1)/2;
   FiniteElementSpace flux_fespace(mesh, fec, tdim);
   ZienkiewiczZhuEstimator estimator(*integ, x, flux_fespace);
  
   ThresholdRefiner refiner(estimator);
   refiner.SetTotalErrorFraction(0.7);
   
   a->Assemble();
  
   //a.EliminateEssentialBC(ess_tdof_list, x, *b);
   //
   //Vector lambda(mesh->attributes.Max());
   //lambda = 1.0;
   //lambda(0) = lambda(1)*50;
   //PWConstCoefficient lambda_func(lambda);
   //Vector mu(mesh->attributes.Max());
   //mu = 1.0;
   //mu(0) = mu(1)*50;
   //PWConstCoefficient mu_func(mu);
   //

   //
   OperatorPtr A; 
   //SparseMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   cout << "Size of linear system: " << A->Height() << endl;

   //
   //GSSmoother M(A);
   //PCG(A, M, B, X, 1, 200, 1e-12, 0.0);
   if (!pa)
   {
#ifndef MFEM_USE_SUITESPARSE
      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 1, 500, 1e-8, 0.0);


      if (adaptive)
      {
         // Adaptive meshing parameters
         double desired_error = 0.05; // Desired error threshold
         int max_refinements = 5;     // Max number of adaptive refinement steps
         int refinement_step = 0;
         double error = std::numeric_limits<double>::max();
     
         std::cout << "Refinement step " << refinement_step << ", Total Error: " << error << std::endl;

         for (int refinement_step = 1; refinement_step <= max_refinements; refinement_step++)
         {
            refiner.Apply(*mesh);
            if (refiner.Stop())
            {
               cout << "Stopping criterion satisfied. Stop." << endl;
               break;
            }
            // Adaptive refinement based on error estimates

            // Update the finite element space and solution for the new, refined mesh
            fespace.Update();
            x.Update(); // Project the old solution onto the new, refined space
         
            a->Update();
            b.Update();

            b.Assemble();
	        a->Assemble();

	        Array<int> ess_bdr(mesh->bdr_attributes.Max());
            ess_bdr = 1;
            Array<int> ess_tdof_list;
            x.ProjectBdrCoefficient(zero, ess_bdr);
            fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

	        OperatorPtr A;
            Vector B, X;
            a->FormLinearSystem(ess_tdof_list, x, b, A, X, B);
            GSSmoother M((SparseMatrix&)(*A));

            PCG(*A, M, B, X, 1, 200, 1e-12, 0.0);
            cout << "Solution done" << endl;
            a->RecoverFEMSolution(X, b, x);

            std::cout << "Refinement step " << refinement_step << ", Total Error: " << error << std::endl;
	     }
      }
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
            ceed::AlgebraicSolver M(*a, ess_tdof_list);
            PCG(*A, M, B, X, 1, 400, 1e-12, 0.0);
         }
         else
         {
            OperatorJacobiSmoother M(*a, ess_tdof_list);
            PCG(*A, M, B, X, 1, 400, 1e-12, 0.0);
         }
      }
      else
      {
   //GSSmoother M((SparseMatrix&)(*A));
   //PCG(*A, M, B, X, 1, 200, 1e-12, 0.0);
         CG(*A, B, X, 1, 400, 1e-12, 0.0);
      }
   }
   

   cout << "Solution done" << endl;
   a->RecoverFEMSolution(X, b, x);
   
   //
   if (!mesh->NURBSext)
   {
      mesh->SetNodalFESpace(&fespace);
   }
   //
   //
   GridFunction *nodes = mesh->GetNodes();
   *nodes += x;
   x *= -1;
   //
   ofstream mesh_ofs("0.mesh");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);
   ofstream sol_ofs("0.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }
   
   delete a;
   //delete b;
   if (delete_fec)
   {
      delete fec;
   }
   delete mesh;

   /*

   ofstream mesh_ofs("0.mesh");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);
   ofstream sol_ofs("0.gf");
   sol_ofs.precision(8);
   scalar.Save(sol_ofs);
   */

   return 0;
}
