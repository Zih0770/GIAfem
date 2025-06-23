#include <mfem.hpp>
#include <giafem.hpp>
#include <iostream>
#include <cmath>

using namespace std;
using namespace mfem;
using namespace giafem;

real_t rho_func(const Vector &coord);
real_t mu_func(const Vector &coord);
real_t lamb_func(const Vector &coord);
real_t loading_func(const Vector &coord);

int main(int argc, char *argv[])
{
    const char *mesh_file = "mesh/Earth_space.msh";
    const char *output_name = "data/ex2";
    const char *elasticity_model = "linear";
    real_t rel_tol = 1e-10;
    int order = 1;
    int lMax = 10;
    int ref_levels = -1;
    bool static_cond = false;
    bool pa = false;
    const char *device_config = "cpu";
    bool visualization = false;

    real_t coeff_rho = - 4.0 * M_PI * Constants::G;


    //Parsing
    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
            "Mesh file to use.");
    args.AddOption(&elasticity_model, "-em", "--elasticity-model",
            "Elasticity model to use: linear, neo-hookean, etc.");
    args.AddOption(&rel_tol, "-rt", "--rel-tol",
            "Relative tolerance for linear solving.");
    args.AddOption(&order, "-o", "--order",
            "Order (degree) of the finite elements.");
    args.AddOption(&lMax, "-l", "--lMax", "Truncation degree for the DtN map.");
    args.AddOption(&ref_levels, "-rs", "--refine-serial",
            "Number of times to refine the mesh uniformly in serial.");
    args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
            "--no-static-condensation", "Enable static condensation.");
    args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
            "--no-partial-assembly", "Enable Partial Assembly.");
    args.AddOption(&device_config, "-d", "--device",
            "Device configuration string, see Device::Configure().");
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

    //Mesh
    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();
    for (int l = 0; l < ref_levels; l++)
    {
        mesh->UniformRefinement();
    }
    mesh->SetAttributes();

    Array<int> attr_cond = mesh->attributes;
    attr_cond.DeleteLast();

    SubMesh mesh_cond(SubMesh::CreateFromDomain(*mesh, attr_cond));

    //FE Space
    int order_phi = order; int order_dphi = order_phi - 1; int order_properties = order - 1; int order_w = 2 * (order - 1); 
    H1_FECollection fec_u(order, dim);
    H1_FECollection fec_phi(order_phi, dim);
    L2_FECollection fec_properties(order_properties, dim);
    L2_FECollection fec_dphi(order_dphi, dim);
    L2_FECollection fec_w(order_w, dim);
    FiniteElementSpace fes_phi(mesh, &fec_phi);
    FiniteElementSpace fes_phi_cond(&mesh_cond, &fec_phi);
    FiniteElementSpace fes_dphi(mesh, &fec_dphi);
    FiniteElementSpace fes_dphi_cond(&mesh_cond, &fec_dphi);
    FiniteElementSpace fes_u_cond(&mesh_cond, &fec_u, 3);
    FiniteElementSpace fes_properties_cond(&mesh_cond, &fec_properties);
    FiniteElementSpace fes_w_cond(&mesh_cond, &fec_w);
    int u_size = fes_u_cond.GetTrueVSize();
    int phi_size = fes_phi.GetTrueVSize();
    cout << "Number of u-unknowns: " << u_size << endl;
    cout << "Number of phi-unknowns: " << phi_size << endl;
    GridFunction u_gf_cond(&fes_u_cond); GridFunction phi_gf(&fes_phi); GridFunction phi_gf_cond(&fes_phi_cond); 
    GridFunction phi0_gf(&fes_phi); GridFunction phi0_gf_cond(&fes_phi_cond); GridFunction dphi0_gf(&fes_dphi); 
    GridFunction dphi0_gf_cond(&fes_dphi_cond); GridFunction w_gf(&fes_w_cond);
    u_gf_cond = 0.0; phi_gf = 0.0; phi_gf_cond = 0.0; phi0_gf = 0.0; phi0_gf_cond = 0.0; dphi0_gf = 0.0; dphi0_gf_cond = 0.0; w_gf = 0.0;
    FunctionCoefficient rho_coeff(rho_func);
    FunctionCoefficient mu_coeff(mu_func);
    FunctionCoefficient lamb_coeff(lamb_func);
    FunctionCoefficient loading_coeff(loading_func);

    Array<int> block_offsets(3);
    block_offsets[0] = 0;
    block_offsets[1] = fes_u_cond.GetVSize();
    block_offsets[2] = fes_phi.GetVSize();
    block_offsets.PartialSum();

    std::cout << "***********************************************************\n";
    std::cout << "dim(u) = " << block_offsets[1] - block_offsets[0] << "\n";
    std::cout << "dim(phi) = " << block_offsets[2] - block_offsets[1] << "\n";
    std::cout << "dim(u+phi) = " << block_offsets.Last() << "\n";
    std::cout << "***********************************************************\n";

    MemoryType mt = device.GetMemoryType();
    BlockVector X(block_offsets, mt), Rhs(block_offsets, mt);

    Array<int> ess_tdof_list;

    Array<int> Earth_body_marker;
    Earth_body_marker = Array<int>(mesh->attributes.Size());
    Earth_body_marker = 1;
    Earth_body_marker[mesh->attributes.Size() - 1] = 0;


    Array<int> bdr_marker;
    auto size = mesh->bdr_attributes.Size();
    bdr_marker = Array<int>(size);
    bdr_marker = 0;
    bdr_marker[size - 2] = 1;


    Array<int> bdr_marker_cond;
    auto size_cond = mesh_cond.bdr_attributes.Size();
    bdr_marker_cond = Array<int>(size_cond);
    bdr_marker_cond = 0;
    bdr_marker_cond[size - 1] = 1;

    //Compute the equilibrium state
    auto DtN = DirichletToNeumannOperator(&fes_phi, lMax);

    ProductCoefficient rhs_coeff(- 4.0 * M_PI * Constants::G, rho_coeff);
    LinearForm b0(&fes_phi);
    b0.AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
    b0.Assemble();

    BilinearForm a0(&fes_phi);
    auto one = ConstantCoefficient(1.0);
    a0.AddDomainIntegrator(new DiffusionIntegrator(one));
    a0.Assemble();

    OperatorPtr A0;
    Vector B0, Phi0;

    a0.FormLinearSystem(ess_tdof_list, phi0_gf, b0, A0, Phi0, B0);
    cout << "Size of linear system: " << A0->Height() << endl;

    auto S = SumOperator(A0.Ptr(), 1, &DtN, 1, false, false);

    GSSmoother M((SparseMatrix &)(*A0));
    //DSmoother M((SparseMatrix &)(*A));

    auto solver0 = CGSolver();
    //auto solver = BiCGSTABSolver();
    solver0.SetOperator(S);
    solver0.SetPreconditioner(M);
    solver0.SetRelTol(rel_tol);
    solver0.SetMaxIter(2000);
    solver0.SetPrintLevel(1);
    solver0.Mult(B0, Phi0);

    a0.RecoverFEMSolution(Phi0, b0, phi0_gf);

    DiscreteLinearOperator Grad(&fes_phi_cond, &fes_dphi_cond);
    Grad.AddDomainInterpolator(new GradInterpolator);
    Grad.Assemble();

    //DiscreteLinearOperator Grad(&fes_phi, &fes_dphi);
    //Grad.AddDomainInterpolator(new GradInterpolator);
    //Grad.Assemble();
    //Grad.Finalize();
    //Grad.Mult(phi0_gf, dphi0_gf);
    //VectorGridFunctionCoefficient dphi0_coeff(&dphi0_gf);
    GridFunctionCoefficient phi0_coeff(&phi0_gf);
    mesh_cond.Transfer(phi0_gf, phi0_gf_cond);
    Grad.Mult(phi0_gf_cond, dphi0_gf_cond);
    VectorGridFunctionCoefficient dphi0_cond_coeff(&dphi0_gf_cond);
    //GradientGridFunctionCoefficient dphi0_cond_coeff(&phi0_gf_cond);
    //dphi0_gf_cond.ProjectCoefficient(&dphi0_cond_coeff);
    GradientVectorGridFunctionCoefficient ddphi0_cond_coeff(&dphi0_gf_cond);
    ScalarVectorProductCoefficient dphi0_sig_cond_coeff(loading_coeff, dphi0_cond_coeff);

    if (visualization)
    {
        char vishost[] = "localhost";
        int  visport   = 19916;
        socketstream sol_sock(vishost, visport);
        sol_sock.precision(8);
        sol_sock << "solution\n" << *mesh << phi0_gf << flush;
    }


    //Coupled problem
    LinearForm *b1(new LinearForm);
    b1->Update(&fes_u_cond, Rhs.GetBlock(0), 0);
    b1->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(dphi0_sig_cond_coeff), bdr_marker_cond); //! by luck
    b1->Assemble();
    b1->SyncAliasMemory(Rhs);

    LinearForm *b2(new LinearForm);
    b2->Update(&fes_phi, Rhs.GetBlock(1), 0);
    b2->AddDomainIntegrator(new BoundaryLFIntegrator(loading_coeff), bdr_marker);
    b2->Assemble();
    b2->SyncAliasMemory(Rhs);

    BilinearForm *a11(new BilinearForm(&fes_u_cond));
    BilinearForm *a22(new BilinearForm(&fes_phi));
    MixedBilinearForm *a12(new MixedBilinearForm(&fes_phi, &fes_u_cond));
    //MixedBilinearForm *a21(new MixedBilinearForm(&fes_u, &fes_phi));
    //FiniteElementSpace* fes_u_ptr = &fes_u_cond;
    //MixedBilinearForm *a21(new MixedBilinearForm(fes_u_ptr, &fes_phi));
    FiniteElementSpace fes_u(mesh, &fec_u, 3);
    //MixedBilinearForm *a21(new MixedBilinearForm(&fes_u_whole, &fes_phi));
    CondTrialMixedBilinearForm *a21(new CondTrialMixedBilinearForm(&fes_u, &fes_u_cond, &fes_phi, &fes_phi_cond));
    
    //if (pa) { a11->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
    a11->AddDomainIntegrator(new ElasticityIntegrator(lamb_coeff, mu_coeff));
    a11->Assemble();
    a11->Finalize();

    ConstantCoefficient c0(1.0 / (4.0 * M_PI * Constants::G)); 
    a22->AddDomainIntegrator(new DiffusionIntegrator(c0));
    a22->Assemble();
    a22->Finalize();

    ProductCoefficient half_rho_coeff(0.5, rho_coeff);
    ProductCoefficient minus_half_rho_coeff(-0.5, rho_coeff);
    //a12->AddDomainIntegrator(new mfemElasticity::DomainVectorGradVectorIntegrator(dphi0_cond_coeff, half_rho_coeff));
    a12->AddDomainIntegrator(new GradProjectionIntegrator(half_rho_coeff, dphi0_cond_coeff, ddphi0_cond_coeff));
    a12->AddDomainIntegrator(new AdvectionProjectionIntegrator(half_rho_coeff, dphi0_cond_coeff, ddphi0_cond_coeff));
    a12->AddDomainIntegrator(new DivVecIntegrator(minus_half_rho_coeff, dphi0_cond_coeff));
    a12->AddDomainIntegrator(new ProjDivIntegrator(minus_half_rho_coeff, dphi0_cond_coeff));

    a12->AddDomainIntegrator(new GradientIntegrator(rho_coeff));
    //a12->AddDomainIntegrator(new mfemElasticity::DomainVectorGradScalarIntegrator(rho_coeff));
    a12->Assemble(); //by luck
    a12->Finalize();


    //ProductCoefficient minus_rho_coeff(-1.0, rho_coeff);
    cout << Earth_body_marker.Size()<<" "<<Earth_body_marker[1] << endl;
    a21->AddDomainIntegrator(new AdvectionScalarIntegrator(rho_coeff), Earth_body_marker);
    //a21->AddDomainIntegrator(new MixedScalarWeakDerivativeIntegrator(minus_rho_coeff), Earth_body_marker);
    //fes_u_ptr = &fes_u;
    a21->Assemble();
    a21->Finalize();



    BlockOperator EGOp(block_offsets);
    BlockDiagonalPreconditioner EGPrec(block_offsets);

    SparseMatrix &A11(a11->SpMat());
    SparseMatrix &A12(a12->SpMat());
    SparseMatrix &A21(a21->SpMat());
    SparseMatrix &A22_0(a22->SpMat());
    auto A22 = SumOperator(&A22_0, 1, &DtN, 1, false, false);


    EGOp.SetBlock(0,0, &A11);
    EGOp.SetBlock(0,1, &A12);
    EGOp.SetBlock(1,0, &A21);
    EGOp.SetBlock(1,1, &A22);


    GSSmoother prec11(A11);
    GSSmoother prec22(A22_0);
    EGOp.SetDiagonalBlock(0, &prec11);
    EGOp.SetDiagonalBlock(1, &prec22);

    GMRESSolver solver;
    solver.SetRelTol(rel_tol);
    solver.SetMaxIter(5000);
    solver.SetOperator(EGOp);
    solver.SetPreconditioner(EGPrec);
    solver.SetPrintLevel(1);
    X = 0.0;
    solver.Mult(Rhs, X);
    if (device.IsEnabled()) { X.HostRead(); }

    if (solver.GetConverged())
    {
        std::cout << "MINRES converged in " << solver.GetNumIterations()
            << " iterations with a residual norm of "
            << solver.GetFinalNorm() << ".\n";
    }
    else
    {
        std::cout << "MINRES did not converge in " << solver.GetNumIterations()
            << " iterations. Residual norm is " << solver.GetFinalNorm()
            << ".\n";
    }

    u_gf_cond.MakeRef(&fes_u_cond, X.GetBlock(0), 0);
    phi_gf.MakeRef(&fes_phi, X.GetBlock(1), 0);
    mesh_cond.Transfer(phi_gf, phi_gf_cond);


    if (visualization)
    {
        char vishost[] = "localhost";
        int  visport   = 19916;
        socketstream u_sock(vishost, visport);
        u_sock.precision(8);
        u_sock << "solution\n" << mesh_cond << u_gf_cond << "window_title 'Velocity'" << endl;
        socketstream phi_sock(vishost, visport);
        phi_sock.precision(8);
        phi_sock << "solution\n" << mesh_cond << phi_gf_cond << "window_title 'Gravity Potential'" << endl;
    }


    {
        ofstream mesh_ofs("data/ex2.mesh");
        mesh_ofs.precision(8);
        mesh_cond.Print(mesh_ofs);

        ofstream u_ofs("data/ex2_u.gf");
        u_ofs.precision(8);
        u_gf_cond.Save(u_ofs);

        ofstream phi_ofs("data/ex2_phi.gf");
        phi_ofs.precision(8);
        phi_gf_cond.Save(phi_ofs);
    }



    delete mesh, b1, b2, a11, a12, a21, a22;

    return 0;
}

real_t rho_func(const Vector &coord)
{
    real_t r = coord.Norml2();
    if (r > Constants::R){
        return 0.0;
    } else{
        real_t r_norm = r / Constants::R;
        //real_t theta = acos(coord[2] / r); // polar angle
        //real_t phi = atan2(coord[1], coord[0]); // azimuthal angle
        real_t rho_surface = 2.6e3;  // Pa
        real_t rho_center = 1.3e4;    // Pa
        real_t base_rho =  rho_center + (rho_surface - rho_center) * r_norm;
        return base_rho;
    }
}


real_t mu_func(const Vector &coord)
{
    real_t r = coord.Norml2();
    real_t r_norm = r / 6371e3;
    real_t theta = acos(coord[2] / r); // polar angle
    real_t phi = atan2(coord[1], coord[0]); // azimuthal angle
    real_t mu_surface = 70e9;  // Pa
    real_t mu_center = 140e9;    // Pa
    real_t base_mu =  mu_center + (mu_surface - mu_center) * r_norm;
    real_t polar_perturb = 0.015 * (1.0 + cos(2.0 * theta));
    real_t azimuthal_perturb = 0.05 * sin(2.0 * phi);
    return base_mu * (1.0 + polar_perturb) * (1.0 + azimuthal_perturb);
}

real_t lamb_func(const Vector &coord)
{
    real_t r = coord.Norml2();
    real_t r_norm = r / 6371e3;
    real_t theta = acos(coord[2] / r);
    real_t phi = atan2(coord[1], coord[0]);
    real_t lamb_surface = 100e9;
    real_t lamb_center = 300e9;   
    real_t base_lamb = lamb_center + (lamb_surface - lamb_center) * r_norm;
    real_t polar_perturb = 0.015 * (1.0 + cos(2.0 * theta));
    real_t azimuthal_perturb = 0.05 * sin(2.0 * phi);
    return base_lamb * (1.0 + polar_perturb) * (1.0 + azimuthal_perturb);
}

real_t loading_func(const Vector &coord)
{
    real_t r = coord.Norml2();
    real_t theta = acos(coord[2] / r);
    real_t phi = atan2(coord[1], coord[0]);
    // Max loading at poles (glaciers): e.g., 10 MPa (~1 km ice)
    const real_t polar_load = 10e6;
    // Equatorial loading (oceans): e.g., 1 MPa (~100 m water depth)
    const real_t equator_load = 1e6;
    real_t base_load = equator_load + (polar_load - equator_load) / 2.0 * cos(2.0 * theta);
    real_t azimuthal_perturb = 0.2 * sin(2.0 * phi);
    return base_load * (1.0 + azimuthal_perturb);
}
